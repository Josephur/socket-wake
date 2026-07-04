// SPDX-License-Identifier: Apache-2.0
//! Weights file parser.
//!
//! SAFETY: this module contains `unsafe` to reinterpret the parsed bytes
//! as typed slices. The pointers are bounded by the layer header (which
//! we read before constructing them) and tied to the original `Weights`
//! borrow. We isolate the unsafe to this one module so the crate-level
//! `#![deny(unsafe_code)]` still gates the rest of the runtime.
#![allow(unsafe_code)]
//!
//! Format (SWWT v2):
//!
//! ```text
//! magic:        [u8; 4]   = b"SWWT"
//! version:      u16 LE    = 2
//! n_layers:     u16 LE
//! input_scale:  f32 LE    -- mel i8 -> conv input: q = clamp(rint(v * input_scale))
//! logit_thr:    i8        -- state machine fires on (q_target - q_not) >= thr
//! hold:         u8        -- consecutive above-threshold inferences to fire
//! refractory:   u16 LE    -- inference steps of lockout after a fire
//! for each layer:
//!   kind:       u8        (0 = depthwise, 1 = pointwise, 2 = dense, 3 = conv2d)
//!   in_h:       u16 LE
//!   in_w:       u16 LE
//!   in_c:       u16 LE
//!   out_c:      u16 LE
//!   k_h:        u8
//!   k_w:        u8
//!   stride:     u8
//!   relu:       u8        (0 or 1)
//!   scale:      f32 LE    -- requant multiplier M = s_in * s_w / s_out
//!   bias:       [f32 LE; out_c]  -- B = bias / s_out, added AFTER scaling
//!   weights:    [i8; <varies by kind>]
//! ```
//!
//! Requantization contract (must bit-match the Python exporter's integer
//! reference, `socket_wake.int8_ref`):
//!
//! ```text
//! out = clamp(rint(acc as f32 * scale + bias[o]), lo, 127)
//! lo = 0 if relu else -127; rint = round half to even
//! ```
//!
//! Weight layouts:
//!   depthwise: (in_c, k_h, k_w)               -- one kxk filter per channel
//!   pointwise: (in_c, out_c)                  -- shared 1x1 conv
//!   dense:     (in_c, out_c)                  -- if in_h*in_w > 1 the layer
//!              sums over all spatial positions with shared weights, i.e.
//!              global-average-pool is folded into the requant scale
//!   conv2d:    (out_c, k_h, k_w, in_c)        -- general conv (OHWI)

use core::convert::TryInto;

#[derive(Debug, PartialEq)]
pub enum WeightError {
    Truncated,
    BadMagic,
    UnsupportedVersion(u16),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerKind {
    Depthwise = 0,
    Pointwise = 1,
    Dense = 2,
    Conv2d = 3,
}

impl LayerKind {
    fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(LayerKind::Depthwise),
            1 => Some(LayerKind::Pointwise),
            2 => Some(LayerKind::Dense),
            3 => Some(LayerKind::Conv2d),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LayerDesc {
    pub kind: LayerKind,
    pub in_h: u16,
    pub in_w: u16,
    pub in_c: u16,
    pub out_c: u16,
    pub k_h: u8,
    pub k_w: u8,
    pub stride: u8,
    pub relu: bool,
    pub scale: f32,
}

/// Model-level parameters carried in the file header.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelParams {
    /// Input requant: q = clamp(rint(mel_i8 * input_scale), -127, 127).
    pub input_scale: f32,
    /// State-machine threshold on the quantized logit margin.
    pub logit_thr: i8,
    /// Consecutive above-threshold inferences required to fire.
    pub hold: u8,
    /// Inference steps of lockout after a fire.
    pub refractory: u16,
}

#[derive(Debug, PartialEq)]
pub struct Weights<'a> {
    bytes: &'a [u8],
    n_layers: u16,
    params: ModelParams,
}

impl<'a> Weights<'a> {
    pub const MAGIC: &'static [u8; 4] = b"SWWT";
    pub const VERSION: u16 = 2;
    pub const HEADER_LEN: usize = 4 + 2 + 2 + 4 + 1 + 1 + 2; // = 16

    pub fn parse(bytes: &'a [u8]) -> Result<Self, WeightError> {
        // Validate magic + version before demanding the full v2 header so
        // old v1 blobs (8-byte header) report UnsupportedVersion, not
        // Truncated.
        if bytes.len() < 6 {
            return Err(WeightError::Truncated);
        }
        if &bytes[0..4] != Self::MAGIC {
            return Err(WeightError::BadMagic);
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != Self::VERSION {
            return Err(WeightError::UnsupportedVersion(version));
        }
        if bytes.len() < Self::HEADER_LEN {
            return Err(WeightError::Truncated);
        }
        let n_layers = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        let input_scale = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let logit_thr = bytes[12] as i8;
        let hold = bytes[13];
        let refractory = u16::from_le_bytes(bytes[14..16].try_into().unwrap());
        Ok(Self {
            bytes,
            n_layers,
            params: ModelParams { input_scale, logit_thr, hold, refractory },
        })
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers as usize
    }

    pub fn params(&self) -> ModelParams {
        self.params
    }

    pub fn raw(&self) -> &'a [u8] {
        self.bytes
    }
}

/// Number of i8 weights a layer of this shape carries.
pub fn weight_count(desc: &LayerDesc) -> usize {
    let (in_c, out_c) = (desc.in_c as usize, desc.out_c as usize);
    let (k_h, k_w) = (desc.k_h as usize, desc.k_w as usize);
    match desc.kind {
        LayerKind::Depthwise => in_c * k_h * k_w,
        LayerKind::Pointwise => in_c * out_c,
        LayerKind::Dense => in_c * out_c,
        LayerKind::Conv2d => out_c * k_h * k_w * in_c,
    }
}

/// Iterator over layers. Each `next()` returns the parsed `LayerDesc` plus
/// borrowed slices into the original byte buffer for bias (f32 LE) and
/// weights (i8).
pub struct Layers<'a> {
    cursor: usize,
    n_layers: u16,
    seen: u16,
    bytes: &'a [u8],
}

#[derive(Debug)]
pub enum LayerIterError {
    Truncated,
    UnknownKind(u8),
}

impl<'a> Layers<'a> {
    pub fn new(w: &'a Weights<'a>) -> Self {
        Self {
            cursor: Weights::HEADER_LEN,
            n_layers: w.n_layers,
            seen: 0,
            bytes: w.bytes,
        }
    }
}

impl<'a> Iterator for Layers<'a> {
    type Item = Result<(LayerDesc, &'a [u8], &'a [i8]), LayerIterError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.seen >= self.n_layers {
            return None;
        }
        self.seen += 1;
        let start = self.cursor;

        const LAYER_HEADER_LEN: usize = 1 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 1 + 4;
        if self.bytes.len() < start + LAYER_HEADER_LEN {
            return Some(Err(LayerIterError::Truncated));
        }
        let kind_byte = self.bytes[start];
        let kind = match LayerKind::from_u8(kind_byte) {
            Some(k) => k,
            None => return Some(Err(LayerIterError::UnknownKind(kind_byte))),
        };
        let in_h = u16::from_le_bytes(self.bytes[start + 1..start + 3].try_into().unwrap());
        let in_w = u16::from_le_bytes(self.bytes[start + 3..start + 5].try_into().unwrap());
        let in_c = u16::from_le_bytes(self.bytes[start + 5..start + 7].try_into().unwrap());
        let out_c = u16::from_le_bytes(self.bytes[start + 7..start + 9].try_into().unwrap());
        let k_h = self.bytes[start + 9];
        let k_w = self.bytes[start + 10];
        let stride = self.bytes[start + 11];
        let relu = self.bytes[start + 12] != 0;
        let scale_bytes: [u8; 4] = self.bytes[start + 13..start + 17].try_into().unwrap();
        let scale = f32::from_le_bytes(scale_bytes);
        let mut cursor = start + LAYER_HEADER_LEN;

        let bias_len_bytes = out_c as usize * 4;
        if self.bytes.len() < cursor + bias_len_bytes {
            return Some(Err(LayerIterError::Truncated));
        }
        let bias_bytes = &self.bytes[cursor..cursor + bias_len_bytes];
        cursor += bias_len_bytes;

        let desc = LayerDesc { kind, in_h, in_w, in_c, out_c, k_h, k_w, stride, relu, scale };
        let n_weights = weight_count(&desc);
        if self.bytes.len() < cursor + n_weights {
            return Some(Err(LayerIterError::Truncated));
        }
        let weights_bytes = &self.bytes[cursor..cursor + n_weights];
        self.cursor = cursor + n_weights;

        // SAFETY: i8 and u8 share alignment 1; pointer arithmetic stays
        // inside the parent slice; the lifetime is tied to the `Weights`
        // borrow.
        let weights: &'a [i8] = unsafe {
            core::slice::from_raw_parts(
                weights_bytes.as_ptr() as *const i8,
                n_weights,
            )
        };
        Some(Ok((desc, bias_bytes, weights)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(n_layers: u16) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"SWWT");
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&n_layers.to_le_bytes());
        b.extend_from_slice(&1.0f32.to_le_bytes()); // input_scale
        b.push(20u8);                               // logit_thr
        b.push(2u8);                                // hold
        b.extend_from_slice(&32u16.to_le_bytes());  // refractory
        b
    }

    #[test]
    fn rejects_truncated_input() {
        assert_eq!(Weights::parse(&[0u8; 3]), Err(WeightError::Truncated));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = vec![0u8; 16];
        bytes[0..4].copy_from_slice(b"NOPE");
        assert_eq!(Weights::parse(&bytes), Err(WeightError::BadMagic));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = vec![0u8; 16];
        bytes[0..4].copy_from_slice(b"SWWT");
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        assert_eq!(
            Weights::parse(&bytes),
            Err(WeightError::UnsupportedVersion(99))
        );
    }

    #[test]
    fn parses_minimal_zero_layer_header() {
        let bytes = header(0);
        let w = Weights::parse(&bytes).expect("parse");
        assert_eq!(w.n_layers(), 0);
        let p = w.params();
        assert_eq!(p.input_scale, 1.0);
        assert_eq!(p.logit_thr, 20);
        assert_eq!(p.hold, 2);
        assert_eq!(p.refractory, 32);
    }

    #[test]
    fn parses_one_dense_layer() {
        let mut bytes = header(1);
        bytes.push(2); // dense
        for v in [1u16, 1, 4, 2] { bytes.extend_from_slice(&v.to_le_bytes()); }
        bytes.push(1); // k_h
        bytes.push(1); // k_w
        bytes.push(1); // stride
        bytes.push(0); // relu
        bytes.extend_from_slice(&0.5f32.to_le_bytes()); // scale
        for b in [1.0f32, -2.0] { bytes.extend_from_slice(&b.to_le_bytes()); }
        for w in [1i8, 2, 3, 4, 5, 6, 7, 8] { bytes.push(w as u8); }
        let w = Weights::parse(&bytes).expect("parse");
        let mut layers = Layers::new(&w);
        let (desc, bias, weights) = layers.next().unwrap().unwrap();
        assert_eq!(desc.kind, LayerKind::Dense);
        assert_eq!((desc.in_c, desc.out_c), (4, 2));
        assert!(!desc.relu);
        assert_eq!(desc.scale, 0.5);
        assert_eq!(bias.len(), 8);
        assert_eq!(weights, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(layers.next().is_none());
    }

    #[test]
    fn truncated_layer_reports_error() {
        let mut bytes = header(1);
        bytes.push(2);
        let w = Weights::parse(&bytes).expect("parse");
        let mut layers = Layers::new(&w);
        assert!(matches!(layers.next(), Some(Err(LayerIterError::Truncated))));
    }
}
