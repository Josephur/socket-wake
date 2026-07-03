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
//! Format:
//!
//! ```text
//! magic:        [u8; 4]   = b"SWWT"
//! version:      u16 LE    = 1
//! n_layers:     u16 LE
//! for each layer:
//!   kind:       u8        (0 = depthwise, 1 = pointwise, 2 = dense)
//!   in_h:       u16 LE
//!   in_w:       u16 LE
//!   in_c:       u16 LE
//!   out_c:      u16 LE
//!   k_h:        u8
//!   k_w:        u8
//!   scale:      f32 LE
//!   bias:       [i32 LE; out_c]
//!   weights:    [i8; <varies by kind>]
//! ```

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
}

impl LayerKind {
    fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(LayerKind::Depthwise),
            1 => Some(LayerKind::Pointwise),
            2 => Some(LayerKind::Dense),
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
    pub scale: f32,
}

pub struct Weights<'a> {
    bytes: &'a [u8],
    n_layers: u16,
}

impl<'a> Weights<'a> {
    pub const MAGIC: &'static [u8; 4] = b"SWWT";
    pub const VERSION: u16 = 1;
    pub const HEADER_LEN: usize = 8;

    pub fn parse(bytes: &'a [u8]) -> Result<Self, WeightError> {
        if bytes.len() < Self::HEADER_LEN {
            return Err(WeightError::Truncated);
        }
        if &bytes[0..4] != Self::MAGIC {
            return Err(WeightError::BadMagic);
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != Self::VERSION {
            return Err(WeightError::UnsupportedVersion(version));
        }
        let n_layers = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        Ok(Self { bytes, n_layers })
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers as usize
    }

    pub fn raw(&self) -> &'a [u8] {
        self.bytes
    }
}

/// Iterator over layers. Each `next()` returns the parsed `LayerDesc` plus
/// borrowed slices into the original byte buffer for bias and weights.
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

        const LAYER_HEADER_LEN: usize = 1 + 2 + 2 + 2 + 2 + 1 + 1 + 4;
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
        let scale_bytes: [u8; 4] = self.bytes[start + 11..start + 15].try_into().unwrap();
        let scale = f32::from_le_bytes(scale_bytes);
        let mut cursor = start + LAYER_HEADER_LEN;

        let bias_len_bytes = out_c as usize * 4;
        if self.bytes.len() < cursor + bias_len_bytes {
            return Some(Err(LayerIterError::Truncated));
        }
        let bias_bytes = &self.bytes[cursor..cursor + bias_len_bytes];
        cursor += bias_len_bytes;

        let weight_count: usize = match kind {
            LayerKind::Depthwise => (in_h as usize) * (in_w as usize) * (in_c as usize)
                                    * (k_h as usize) * (k_w as usize),
            LayerKind::Pointwise => (in_h as usize) * (in_w as usize) * (in_c as usize)
                                    * (out_c as usize),
            LayerKind::Dense => (in_c as usize) * (out_c as usize),
        };
        if self.bytes.len() < cursor + weight_count {
            return Some(Err(LayerIterError::Truncated));
        }
        let weights_bytes = &self.bytes[cursor..cursor + weight_count];
        self.cursor = cursor + weight_count;

        let desc = LayerDesc { kind, in_h, in_w, in_c, out_c, k_h, k_w, scale };
        // SAFETY: i8 and u8 share alignment 1; pointer arithmetic stays
        // inside the parent slice; the lifetime is tied to the `Weights`
        // borrow.
        let weights: &'a [i8] = unsafe {
            core::slice::from_raw_parts(
                weights_bytes.as_ptr() as *const i8,
                weight_count,
            )
        };
        Some(Ok((desc, bias_bytes, weights)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"SWWT");
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        let w = Weights::parse(&bytes).expect("parse");
        assert_eq!(w.n_layers(), 0);
    }
}