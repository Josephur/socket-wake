// SPDX-License-Identifier: Apache-2.0
//! INT8 DS-CNN-L inference kernel.
//!
//! The runtime accepts a single mel frame (INT8, length N_MELS) and a
//! parsed `Weights`. It runs:
//!   stem conv (1 -> 64) -> 4x DS blocks (64 -> 64) -> global avg pool ->
//!   dense (64 -> n_classes)
//!
//! All math is INT8 with INT32 accumulators, requantized per layer using
//! the per-layer `scale` from the weights file.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::weights::{self, LayerDesc, LayerKind, Weights};

#[derive(Debug, PartialEq)]
pub enum CnnError {
    /// A layer's declared shape doesn't match what the prior layer produced.
    ShapeMismatch,
    /// A configuration we don't support (e.g. kernel > 3x3 in v1).
    UnsupportedConfig(&'static str),
}

/// Output of `Cnn::run`: class logits (one per class). For binary
/// wake-word detection v1 expects exactly 2 classes (target vs. not-target).
pub type Logits = Vec<i8>;

pub struct Cnn;

impl Cnn {
    /// Run forward pass. Returns per-class logits or an error.
    pub fn run(input: &[i8], weights: &Weights<'_>) -> Result<Logits, CnnError> {
        // The DS-CNN-L architecture is fixed at compile time: a stem + 4 DS
        // blocks + dense. We let weights still carry per-layer descs so the
        // model format is general, but validate that what's there matches
        // our expected topology.
        let mut layers = weights::Layers::new(weights);
        let mut activations: Vec<i8> = input.to_vec();

        while let Some(layer) = layers.next() {
            let layer = layer.map_err(|_| CnnError::ShapeMismatch)?;
            activations = apply_layer(layer, &activations)?;
        }
        Ok(activations)
    }
}

fn apply_layer(
    layer: (LayerDesc, &[u8], &[i8]),
    input: &[i8],
) -> Result<Vec<i8>, CnnError> {
    let (desc, bias_bytes, layer_weights) = layer;
    let bias = decode_bias(bias_bytes, desc.out_c as usize);

    match desc.kind {
        LayerKind::Dense => dense(input, &bias, layer_weights, desc),
        LayerKind::Pointwise => conv2d_pw(input, &bias, layer_weights, desc),
        LayerKind::Depthwise => conv2d_dw(input, &bias, layer_weights, desc),
    }
}

/// Decode `bias_bytes` (length = `out_c * 4` little-endian i32) into a
/// stack buffer for v1 (out_c <= 256). v2 callers can swap this for a heap
/// allocation; for our tiny models the stack copy is fine.
fn decode_bias(bytes: &[u8], out_c: usize) -> [i32; 256] {
    let mut buf = [0i32; 256];
    for i in 0..out_c {
        let b: [u8; 4] = bytes[i * 4..i * 4 + 4].try_into().unwrap();
        buf[i] = i32::from_le_bytes(b);
    }
    buf
}

fn requant(x: i32, scale: f32) -> i8 {
    let q = (x as f32 * scale).round();
    if q > 127.0 { 127 } else if q < -127.0 { -127 } else { q as i8 }
}

fn dense(input: &[i8], bias: &[i32], weights: &[i8], desc: LayerDesc) -> Result<Vec<i8>, CnnError> {
    if input.len() != desc.in_c as usize {
        return Err(CnnError::ShapeMismatch);
    }
    let mut out = Vec::with_capacity(desc.out_c as usize);
    for j in 0..desc.out_c as usize {
        let mut acc: i32 = bias[j];
        for i in 0..desc.in_c as usize {
            acc += (input[i] as i32) * (weights[i * desc.out_c as usize + j] as i32);
        }
        out.push(requant(acc, desc.scale));
    }
    Ok(out)
}

fn conv2d_pw(
    input: &[i8],
    bias: &[i32],
    weights: &[i8],
    desc: LayerDesc,
) -> Result<Vec<i8>, CnnError> {
    // Pointwise 1x1 conv: input shape (in_h, in_w, in_c), output (in_h, in_w, out_c).
    if desc.k_h != 1 || desc.k_w != 1 {
        return Err(CnnError::UnsupportedConfig("pointwise kernel must be 1x1"));
    }
    let hw = (desc.in_h as usize) * (desc.in_w as usize);
    if input.len() != hw * (desc.in_c as usize) {
        return Err(CnnError::ShapeMismatch);
    }
    let mut out = vec![0i8; hw * (desc.out_c as usize)];
    for o in 0..desc.out_c as usize {
        for ij in 0..hw {
            let mut acc = bias[o];
            for i in 0..desc.in_c as usize {
                let w = weights[((ij * desc.in_c as usize) + i) * desc.out_c as usize + o];
                acc += (input[ij * desc.in_c as usize + i] as i32) * (w as i32);
            }
            out[ij * desc.out_c as usize + o] = requant(acc, desc.scale);
        }
    }
    Ok(out)
}

fn conv2d_dw(
    input: &[i8],
    bias: &[i32],
    weights: &[i8],
    desc: LayerDesc,
) -> Result<Vec<i8>, CnnError> {
    // Depthwise conv: one filter per input channel. We support kxk with
    // same padding (k=3 only in v1). For an input (in_h, in_w, in_c) we
    // produce (in_h, in_w, in_c).
    if desc.k_h != desc.k_w || desc.k_h != 3 {
        return Err(CnnError::UnsupportedConfig("depthwise kernel must be 3x3"));
    }
    if desc.out_c != desc.in_c {
        return Err(CnnError::ShapeMismatch);
    }
    let h = desc.in_h as usize;
    let w = desc.in_w as usize;
    if input.len() != h * w * desc.in_c as usize {
        return Err(CnnError::ShapeMismatch);
    }
    let k = 3usize;
    let mut out = vec![0i8; h * w * desc.in_c as usize];
    // weights laid out as (in_c, k, k) so per-channel kernel is contiguous.
    for c in 0..desc.in_c as usize {
        let base_w = c * k * k;
        for y in 0..h {
            for x in 0..w {
                let mut acc = bias[c];
                for ky in 0..k {
                    for kx in 0..k {
                        let sy = y as isize + ky as isize - 1;
                        let sx = x as isize + kx as isize - 1;
                        if sy < 0 || sy >= h as isize || sx < 0 || sx >= w as isize {
                            continue;
                        }
                        let in_idx = (sy as usize) * w * desc.in_c as usize
                                   + (sx as usize) * desc.in_c as usize
                                   + c;
                        let w_idx = base_w + ky * k + kx;
                        acc += (input[in_idx] as i32) * (weights[w_idx] as i32);
                    }
                }
                let out_idx = y * w * desc.in_c as usize + x * desc.in_c as usize + c;
                out[out_idx] = requant(acc, desc.scale);
            }
        }
    }
    Ok(out)
}

// (weights module is imported above.)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{LayerKind, Weights};

    fn empty_blob() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"SWWT");
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes());
        b
    }

    fn blob_with_layers(layers: &[(u8, u16, u16, u16, u16, u8, u8, f32, &[i32], &[i8])]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"SWWT");
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&(layers.len() as u16).to_le_bytes());
        for (kind, ih, iw, ic, oc, kh, kw, scale, bias, weights) in layers {
            b.push(*kind);
            b.extend_from_slice(&ih.to_le_bytes());
            b.extend_from_slice(&iw.to_le_bytes());
            b.extend_from_slice(&ic.to_le_bytes());
            b.extend_from_slice(&oc.to_le_bytes());
            b.push(*kh);
            b.push(*kw);
            b.extend_from_slice(&scale.to_le_bytes());
            for v in *bias { b.extend_from_slice(&v.to_le_bytes()); }
            for v in *weights { b.push(*v as u8); }
        }
        b
    }

    #[test]
    fn cnn_with_zero_layers_returns_input() {
        let blob = empty_blob();
        let w = Weights::parse(&blob).unwrap();
        let input = vec![1i8, 2, 3, 4];
        let out = Cnn::run(&input, &w).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn dense_identity_layer_returns_input() {
        // 4x4 dense identity (scale=1).
        let mut weights = [0i8; 16];
        for i in 0..4 { weights[i * 4 + i] = 1; }
        let blob = blob_with_layers(&[(
            LayerKind::Dense as u8, 1, 1, 4, 4, 1, 1, 1.0,
            &[0, 0, 0, 0],
            &weights,
        )]);
        let w = Weights::parse(&blob).unwrap();
        let out = Cnn::run(&[1, 2, 3, 4], &w).unwrap();
        assert_eq!(out, vec![1, 2, 3, 4]);
    }

    #[test]
    fn dense_permutes_columns() {
        // 4x4 dense: out = [in[3], in[0], in[1], in[2]] (rotation).
        let mut weights = [0i8; 16];
        weights[0 * 4 + 0] = 0; weights[1 * 4 + 0] = 0;
        weights[2 * 4 + 0] = 0; weights[3 * 4 + 0] = 1;
        weights[0 * 4 + 1] = 1; weights[1 * 4 + 1] = 0;
        weights[2 * 4 + 1] = 0; weights[3 * 4 + 1] = 0;
        weights[0 * 4 + 2] = 0; weights[1 * 4 + 2] = 1;
        weights[2 * 4 + 2] = 0; weights[3 * 4 + 2] = 0;
        weights[0 * 4 + 3] = 0; weights[1 * 4 + 3] = 0;
        weights[2 * 4 + 3] = 1; weights[3 * 4 + 3] = 0;
        let blob = blob_with_layers(&[(
            LayerKind::Dense as u8, 1, 1, 4, 4, 1, 1, 1.0,
            &[0, 0, 0, 0],
            &weights,
        )]);
        let w = Weights::parse(&blob).unwrap();
        let out = Cnn::run(&[1, 2, 3, 4], &w).unwrap();
        assert_eq!(out, vec![4, 1, 2, 3]);
    }
}