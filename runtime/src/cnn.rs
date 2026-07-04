// SPDX-License-Identifier: Apache-2.0
//! INT8 CNN inference kernels for the SWWT v2 layer graph.
//!
//! The runtime accepts an INT8 activation tensor in HWC layout
//! (in_h, in_w, in_c) and a parsed `Weights`, and applies each layer in
//! file order. For KWSConvNet the graph is:
//!   conv2d 3x3 s2 (1 -> 16) -> dw 3x3 s2 -> pw (16 -> 32) ->
//!   dw 3x3 s2 -> pw (32 -> 32) -> dense (GAP folded, 32 -> 2 logits)
//!
//! All math is INT8 x INT8 -> INT32 accumulation, requantized per layer:
//!
//!   out = clamp(rint(acc as f32 * scale + bias[o]), lo, 127)
//!   lo = 0 if relu else -127; rint = round half to even
//!
//! This must stay bit-identical to `python/socket_wake/int8_ref.py`, which
//! generates the parity vectors checked in `tests/parity.rs`.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::weights::{self, LayerDesc, LayerKind, Weights};

#[derive(Debug, PartialEq)]
pub enum CnnError {
    /// A layer's declared shape doesn't match what the prior layer produced.
    ShapeMismatch,
    /// A configuration we don't support (e.g. even kernels).
    UnsupportedConfig(&'static str),
}

/// Output of `Cnn::run`: class logits, quantized at the export-chosen
/// logit scale. For binary wake-word detection: [not-target, target].
pub type Logits = Vec<i8>;

pub struct Cnn;

impl Cnn {
    /// Run the forward pass. `input` is HWC, matching the first layer's
    /// declared (in_h, in_w, in_c). Returns per-class logits or an error.
    pub fn run(input: &[i8], weights: &Weights<'_>) -> Result<Logits, CnnError> {
        let mut layers = weights::Layers::new(weights);
        let mut activations: Vec<i8> = input.to_vec();

        for layer in &mut layers {
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
    if bias_bytes.len() != desc.out_c as usize * 4 {
        return Err(CnnError::ShapeMismatch);
    }
    let bias = decode_bias(bias_bytes, desc.out_c as usize);

    match desc.kind {
        LayerKind::Dense => dense(input, &bias, layer_weights, desc),
        LayerKind::Pointwise => conv2d_pw(input, &bias, layer_weights, desc),
        LayerKind::Depthwise => conv2d_dw(input, &bias, layer_weights, desc),
        LayerKind::Conv2d => conv2d(input, &bias, layer_weights, desc),
    }
}

/// Decode `bias_bytes` (length = `out_c * 4`, little-endian f32, already in
/// output-scale units) into a stack buffer for v1 (out_c <= 256).
fn decode_bias(bytes: &[u8], out_c: usize) -> [f32; 256] {
    let mut buf = [0.0f32; 256];
    for i in 0..out_c {
        let b: [u8; 4] = bytes[i * 4..i * 4 + 4].try_into().unwrap();
        buf[i] = f32::from_le_bytes(b);
    }
    buf
}

/// Requantize an INT32 accumulator to INT8. `bias` is pre-divided by the
/// output scale so the whole affine step is one f32 multiply-add. The
/// operation order and round-half-to-even must match int8_ref.py exactly.
#[inline]
fn requant(acc: i32, scale: f32, bias: f32, relu: bool) -> i8 {
    let q = (acc as f32 * scale + bias).round_ties_even();
    let lo = if relu { 0.0 } else { -127.0 };
    if q > 127.0 { 127 } else if q < lo { lo as i8 } else { q as i8 }
}

/// Spatial output size for a `same`-padded conv: pad = (k-1)/2 on both
/// sides, out = floor((in + 2*pad - k) / stride) + 1.
#[inline]
fn out_dim(in_d: usize, k: usize, stride: usize) -> usize {
    let pad = (k - 1) / 2;
    (in_d + 2 * pad - k) / stride + 1
}

/// Dense layer over an (in_h, in_w, in_c) tensor with weights shared
/// across spatial positions: acc_j = sum_{hw,i} x[hw,i] * w[i,j]. With
/// in_h*in_w == 1 this is a plain matmul; with a larger spatial extent it
/// is global-average-pooling folded into the requant scale (the exporter
/// divides `scale` by h*w).
fn dense(input: &[i8], bias: &[f32], weights: &[i8], desc: LayerDesc) -> Result<Vec<i8>, CnnError> {
    let hw = desc.in_h as usize * desc.in_w as usize;
    let in_c = desc.in_c as usize;
    let out_c = desc.out_c as usize;
    if input.len() != hw * in_c {
        return Err(CnnError::ShapeMismatch);
    }
    let mut out = Vec::with_capacity(out_c);
    for j in 0..out_c {
        let mut acc: i32 = 0;
        for p in 0..hw {
            let row = &input[p * in_c..(p + 1) * in_c];
            for (i, &x) in row.iter().enumerate() {
                acc += (x as i32) * (weights[i * out_c + j] as i32);
            }
        }
        out.push(requant(acc, desc.scale, bias[j], desc.relu));
    }
    Ok(out)
}

/// Pointwise 1x1 conv with shared weights, layout (in_c, out_c):
/// input (h, w, in_c) -> output (h, w, out_c). Stride 1 only.
fn conv2d_pw(
    input: &[i8],
    bias: &[f32],
    weights: &[i8],
    desc: LayerDesc,
) -> Result<Vec<i8>, CnnError> {
    if desc.k_h != 1 || desc.k_w != 1 {
        return Err(CnnError::UnsupportedConfig("pointwise kernel must be 1x1"));
    }
    if desc.stride != 1 {
        return Err(CnnError::UnsupportedConfig("pointwise stride must be 1"));
    }
    let hw = (desc.in_h as usize) * (desc.in_w as usize);
    let in_c = desc.in_c as usize;
    let out_c = desc.out_c as usize;
    if input.len() != hw * in_c {
        return Err(CnnError::ShapeMismatch);
    }
    let mut out = vec![0i8; hw * out_c];
    for p in 0..hw {
        let row = &input[p * in_c..(p + 1) * in_c];
        for o in 0..out_c {
            let mut acc: i32 = 0;
            for (i, &x) in row.iter().enumerate() {
                acc += (x as i32) * (weights[i * out_c + o] as i32);
            }
            out[p * out_c + o] = requant(acc, desc.scale, bias[o], desc.relu);
        }
    }
    Ok(out)
}

/// Depthwise conv, one kxk filter per channel, weights (in_c, k_h, k_w),
/// `same` padding, stride 1 or 2: input (h, w, c) -> (out_h, out_w, c).
fn conv2d_dw(
    input: &[i8],
    bias: &[f32],
    weights: &[i8],
    desc: LayerDesc,
) -> Result<Vec<i8>, CnnError> {
    if desc.k_h != desc.k_w || desc.k_h % 2 == 0 {
        return Err(CnnError::UnsupportedConfig("depthwise kernel must be odd square"));
    }
    if desc.out_c != desc.in_c {
        return Err(CnnError::ShapeMismatch);
    }
    let h = desc.in_h as usize;
    let w = desc.in_w as usize;
    let c = desc.in_c as usize;
    if input.len() != h * w * c {
        return Err(CnnError::ShapeMismatch);
    }
    let k = desc.k_h as usize;
    let s = desc.stride as usize;
    let pad = (k - 1) / 2;
    let out_h = out_dim(h, k, s);
    let out_w = out_dim(w, k, s);
    let mut out = vec![0i8; out_h * out_w * c];
    for ch in 0..c {
        let base_w = ch * k * k;
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc: i32 = 0;
                for ky in 0..k {
                    for kx in 0..k {
                        let sy = (oy * s + ky) as isize - pad as isize;
                        let sx = (ox * s + kx) as isize - pad as isize;
                        if sy < 0 || sy >= h as isize || sx < 0 || sx >= w as isize {
                            continue;
                        }
                        let in_idx = (sy as usize) * w * c + (sx as usize) * c + ch;
                        acc += (input[in_idx] as i32)
                            * (weights[base_w + ky * k + kx] as i32);
                    }
                }
                out[oy * out_w * c + ox * c + ch] =
                    requant(acc, desc.scale, bias[ch], desc.relu);
            }
        }
    }
    Ok(out)
}

/// General conv, weights (out_c, k_h, k_w, in_c) = OHWI, `same` padding,
/// stride 1 or 2: input (h, w, in_c) -> (out_h, out_w, out_c). Used for
/// the stem (in_c = 1).
fn conv2d(
    input: &[i8],
    bias: &[f32],
    weights: &[i8],
    desc: LayerDesc,
) -> Result<Vec<i8>, CnnError> {
    if desc.k_h != desc.k_w || desc.k_h % 2 == 0 {
        return Err(CnnError::UnsupportedConfig("conv2d kernel must be odd square"));
    }
    let h = desc.in_h as usize;
    let w = desc.in_w as usize;
    let in_c = desc.in_c as usize;
    let out_c = desc.out_c as usize;
    if input.len() != h * w * in_c {
        return Err(CnnError::ShapeMismatch);
    }
    let k = desc.k_h as usize;
    let s = desc.stride as usize;
    let pad = (k - 1) / 2;
    let out_h = out_dim(h, k, s);
    let out_w = out_dim(w, k, s);
    let mut out = vec![0i8; out_h * out_w * out_c];
    for o in 0..out_c {
        let base_w = o * k * k * in_c;
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc: i32 = 0;
                for ky in 0..k {
                    let sy = (oy * s + ky) as isize - pad as isize;
                    if sy < 0 || sy >= h as isize {
                        continue;
                    }
                    for kx in 0..k {
                        let sx = (ox * s + kx) as isize - pad as isize;
                        if sx < 0 || sx >= w as isize {
                            continue;
                        }
                        let in_base = (sy as usize) * w * in_c + (sx as usize) * in_c;
                        let w_base = base_w + (ky * k + kx) * in_c;
                        for i in 0..in_c {
                            acc += (input[in_base + i] as i32)
                                * (weights[w_base + i] as i32);
                        }
                    }
                }
                out[oy * out_w * out_c + ox * out_c + o] =
                    requant(acc, desc.scale, bias[o], desc.relu);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::Weights;

    struct TestLayer {
        kind: u8,
        in_h: u16,
        in_w: u16,
        in_c: u16,
        out_c: u16,
        k: u8,
        stride: u8,
        relu: bool,
        scale: f32,
        bias: Vec<f32>,
        weights: Vec<i8>,
    }

    fn blob(layers: &[TestLayer]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"SWWT");
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&(layers.len() as u16).to_le_bytes());
        b.extend_from_slice(&1.0f32.to_le_bytes()); // input_scale
        b.push(20);                                 // logit_thr
        b.push(2);                                  // hold
        b.extend_from_slice(&32u16.to_le_bytes());  // refractory
        for l in layers {
            b.push(l.kind);
            for v in [l.in_h, l.in_w, l.in_c, l.out_c] {
                b.extend_from_slice(&v.to_le_bytes());
            }
            b.push(l.k);
            b.push(l.k);
            b.push(l.stride);
            b.push(l.relu as u8);
            b.extend_from_slice(&l.scale.to_le_bytes());
            for v in &l.bias { b.extend_from_slice(&v.to_le_bytes()); }
            for v in &l.weights { b.push(*v as u8); }
        }
        b
    }

    #[test]
    fn cnn_with_zero_layers_returns_input() {
        let bytes = blob(&[]);
        let w = Weights::parse(&bytes).unwrap();
        let input = vec![1i8, 2, 3, 4];
        let out = Cnn::run(&input, &w).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn dense_identity_layer_returns_input() {
        let mut weights = vec![0i8; 16];
        for i in 0..4 { weights[i * 4 + i] = 1; }
        let bytes = blob(&[TestLayer {
            kind: 2, in_h: 1, in_w: 1, in_c: 4, out_c: 4, k: 1, stride: 1,
            relu: false, scale: 1.0, bias: vec![0.0; 4], weights,
        }]);
        let w = Weights::parse(&bytes).unwrap();
        let out = Cnn::run(&[1, 2, 3, 4], &w).unwrap();
        assert_eq!(out, vec![1, 2, 3, 4]);
    }

    #[test]
    fn dense_applies_scale_bias_and_clamp() {
        // acc = 100, scale = 0.5, bias = 1.5 -> rint(51.5) = 52 (ties-even
        // not triggered); second output clamps at 127.
        let bytes = blob(&[TestLayer {
            kind: 2, in_h: 1, in_w: 1, in_c: 1, out_c: 2, k: 1, stride: 1,
            relu: false, scale: 0.5, bias: vec![1.5, 1000.0],
            weights: vec![1, 1],
        }]);
        let w = Weights::parse(&bytes).unwrap();
        let out = Cnn::run(&[100], &w).unwrap();
        assert_eq!(out, vec![52, 127]);
    }

    #[test]
    fn dense_rounds_half_to_even() {
        // acc = 5, scale = 0.5 -> 2.5 rounds to 2 (half to even), and
        // acc = 7 -> 3.5 rounds to 4.
        let bytes = blob(&[TestLayer {
            kind: 2, in_h: 1, in_w: 1, in_c: 2, out_c: 2, k: 1, stride: 1,
            relu: false, scale: 0.5, bias: vec![0.0, 0.0],
            weights: vec![1, 0, 0, 1],
        }]);
        let w = Weights::parse(&bytes).unwrap();
        assert_eq!(Cnn::run(&[5, 7], &w).unwrap(), vec![2, 4]);
    }

    #[test]
    fn relu_clamps_negative_to_zero() {
        let bytes = blob(&[TestLayer {
            kind: 2, in_h: 1, in_w: 1, in_c: 1, out_c: 1, k: 1, stride: 1,
            relu: true, scale: 1.0, bias: vec![0.0], weights: vec![1],
        }]);
        let w = Weights::parse(&bytes).unwrap();
        assert_eq!(Cnn::run(&[-5], &w).unwrap(), vec![0]);
    }

    #[test]
    fn dense_folds_gap_over_spatial_positions() {
        // 2x2x1 input, dense with shared weight 1, scale 0.25 = 1/(h*w):
        // mean of [4, 8, 12, 16] = 10.
        let bytes = blob(&[TestLayer {
            kind: 2, in_h: 2, in_w: 2, in_c: 1, out_c: 1, k: 1, stride: 1,
            relu: false, scale: 0.25, bias: vec![0.0], weights: vec![1],
        }]);
        let w = Weights::parse(&bytes).unwrap();
        assert_eq!(Cnn::run(&[4, 8, 12, 16], &w).unwrap(), vec![10]);
    }

    #[test]
    fn pointwise_shares_weights_across_positions() {
        // 1x2x2 input, pw (2 -> 1) with weights [1, 2] shared at both
        // positions: [1*1+2*2, 3*1+4*2] = [5, 11].
        let bytes = blob(&[TestLayer {
            kind: 1, in_h: 1, in_w: 2, in_c: 2, out_c: 1, k: 1, stride: 1,
            relu: false, scale: 1.0, bias: vec![0.0], weights: vec![1, 2],
        }]);
        let w = Weights::parse(&bytes).unwrap();
        assert_eq!(Cnn::run(&[1, 2, 3, 4], &w).unwrap(), vec![5, 11]);
    }

    #[test]
    fn depthwise_stride2_downsamples() {
        // 4x4x1, 3x3 identity kernel (center = 1), stride 2 -> 2x2 samples
        // at input positions (0,0), (0,2), (2,0), (2,2).
        let mut kernel = vec![0i8; 9];
        kernel[4] = 1;
        let input: Vec<i8> = (1..=16).collect();
        let bytes = blob(&[TestLayer {
            kind: 0, in_h: 4, in_w: 4, in_c: 1, out_c: 1, k: 3, stride: 2,
            relu: false, scale: 1.0, bias: vec![0.0], weights: kernel,
        }]);
        let w = Weights::parse(&bytes).unwrap();
        assert_eq!(Cnn::run(&input, &w).unwrap(), vec![1, 3, 9, 11]);
    }

    #[test]
    fn conv2d_stride2_shape_matches_pytorch_same_padding() {
        // 5x5x1 -> k3 s2 p1 -> 3x3. Sum kernel counts in-bounds neighbors.
        let input = vec![1i8; 25];
        let bytes = blob(&[TestLayer {
            kind: 3, in_h: 5, in_w: 5, in_c: 1, out_c: 1, k: 3, stride: 2,
            relu: false, scale: 1.0, bias: vec![0.0], weights: vec![1; 9],
        }]);
        let w = Weights::parse(&bytes).unwrap();
        let out = Cnn::run(&input, &w).unwrap();
        // Corners see 2x2 = 4, edges 2x3 = 6, center 3x3 = 9.
        assert_eq!(out, vec![4, 6, 4, 6, 9, 6, 4, 6, 4]);
    }

    #[test]
    fn conv2d_mixes_input_channels() {
        // 1x1x2 input [3, 5], one output channel, k=1 via 3x3 with only
        // center taps set: w_center = [2, -1] -> 3*2 - 5 = 1.
        let mut weights = vec![0i8; 18]; // (o=1, 3, 3, in_c=2)
        weights[(1 * 3 + 1) * 2] = 2;
        weights[(1 * 3 + 1) * 2 + 1] = -1;
        let bytes = blob(&[TestLayer {
            kind: 3, in_h: 1, in_w: 1, in_c: 2, out_c: 1, k: 3, stride: 1,
            relu: false, scale: 1.0, bias: vec![0.0], weights,
        }]);
        let w = Weights::parse(&bytes).unwrap();
        assert_eq!(Cnn::run(&[3, 5], &w).unwrap(), vec![1]);
    }

    #[test]
    fn layer_chain_threads_shapes() {
        // conv2d 2x2x1 -> 1x1x1 (k3 s2), then dense 1 -> 1 doubling.
        let mut kernel = vec![0i8; 9];
        kernel[4] = 1; // center tap: picks input (0,0)
        let bytes = blob(&[
            TestLayer {
                kind: 3, in_h: 2, in_w: 2, in_c: 1, out_c: 1, k: 3, stride: 2,
                relu: true, scale: 1.0, bias: vec![0.0], weights: kernel,
            },
            TestLayer {
                kind: 2, in_h: 1, in_w: 1, in_c: 1, out_c: 1, k: 1, stride: 1,
                relu: false, scale: 2.0, bias: vec![0.0], weights: vec![1],
            },
        ]);
        let w = Weights::parse(&bytes).unwrap();
        assert_eq!(Cnn::run(&[7, 0, 0, 0], &w).unwrap(), vec![14]);
    }
}
