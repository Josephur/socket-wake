// SPDX-License-Identifier: Apache-2.0
use socket_wake_runtime::weights::{LayerKind, WeightError, Weights};

/// SWWT v2 header: magic, version=2, n_layers, input_scale, logit_thr,
/// hold, refractory.
fn header(n_layers: u16) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"SWWT");
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&n_layers.to_le_bytes());
    bytes.extend_from_slice(&0.5f32.to_le_bytes()); // input_scale
    bytes.push(20u8 as u8);                         // logit_thr
    bytes.push(2);                                  // hold
    bytes.extend_from_slice(&32u16.to_le_bytes());  // refractory
    bytes
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = vec![0u8; 64];
    bytes[0..4].copy_from_slice(b"NOPE");
    let r = Weights::parse(&bytes);
    assert!(matches!(r, Err(WeightError::BadMagic)));
}

#[test]
fn rejects_truncated_header() {
    let bytes = vec![0u8; 3];
    let r = Weights::parse(&bytes);
    assert!(matches!(r, Err(WeightError::Truncated)));
}

#[test]
fn rejects_unsupported_version() {
    let mut bytes = vec![0u8; 16];
    bytes[0..4].copy_from_slice(b"SWWT");
    bytes[4..6].copy_from_slice(&42u16.to_le_bytes());
    assert!(matches!(
        Weights::parse(&bytes),
        Err(WeightError::UnsupportedVersion(42))
    ));
}

#[test]
fn old_v1_blobs_are_rejected() {
    // The pre-v3 exporter wrote version 1; the runtime must refuse it
    // rather than misparse the layer stream.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"SWWT");
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    assert!(matches!(
        Weights::parse(&bytes),
        Err(WeightError::UnsupportedVersion(1))
    ));
}

#[test]
fn parses_zero_layer_header_with_params() {
    let bytes = header(0);
    let w = Weights::parse(&bytes).expect("parse");
    assert_eq!(w.n_layers(), 0);
    let p = w.params();
    assert_eq!(p.input_scale, 0.5);
    assert_eq!(p.logit_thr, 20);
    assert_eq!(p.hold, 2);
    assert_eq!(p.refractory, 32);
}

/// Build a one-layer blob: dense layer, identity weights, zero bias.
fn identity_dense_layer_blob(in_n: u16, out_n: u16) -> Vec<u8> {
    let mut bytes = header(1);
    bytes.push(LayerKind::Dense as u8);
    bytes.extend_from_slice(&1u16.to_le_bytes()); // in_h
    bytes.extend_from_slice(&1u16.to_le_bytes()); // in_w
    bytes.extend_from_slice(&in_n.to_le_bytes()); // in_c
    bytes.extend_from_slice(&out_n.to_le_bytes()); // out_c
    bytes.push(1); // k_h
    bytes.push(1); // k_w
    bytes.push(1); // stride
    bytes.push(0); // relu
    bytes.extend_from_slice(&1.0f32.to_le_bytes()); // scale

    for _ in 0..out_n {
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
    }
    for i in 0..in_n {
        for j in 0..out_n {
            bytes.push(if i == j { 1 } else { 0 });
        }
    }
    bytes
}

#[test]
fn iterates_one_identity_dense_layer() {
    let bytes = identity_dense_layer_blob(4, 4);
    let w = Weights::parse(&bytes).expect("parse");
    assert_eq!(w.n_layers(), 1);

    let mut layers = socket_wake_runtime::weights::Layers::new(&w);
    let (desc, bias, weights) = layers.next().unwrap().expect("first layer");
    assert_eq!(desc.kind, LayerKind::Dense);
    assert_eq!(desc.in_c, 4);
    assert_eq!(desc.out_c, 4);
    assert_eq!(desc.stride, 1);
    assert!(!desc.relu);
    assert_eq!(bias.len(), 4 * 4);   // 4 f32s as bytes
    assert_eq!(weights.len(), 16);
    for chunk in bias.chunks(4) {
        let v = f32::from_le_bytes(chunk.try_into().unwrap());
        assert_eq!(v, 0.0);
    }
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(weights[i * 4 + j], if i == j { 1 } else { 0 });
        }
    }
    assert!(layers.next().is_none());
}

#[test]
fn rejects_truncated_layer_payload() {
    let mut bytes = header(1);
    bytes.push(LayerKind::Dense as u8);
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.push(1);
    bytes.push(1);
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    // No bias, no weights -- truncated.
    let w = Weights::parse(&bytes).expect("parse");
    let mut layers = socket_wake_runtime::weights::Layers::new(&w);
    let err = layers.next().unwrap().err().expect("must fail");
    assert!(matches!(err, socket_wake_runtime::weights::LayerIterError::Truncated));
}
