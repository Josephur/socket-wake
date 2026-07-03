// SPDX-License-Identifier: Apache-2.0
use socket_wake_runtime::weights::{LayerKind, WeightError, Weights};

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
fn parses_zero_layer_header() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"SWWT");
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    let w = Weights::parse(&bytes).expect("parse");
    assert_eq!(w.n_layers(), 0);
}

/// Build a one-layer blob: dense layer, in=4, out=4, identity weights, zero bias.
fn identity_dense_layer_blob(in_n: u16, out_n: u16) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"SWWT");
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // n_layers = 1

    bytes.push(LayerKind::Dense as u8);
    bytes.extend_from_slice(&1u16.to_le_bytes()); // in_h
    bytes.extend_from_slice(&1u16.to_le_bytes()); // in_w
    bytes.extend_from_slice(&in_n.to_le_bytes()); // in_c
    bytes.extend_from_slice(&out_n.to_le_bytes()); // out_c
    bytes.push(1); // k_h
    bytes.push(1); // k_w
    bytes.extend_from_slice(&1.0f32.to_le_bytes()); // scale

    for _ in 0..out_n {
        bytes.extend_from_slice(&0i32.to_le_bytes());
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
    assert_eq!(bias.len(), 4 * 4);   // 4 i32s as bytes
    assert_eq!(weights.len(), 16);
    for chunk in bias.chunks(4) {
        let v = i32::from_le_bytes(chunk.try_into().unwrap());
        assert_eq!(v, 0);
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
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"SWWT");
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // 1 layer
    bytes.push(LayerKind::Dense as u8);
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.push(1);
    bytes.push(1);
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    // No bias, no weights -- truncated.
    let w = Weights::parse(&bytes).expect("parse");
    let mut layers = socket_wake_runtime::weights::Layers::new(&w);
    let err = layers.next().unwrap().err().expect("must fail");
    assert!(matches!(err, socket_wake_runtime::weights::LayerIterError::Truncated));
}