// SPDX-License-Identifier: Apache-2.0
// End-to-end detector wiring test through the C ABI with a synthetic
// zero-layer blob (the detector behaves as a passthrough; a zero-layer
// model emits the raw window, which has != 2 "logits", so the state
// machine never sees input). The real-model path is covered by
// parity_test.rs against the exported v3 blob.

fn v2_header(n_layers: u16) -> Vec<u8> {
    let mut blob = Vec::new();
    blob.extend_from_slice(b"SWWT");
    blob.extend_from_slice(&2u16.to_le_bytes());
    blob.extend_from_slice(&n_layers.to_le_bytes());
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // input_scale
    blob.push(20);                                 // logit_thr
    blob.push(2);                                  // hold
    blob.extend_from_slice(&32u16.to_le_bytes());  // refractory
    blob
}

#[test]
fn detector_compiles_and_links() {
    let blob = v2_header(0);
    let d = socket_wake_runtime::socket_wake_create(
        blob.as_ptr(),
        blob.len(),
        16_000,
    );
    assert!(!d.is_null(), "create must succeed for a valid (empty) blob");
    // Feed 2 s so the 1 s window fills and inference actually runs.
    let silence = vec![0i16; 32_000];
    socket_wake_runtime::socket_wake_feed(d, silence.as_ptr(), silence.len());
    assert!(
        !socket_wake_runtime::socket_wake_detected(d),
        "no detection should fire on silence through an empty model"
    );
    let peak = socket_wake_runtime::socket_wake_peak_ram_bytes(d);
    assert!(
        peak < 24 * 1024,
        "peak RAM must be < 24 KB even for an empty model; got {peak}"
    );
    socket_wake_runtime::socket_wake_destroy(d);
}

#[test]
fn create_rejects_v1_blob() {
    let mut blob = Vec::new();
    blob.extend_from_slice(b"SWWT");
    blob.extend_from_slice(&1u16.to_le_bytes());
    blob.extend_from_slice(&0u16.to_le_bytes());
    let d = socket_wake_runtime::socket_wake_create(blob.as_ptr(), blob.len(), 16_000);
    assert!(d.is_null(), "v1 blobs must be rejected at create()");
}
