// SPDX-License-Identifier: Apache-2.0
// End-to-end detector test. The detector wires mel -> CNN -> state machine.
// A canned model lands in Task 14; until then we exercise the wiring with
// a synthetic zero-layer blob and confirm the C ABI links and behaves.

#[test]
fn detector_compiles_and_links() {
    // Trivial smoke test: a zero-layer blob makes the detector behave as
    // a passthrough; if the C ABI compiles and links, the wiring is sound.
    let blob = vec![
        b'S', b'W', b'W', b'T',     // magic
        1, 0,                        // version = 1 (LE)
        0, 0,                        // n_layers = 0
    ];
    let d = unsafe {
        socket_wake_runtime::socket_wake_create(
            blob.as_ptr(),
            blob.len(),
            16_000,
        )
    };
    assert!(!d.is_null(), "create must succeed for a valid (empty) blob");
    let silence = vec![0i16; 16_000];
    unsafe {
        socket_wake_runtime::socket_wake_feed(
            d,
            silence.as_ptr(),
            silence.len(),
        );
    }
    assert!(
        !unsafe { socket_wake_runtime::socket_wake_detected(d) },
        "no detection should fire on silence through an empty model"
    );
    let peak = unsafe { socket_wake_runtime::socket_wake_peak_ram_bytes(d) };
    assert!(
        peak < 24 * 1024,
        "peak RAM must be < 24 KB even for an empty model; got {peak}"
    );
    unsafe { socket_wake_runtime::socket_wake_destroy(d) };
}