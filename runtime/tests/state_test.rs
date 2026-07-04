// SPDX-License-Identifier: Apache-2.0
use socket_wake_runtime::state::Detector;

#[test]
fn state_silent_until_threshold_crossed() {
    let mut d = Detector::new(50, 3, 32);
    for _ in 0..10 {
        assert!(d.feed(10).is_none());
    }
}

#[test]
fn state_fires_after_required_hold_frames() {
    let mut d = Detector::new(50, 3, 32);
    assert!(d.feed(80).is_none());
    assert!(d.feed(80).is_none());
    assert!(d.feed(80).is_some());
}

#[test]
fn state_resets_after_silence() {
    let mut d = Detector::new(50, 3, 32);
    assert!(d.feed(80).is_none());
    assert!(d.feed(80).is_none());
    assert!(d.feed(0).is_none());      // gap resets the counter
    assert!(d.feed(80).is_none());     // counter back to 1
}

#[test]
fn state_refractory_then_rearms() {
    let mut d = Detector::new(50, 2, 3);
    assert!(d.feed(80).is_none());
    assert!(d.feed(80).is_some());
    // Locked out for 3 inferences.
    assert!(d.feed(80).is_none());
    assert!(d.feed(80).is_none());
    assert!(d.feed(80).is_none());
    // Re-armed.
    assert!(d.feed(80).is_none());
    assert!(d.feed(80).is_some());
}

#[test]
fn state_manual_reset_clears_lockout() {
    let mut d = Detector::new(50, 2, 1000);
    assert!(d.feed(80).is_none());
    assert!(d.feed(80).is_some());
    d.reset();
    assert!(!d.in_lockout());
}
