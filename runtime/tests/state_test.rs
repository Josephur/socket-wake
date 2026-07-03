// SPDX-License-Identifier: Apache-2.0
use socket_wake_runtime::state::Detector;

#[test]
fn state_silent_until_threshold_crossed() {
    let mut d = Detector::new(50, 3);
    for _ in 0..10 {
        assert!(d.feed(10).is_none());
    }
}

#[test]
fn state_fires_after_required_hold_frames() {
    let mut d = Detector::new(50, 3);
    assert!(d.feed(80).is_none());
    assert!(d.feed(80).is_none());
    assert!(d.feed(80).is_some());
}

#[test]
fn state_resets_after_silence() {
    let mut d = Detector::new(50, 3);
    assert!(d.feed(80).is_none());
    assert!(d.feed(80).is_none());
    assert!(d.feed(0).is_none());      // gap resets the counter
    assert!(d.feed(80).is_none());     // counter back to 1
}

#[test]
fn state_resets_can_rearm() {
    let mut d = Detector::new(50, 2);
    assert!(d.feed(80).is_none());
    assert!(d.feed(80).is_some());
    d.reset();
    assert!(!d.fired());
}