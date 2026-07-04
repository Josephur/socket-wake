// SPDX-License-Identifier: Apache-2.0
//! Detection state machine, matching the streaming detector validated in
//! `python/socket_wake/eval_v3.py`: the CNN emits a quantized logit margin
//! (target minus not-target) per inference; we require it to reach
//! `threshold` for `hold` consecutive inferences before firing, with any
//! sub-threshold inference resetting the counter. After a fire the detector
//! enters a refractory lockout for `refractory` inferences, then re-arms
//! automatically.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection;

#[derive(Debug, PartialEq)]
pub struct Detector {
    threshold: i8,
    hold: u8,
    refractory: u16,
    count: u8,
    lockout: u16,
}

impl Detector {
    /// `threshold` is the per-inference logit-margin floor; `hold` is how
    /// many consecutive inferences must clear it before we fire (the v3
    /// benchmark used 2); `refractory` is the post-fire lockout measured in
    /// inferences (the v3 benchmark used 32 = 1 s at a 30 ms cadence).
    pub fn new(threshold: i8, hold: u8, refractory: u16) -> Self {
        Self { threshold, hold, refractory, count: 0, lockout: 0 }
    }

    /// Returns `Some(Detection)` exactly once per wake event. The detector
    /// re-arms automatically once the refractory period has elapsed.
    pub fn feed(&mut self, margin: i8) -> Option<Detection> {
        if self.lockout > 0 {
            self.lockout -= 1;
            return None;
        }
        if margin >= self.threshold {
            self.count = self.count.saturating_add(1);
            if self.count >= self.hold {
                self.count = 0;
                self.lockout = self.refractory;
                return Some(Detection);
            }
        } else {
            self.count = 0;
        }
        None
    }

    /// Clear the hold counter and any refractory lockout, re-arming the
    /// detector immediately.
    pub fn reset(&mut self) {
        self.count = 0;
        self.lockout = 0;
    }

    pub fn threshold(&self) -> i8 { self.threshold }
    pub fn hold(&self) -> u8 { self.hold }
    pub fn refractory(&self) -> u16 { self.refractory }
    pub fn count(&self) -> u8 { self.count }
    pub fn in_lockout(&self) -> bool { self.lockout > 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silent_until_threshold_crossed() {
        let mut d = Detector::new(50, 3, 10);
        for _ in 0..10 {
            assert_eq!(d.feed(10), None);
        }
        assert_eq!(d.count(), 0);
    }

    #[test]
    fn fires_after_required_hold() {
        let mut d = Detector::new(50, 3, 100);
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), Some(Detection));
        // Refractory: additional high frames don't refire.
        assert_eq!(d.feed(80), None);
        assert!(d.in_lockout());
    }

    #[test]
    fn sub_threshold_resets_counter() {
        let mut d = Detector::new(50, 3, 10);
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(0), None);      // gap resets the counter
        assert_eq!(d.count(), 0);
        assert_eq!(d.feed(80), None);     // counter back to 1
    }

    #[test]
    fn rearms_after_refractory() {
        let mut d = Detector::new(50, 2, 3);
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), Some(Detection));
        // 3 inferences of lockout, even above threshold.
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), None);
        // Re-armed: two more hits fire again.
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), Some(Detection));
    }

    #[test]
    fn manual_reset_clears_lockout() {
        let mut d = Detector::new(50, 2, 1000);
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), Some(Detection));
        assert!(d.in_lockout());
        d.reset();
        assert!(!d.in_lockout());
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), Some(Detection));
    }

    #[test]
    fn hold_of_one_fires_immediately() {
        let mut d = Detector::new(50, 1, 10);
        assert_eq!(d.feed(80), Some(Detection));
    }

    #[test]
    fn negative_margins_stay_silent() {
        let mut d = Detector::new(20, 2, 10);
        for _ in 0..5 {
            assert_eq!(d.feed(-100), None);
        }
    }
}
