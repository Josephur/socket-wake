// SPDX-License-Identifier: Apache-2.0
//! Detection state machine. The CNN emits a single logit per frame; we
//! require it to exceed `threshold` for `hold_frames` consecutive frames
//! before firing, with any sub-threshold frame resetting the counter.
//! This is the canonical "debounce" pattern for keyword spotting and
//! trades a small amount of latency (one extra frame) for dramatically
//! fewer false positives.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection;

#[derive(Debug, PartialEq)]
pub struct Detector {
    threshold: i8,
    hold_frames: u8,
    count: u8,
    fired: bool,
}

impl Detector {
    /// `threshold` is the per-frame logit floor; `hold_frames` is how many
    /// consecutive frames must clear it before we fire (typical: 3-5).
    pub fn new(threshold: i8, hold_frames: u8) -> Self {
        Self {
            threshold,
            hold_frames,
            count: 0,
            fired: false,
        }
    }

    /// Returns `Some(Detection)` exactly once per wake event. To re-arm,
    /// call `reset()` after the consumer has handled the detection.
    pub fn feed(&mut self, logit: i8) -> Option<Detection> {
        if logit >= self.threshold {
            self.count = self.count.saturating_add(1);
            if !self.fired && self.count >= self.hold_frames {
                self.fired = true;
                return Some(Detection);
            }
        } else {
            self.count = 0;
        }
        None
    }

    /// Re-arm the detector so it can fire again on the next qualifying
    /// sequence. Call after the consumer has handled the previous
    /// detection (typically inside a single-tap handler).
    pub fn reset(&mut self) {
        self.count = 0;
        self.fired = false;
    }

    pub fn threshold(&self) -> i8 { self.threshold }
    pub fn hold_frames(&self) -> u8 { self.hold_frames }
    pub fn count(&self) -> u8 { self.count }
    pub fn fired(&self) -> bool { self.fired }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silent_until_threshold_crossed() {
        let mut d = Detector::new(50, 3);
        for _ in 0..10 {
            assert_eq!(d.feed(10), None);
        }
        assert_eq!(d.count(), 0);
        assert!(!d.fired());
    }

    #[test]
    fn fires_after_required_hold_frames() {
        let mut d = Detector::new(50, 3);
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), Some(Detection));
        // Once fired, additional high frames don't refire.
        assert_eq!(d.feed(80), None);
    }

    #[test]
    fn resets_after_silence() {
        let mut d = Detector::new(50, 3);
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(0), None);      // gap resets the counter
        assert_eq!(d.count(), 0);
        assert_eq!(d.feed(80), None);     // counter back to 1
    }

    #[test]
    fn manual_reset_re_arms() {
        let mut d = Detector::new(50, 2);
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), Some(Detection));
        d.reset();
        assert!(!d.fired());
        assert_eq!(d.feed(80), None);
        assert_eq!(d.feed(80), Some(Detection));
    }

    #[test]
    fn hold_of_one_fires_immediately() {
        let mut d = Detector::new(50, 1);
        assert_eq!(d.feed(80), Some(Detection));
    }

    #[test]
    fn hold_of_zero_fires_immediately() {
        // With hold_frames=0 the count starts at 0 and the `>= 0` check
        // passes on the first frame -- i.e. hold=0 degenerates to "fire
        // immediately on any above-threshold frame". Tests document this
        // explicitly so future readers don't expect zero hold to mean
        // "never fires".
        let mut d = Detector::new(50, 0);
        assert_eq!(d.feed(80), Some(Detection));
    }
}