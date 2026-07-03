// SPDX-License-Identifier: Apache-2.0
use socket_wake_runtime::arena::Arena;

#[test]
fn arena_tracks_peak_usage() {
    let mut buf = [0u8; 1024];
    let mut a = Arena::new(&mut buf);
    let _x = a.alloc::<u32>(10);          // 40 bytes
    let _y = a.alloc::<i8>(100);          // 100 bytes
    assert_eq!(a.peak(), 140);
    a.reset();                            // does not lower peak
    assert_eq!(a.peak(), 140);
}

#[test]
fn arena_rejects_oversized_alloc() {
    let mut buf = [0u8; 16];
    let mut a = Arena::new(&mut buf);
    assert!(a.alloc::<u32>(1024).is_none());
}