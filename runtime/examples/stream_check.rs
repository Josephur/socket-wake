// SPDX-License-Identifier: Apache-2.0
//! Drive the full runtime (mel -> INT8 CNN -> state machine) over raw
//! 16 kHz s16le PCM files and report detector fires. This is the
//! Rust-side counterpart of eval_v3.py's streaming benchmark: run it over
//! the streams emitted by `python -m socket_wake.dump_streams` to confirm
//! the on-target pipeline reproduces the simulated FAR/recall behavior.
//!
//! Usage:
//!   cargo run --release --example stream_check -- models/hey-socket-v1/streams/*.raw

use std::path::Path;

const WEIGHTS: &[u8] = include_bytes!("../../models/hey-socket-v1/weights_v3.bin");

fn score_file(path: &Path) -> (usize, f64) {
    let bytes = std::fs::read(path).expect("read PCM file");
    let pcm: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let secs = pcm.len() as f64 / 16_000.0;

    let d = socket_wake_runtime::socket_wake_create(
        WEIGHTS.as_ptr(),
        WEIGHTS.len(),
        16_000,
    );
    assert!(!d.is_null(), "detector create failed");
    let mut fires = 0usize;
    for chunk in pcm.chunks(1_600) {
        socket_wake_runtime::socket_wake_feed(d, chunk.as_ptr(), chunk.len());
        if socket_wake_runtime::socket_wake_detected(d) {
            fires += 1;
        }
    }
    socket_wake_runtime::socket_wake_destroy(d);
    (fires, secs)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: stream_check <pcm.raw> [more.raw ...]");
        std::process::exit(2);
    }
    let mut total_fires = 0usize;
    for arg in &args {
        let path = Path::new(arg);
        let (fires, secs) = score_file(path);
        total_fires += fires;
        println!(
            "{:<40} {:>7.1}s  fires={}",
            path.file_name().unwrap().to_string_lossy(),
            secs,
            fires
        );
    }
    println!("total fires: {total_fires}");
}
