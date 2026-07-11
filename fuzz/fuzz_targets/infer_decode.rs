// Copyright 2026 Muhammad Rakibul Islam
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Fuzz the inference-boundary decode: turning raw guest-memory bytes into the
//! `f32` request vector (`bytes_to_f32`). It must be total — never panic on any
//! byte string — and yield exactly `len / 4` floats (trailing partial ignored).
#![no_main]

use kelvane_runtime::internals::bytes_to_f32;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let out = bytes_to_f32(data);
    assert_eq!(out.len(), data.len() / 4, "one f32 per whole 4-byte chunk");
});
