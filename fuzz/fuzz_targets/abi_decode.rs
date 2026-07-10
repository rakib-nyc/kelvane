// Copyright 2026 Muhammad Rakibul Islam
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Fuzz the ABI decode trust boundary: the packed `(ptr << 32 | len)` unpack and
//! the against-linear-memory bounds check in `decode_output_region`. Every field
//! is attacker-controlled here. Invariants: it must never panic, and any `Ok`
//! region must lie fully inside `mem_size` and within the 1 MiB output cap.
#![no_main]

use kelvane_runtime::host::decode_output_region;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }
    let packed = i64::from_le_bytes(data[0..8].try_into().unwrap());
    let mem_size = usize::from_le_bytes(data[8..16].try_into().unwrap());

    if let Ok((off, len)) = decode_output_region(packed, mem_size) {
        let end = off.checked_add(len).expect("Ok region must not overflow");
        assert!(end <= mem_size, "Ok region must be within memory");
        assert!(len <= 1024 * 1024, "Ok length must be within the cap");
    }
});
