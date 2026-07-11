// Copyright 2026 Muhammad Rakibul Islam
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Adversarial / malformed-input suite.
//!
//! Each test drives the real `ModuleRuntime::invoke` path with a deliberately
//! hostile guest module (authored inline as WAT) or hostile inference request,
//! and asserts the host **contains** it: returns `Err` (or a bounded, documented
//! result) and never crashes the host process, hangs, leaks across calls, or
//! reads/writes out of bounds. Every case asserts a specific outcome.

mod common;

use common::{default_rt, fixture, module_file, module_wasm, zeros_observation};
use kelvane_runtime::{ExecutionLimits, ModuleRuntime};

/// A WAT expression that returns the packed `(ptr << 32 | len)` i64.
fn packed(ptr: u32, len: u32) -> String {
    format!(
        "(i64.or (i64.shl (i64.const {ptr}) (i64.const 32)) (i64.const {len}))",
        ptr = ptr as i32,
        len = len as i32
    )
}

/// Invoke a one-off WAT module with the given input, returning the host result.
fn run_wat(name: &str, wat: &str, input: &[u8]) -> anyhow::Result<Vec<u8>> {
    let path = module_file(name, wat);
    let mut rt = default_rt();
    rt.load_module(&path, name)?;
    rt.invoke(name, input)
}

// ===========================================================================
// ABI: exports, traps, malformed packed return
// ===========================================================================

#[test]
fn missing_process_export_is_error() {
    let wat = r#"(module
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024))"#;
    let r = run_wat("missing_process", wat, b"{}");
    assert!(
        r.is_err(),
        "module with no `process` export must be rejected"
    );
}

#[test]
fn missing_memory_export_is_error() {
    let wat = r#"(module
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64) i64.const 0))"#;
    let r = run_wat("missing_memory", wat, b"{}");
    assert!(
        r.is_err(),
        "module with no `memory` export must be rejected"
    );
}

#[test]
fn wrong_alloc_signature_is_error() {
    // `module_alloc` typed i32 -> i64 instead of i32 -> i32.
    let wat = r#"(module
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i64) i64.const 1024)
        (func (export "process") (param i32 i32) (result i64) i64.const 0))"#;
    let r = run_wat("wrong_alloc_sig", wat, b"{}");
    assert!(r.is_err(), "wrong module_alloc signature must be rejected");
}

#[test]
fn alloc_returns_null_is_error() {
    let wat = r#"(module
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i32) i32.const 0)
        (func (export "process") (param i32 i32) (result i64) i64.const 0))"#;
    let r = run_wat("alloc_null", wat, b"hello");
    assert!(r.is_err(), "null alloc pointer must be rejected");
}

#[test]
fn process_trap_is_error_not_crash() {
    let wat = r#"(module
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64) unreachable))"#;
    let r = run_wat("process_trap", wat, b"{}");
    assert!(r.is_err(), "a trapping process must surface as Err");
}

#[test]
fn output_len_over_cap_is_error() {
    // len = 2 MiB > MAX_OUTPUT_BYTES (1 MiB).
    let wat = format!(
        r#"(module
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64) {}))"#,
        packed(0, 2 * 1024 * 1024)
    );
    let r = run_wat("len_over_cap", &wat, b"{}");
    assert!(
        r.is_err(),
        "output length over the 1 MiB cap must be rejected"
    );
}

#[test]
fn output_region_out_of_bounds_is_error() {
    // ptr near end of a 1-page (64 KiB) memory, len pushing past it.
    let wat = format!(
        r#"(module
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64) {}))"#,
        packed(0x0000_F000, 0x0000_4000)
    );
    let r = run_wat("out_oob", &wat, b"{}");
    assert!(r.is_err(), "out-of-bounds output region must be rejected");
}

#[test]
fn garbage_packed_return_is_error() {
    // Both words huge; the length word alone exceeds the cap.
    let wat = format!(
        r#"(module
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64) {}))"#,
        packed(0x7FFF_FFFF, 0x7FFF_FFFF)
    );
    let r = run_wat("garbage_packed", &wat, b"{}");
    assert!(r.is_err(), "garbage packed return must be rejected");
}

#[test]
fn max_pointer_stays_in_bounds_check() {
    // ptr = u32::MAX, len = exactly the cap (1 MiB, allowed by the len check) —
    // the addition would overflow a 32-bit sum but not usize; must be caught as
    // out-of-bounds, never as a panic/UB.
    let wat = format!(
        r#"(module
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64) {}))"#,
        packed(0xFFFF_FFFF, 1024 * 1024)
    );
    let r = run_wat("max_ptr", &wat, b"{}");
    assert!(
        r.is_err(),
        "max-pointer output region must be rejected as OOB"
    );
}

#[test]
fn null_ptr_nonzero_len_reads_within_guest_memory() {
    // ptr=0, len=100: decodes in-bounds (offset 0 is valid guest memory), so the
    // host returns 100 bytes from the guest's own memory. Documented, bounded
    // behavior — NOT a host safety violation (no host memory is exposed).
    let wat = format!(
        r#"(module
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64) {}))"#,
        packed(0, 100)
    );
    let out = run_wat("null_ptr_len", &wat, b"{}").expect("in-bounds read should succeed");
    assert_eq!(
        out.len(),
        100,
        "returns exactly the claimed length, bounded"
    );
}

// ===========================================================================
// Compute: fuel + stack
// ===========================================================================

#[test]
fn infinite_loop_hits_fuel_limit_not_hang() {
    let wat = r#"(module
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64)
            (loop $l br $l)
            i64.const 0))"#;
    let r = run_wat("infinite_loop", wat, b"{}");
    assert!(
        r.is_err(),
        "infinite loop must trap on fuel exhaustion, not hang"
    );
}

#[test]
fn deep_recursion_traps_not_crash() {
    let wat = r#"(module
        (memory (export "memory") 1)
        (func $rec (param i32) (result i32) (call $rec (local.get 0)))
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64)
            (drop (call $rec (i32.const 0)))
            i64.const 0))"#;
    let r = run_wat("deep_recursion", wat, b"{}");
    assert!(
        r.is_err(),
        "unbounded recursion must trap (stack/fuel), not crash host"
    );
}

#[test]
fn fuel_boundary_brackets_success_and_trap() {
    // A bounded ~1e6-iteration loop. Ample fuel completes; tiny fuel traps.
    let wat = r#"(module
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64)
            (local $i i32)
            (loop $l
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br_if $l (i32.lt_u (local.get $i) (i32.const 1000000))))
            i64.const 0))"#;
    let path = module_file("fuel_boundary", wat);

    let mut ok_rt = default_rt();
    ok_rt.load_module(&path, "fb").unwrap();
    assert!(
        ok_rt.invoke("fb", b"{}").is_ok(),
        "ample fuel should complete the bounded loop"
    );

    let starved = ExecutionLimits {
        fuel_per_call: 100,
        ..ExecutionLimits::default()
    };
    let mut low_rt = ModuleRuntime::new(starved).unwrap();
    low_rt.load_module(&path, "fb").unwrap();
    assert!(
        low_rt.invoke("fb", b"{}").is_err(),
        "a tiny fuel budget must trap the same loop"
    );
}

// ===========================================================================
// Memory: cap enforcement
// ===========================================================================

#[test]
fn oversized_initial_memory_is_rejected() {
    // 2000 pages = 125 MiB initial linear memory > the 64 MiB default cap.
    let wat = r#"(module
        (memory (export "memory") 2000)
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64) i64.const 0))"#;
    let r = run_wat("oversized_mem", wat, b"{}");
    assert!(
        r.is_err(),
        "initial memory above the cap must be rejected at instantiation"
    );
}

#[test]
fn memory_grow_loop_stays_capped() {
    // Repeatedly try to grow far beyond the cap; the limiter must deny growth
    // and the host must stay bounded (no OOM, no hang). grow returns -1 when
    // denied; the loop is bounded so the call terminates.
    let wat = r#"(module
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64)
            (local $i i32)
            (loop $l
                (drop (memory.grow (i32.const 100)))
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br_if $l (i32.lt_u (local.get $i) (i32.const 50))))
            i64.const 0))"#;
    // Empty output (packed 0,0) → Ok(vec![]). The point is it returns, capped.
    let out = run_wat("grow_loop", wat, b"{}");
    assert!(out.is_ok(), "capped grow loop should complete: {out:?}");
    assert!(out.unwrap().is_empty());
}

// ===========================================================================
// State isolation: fresh store per call
// ===========================================================================

#[test]
fn per_call_state_is_isolated() {
    // A mutable global incremented each call and written to memory. Because every
    // invoke instantiates a fresh store, both calls must observe the same value
    // (1), proving no state carries across calls.
    let wat = format!(
        r#"(module
        (memory (export "memory") 1)
        (global $g (mut i32) (i32.const 0))
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64)
            (global.set $g (i32.add (global.get $g) (i32.const 1)))
            (i32.store (i32.const 2048) (global.get $g))
            {}))"#,
        packed(2048, 4)
    );
    let path = module_file("state_iso", &wat);
    let mut rt = default_rt();
    rt.load_module(&path, "s").unwrap();
    let a = rt.invoke("s", b"{}").unwrap();
    let b = rt.invoke("s", b"{}").unwrap();
    assert_eq!(a, b, "fresh store per call must reset global state");
    assert_eq!(
        i32::from_le_bytes([a[0], a[1], a[2], a[3]]),
        1,
        "counter must read 1 on every call, never accumulate"
    );
    assert_eq!(rt.call_count("s"), Some(2));
}

// ===========================================================================
// No ambient authority: the guest is granted no filesystem
// ===========================================================================

#[test]
fn no_ambient_filesystem_authority() {
    // The guest imports a WASI filesystem primitive (`fd_prestat_get`, which WASI
    // libc uses to enumerate preopened directories) and calls it on the first
    // preopen slot (fd 3). The runtime grants NO preopens, so this must fail with
    // a non-zero errno — proving the guest has no filesystem capability. The
    // errno is returned as the module's 4-byte output.
    let wat = format!(
        r#"(module
        (import "wasi_snapshot_preview1" "fd_prestat_get"
            (func $fd_prestat_get (param i32 i32) (result i32)))
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64)
            (i32.store (i32.const 100) (call $fd_prestat_get (i32.const 3) (i32.const 2048)))
            {}))"#,
        packed(100, 4)
    );
    let out = run_wat("no_fs", &wat, b"{}").expect("call should complete, not trap");
    let errno = i32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    assert_ne!(
        errno, 0,
        "fd_prestat_get(3) must be denied (no preopens); got success errno 0"
    );
}

// ===========================================================================
// Inference boundary: hostile kelvane::infer requests (WAT calls the import)
// ===========================================================================

/// Build a module that calls `kelvane::infer(in_ptr,in_len,out_ptr,out_cap)` and
/// returns the i32 result count as its 4-byte output (so the test can read it).
fn infer_probe_wat(in_ptr: u32, in_len: i32, out_ptr: u32, out_cap: i32) -> String {
    format!(
        r#"(module
        (import "kelvane" "infer" (func $infer (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (func (export "module_alloc") (param i32) (result i32) i32.const 1024)
        (func (export "process") (param i32 i32) (result i64)
            (local $n i32)
            (local.set $n (call $infer
                (i32.const {in_ptr}) (i32.const {in_len})
                (i32.const {out_ptr}) (i32.const {out_cap})))
            (i32.store (i32.const 100) (local.get $n))
            {ret}))"#,
        in_ptr = in_ptr as i32,
        out_ptr = out_ptr as i32,
        ret = packed(100, 4)
    )
}

/// Run an infer-probe against a runtime with `model` (shape `shape`) loaded, and
/// return the i32 value `kelvane::infer` produced.
fn run_infer_probe(
    tag: &str,
    model_fixture: &str,
    shape: &[usize],
    in_ptr: u32,
    in_len: i32,
    out_ptr: u32,
    out_cap: i32,
) -> i32 {
    let wat = infer_probe_wat(in_ptr, in_len, out_ptr, out_cap);
    let path = module_file(tag, &wat);
    let mut rt = default_rt();
    rt.load_model(&fixture(model_fixture), shape).unwrap();
    rt.load_module(&path, tag).unwrap();
    let out = rt
        .invoke(tag, b"{}")
        .expect("probe module should not fault the host");
    assert_eq!(out.len(), 4);
    i32::from_le_bytes([out[0], out[1], out[2], out[3]])
}

const POLICY: &str = "policy_4x11x11.onnx";
const POLICY_SHAPE: &[usize] = &[1, 4, 11, 11]; // 484 inputs, 7 outputs

#[test]
fn infer_correct_request_returns_output_count() {
    let n = run_infer_probe("infer_ok", POLICY, POLICY_SHAPE, 0, 484, 4000, 64);
    assert_eq!(n, 7, "valid request should return the 7 action scores");
}

#[test]
fn infer_input_too_short_returns_negative() {
    let n = run_infer_probe("infer_short", POLICY, POLICY_SHAPE, 0, 4, 4000, 64);
    assert!(
        n < 0,
        "wrong (short) input length must be rejected: got {n}"
    );
}

#[test]
fn infer_input_too_long_returns_negative() {
    let n = run_infer_probe("infer_long", POLICY, POLICY_SHAPE, 0, 1000, 8000, 64);
    assert!(n < 0, "wrong (long) input length must be rejected: got {n}");
}

#[test]
fn infer_out_cap_smaller_than_output_truncates() {
    // out_cap=1 while the model emits 7: host writes min(7,1)=1, bounded.
    let n = run_infer_probe("infer_cap", POLICY, POLICY_SHAPE, 0, 484, 4000, 1);
    assert_eq!(n, 1, "small out_cap must bound the write to out_cap");
}

#[test]
fn infer_huge_in_len_is_rejected_without_giant_alloc() {
    // Regression: `in_len` is attacker-controlled; a value near i32::MAX would
    // make the host try to allocate ~8 GiB before any bounds check. Must be
    // rejected (returns negative) cheaply, bounded by the guest's memory size.
    let n = run_infer_probe(
        "infer_huge",
        POLICY,
        POLICY_SHAPE,
        0,
        2_000_000_000,
        4000,
        64,
    );
    assert!(n < 0, "an absurd in_len must be rejected, got {n}");
}

#[test]
fn infer_in_ptr_out_of_bounds_returns_negative() {
    let n = run_infer_probe(
        "infer_in_oob",
        POLICY,
        POLICY_SHAPE,
        0x7FFF_0000,
        484,
        4000,
        64,
    );
    assert!(n < 0, "out-of-bounds in_ptr must be rejected: got {n}");
}

#[test]
fn infer_out_ptr_out_of_bounds_returns_negative() {
    let n = run_infer_probe(
        "infer_out_oob",
        POLICY,
        POLICY_SHAPE,
        0,
        484,
        0x7FFF_0000,
        64,
    );
    assert!(n < 0, "out-of-bounds out_ptr must be rejected: got {n}");
}

#[test]
fn infer_with_no_model_returns_negative() {
    // No model loaded: the import must return -1, not fault.
    let wat = infer_probe_wat(0, 484, 4000, 64);
    let n = {
        let path = module_file("infer_nomodel", &wat);
        let mut rt = default_rt();
        rt.load_module(&path, "nm").unwrap();
        let out = rt.invoke("nm", b"{}").unwrap();
        i32::from_le_bytes([out[0], out[1], out[2], out[3]])
    };
    assert!(n < 0, "infer with no model must return negative, got {n}");
}

// ===========================================================================
// Model loading: malformed / non-ONNX / shape mismatch
// ===========================================================================

#[test]
fn non_onnx_file_fails_to_load() {
    let bogus = std::env::temp_dir().join(format!("kelvane_bogus_{}.onnx", std::process::id()));
    std::fs::write(&bogus, b"this is definitely not an onnx protobuf").unwrap();
    let mut rt = default_rt();
    let r = rt.load_model(&bogus, &[1, 4, 11, 11]);
    assert!(
        r.is_err(),
        "a non-ONNX file must fail to load, not be accepted"
    );
}

#[test]
fn declared_shape_mismatch_fails() {
    // mlp_16 expects [1,16]; forcing the policy shape must be rejected.
    let mut rt = default_rt();
    let r = rt.load_model(&fixture("mlp_16.onnx"), &[1, 4, 11, 11]);
    assert!(
        r.is_err(),
        "a declared input shape incompatible with the model must be rejected"
    );
}

// ===========================================================================
// Variable input shapes (2d): >1 differently-shaped model end-to-end
// ===========================================================================

fn run_shape(model: &str, shape: &[usize], n_features: usize, max_action: u64) {
    let mut rt = default_rt();
    rt.load_model(&fixture(model), shape).unwrap();
    rt.load_module(&module_wasm("policy_module"), "p").unwrap();
    let obs = zeros_observation(n_features);
    let out = rt.invoke("p", obs.as_bytes()).unwrap();
    let decision: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(decision["module"], "policy");
    let action = decision["action"].as_u64().expect("action is a number");
    assert!(
        action < max_action,
        "action {action} out of range for {model} ({max_action} actions)"
    );
}

#[test]
fn shape_policy_4x11x11_runs() {
    run_shape(POLICY, POLICY_SHAPE, 484, 7);
}

#[test]
fn shape_mlp_16_runs() {
    run_shape("mlp_16.onnx", &[1, 16], 16, 3);
}

#[test]
fn shape_img_3x8x8_runs() {
    run_shape("img_3x8x8.onnx", &[1, 3, 8, 8], 192, 5);
}
