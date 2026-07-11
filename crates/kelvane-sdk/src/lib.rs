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

//! Guest SDK and host/guest ABI for Kelvane WebAssembly modules.
//!
//! A Kelvane module is a `wasm32-wasip1` `cdylib` that the host
//! ([`kelvane-runtime`]) instantiates in a sandbox and drives through a tiny C
//! ABI. This crate factors out the unsafe linear-memory plumbing so a Rust guest
//! only supplies a byte-in / byte-out function and invokes [`export_module!`].
//! The ABI below is language-agnostic — a guest in any language that targets
//! `wasm32-wasip1` and matches these signatures will work.
//!
//! # ABI contract (host ⇄ guest)
//!
//! ## Exports the guest must provide
//!
//! * `memory` — the module's exported linear memory. **Required**; the host
//!   errors if it is absent.
//! * `module_alloc(len: i32) -> i32` — allocate `len` bytes and return an offset
//!   into `memory`. Returning `0` is treated as allocation failure and the call
//!   errors.
//! * `process(in_ptr: i32, in_len: i32) -> i64` — read `in_len` input bytes at
//!   `in_ptr`, produce output bytes, and return them **length-delimited** as a
//!   packed `i64`.
//! * `module_dealloc(ptr: i32, len: i32)` — *optional*; if exported, the host
//!   calls it with the **input** buffer's `(ptr, len)` after the call.
//!
//! ## Packed return value (bit layout)
//!
//! `process` returns one `i64` encoding the output region as `(ptr << 32) | len`:
//!
//! ```text
//!  bit 63                             32 31                              0
//! ┌────────────────────────────────────┬─────────────────────────────────┐
//! │        out_ptr  (u32 offset)        │        out_len  (u32 bytes)      │
//! └────────────────────────────────────┴─────────────────────────────────┘
//! ```
//!
//! The host reads exactly `out_len` bytes at `out_ptr`. It rejects the call if
//! `out_len` exceeds **1 MiB** or if `[out_ptr, out_ptr + out_len)` is not fully
//! inside `memory`. Empty output (`len == 0`) is valid. See [`pack`].
//!
//! ## Ownership / lifetime
//!
//! The host owns the **input** buffer (it allocates it via `module_alloc`, writes
//! into it, and — if `module_dealloc` is exported — frees it). The guest owns the
//! **output** buffer; the host only reads it. Because the host builds a **fresh
//! store per call** and drops it afterward, all guest allocations are reclaimed
//! between calls regardless — `module_dealloc` is for host-side hygiene, not
//! correctness.
//!
//! ## The `kelvane::infer` host import (optional)
//!
//! A guest may import host-owned neural inference:
//!
//! ```text
//! (import "kelvane" "infer"
//!    (func (param i32 i32 i32 i32) (result i32)))
//! //         in_ptr in_len out_ptr out_cap -> n
//! ```
//!
//! * `in_ptr` — offset of the input tensor: a flat, little-endian `f32` array.
//! * `in_len` — **number of `f32` values** (not bytes) at `in_ptr`. It must equal
//!   the loaded model's input length (the product of its fixed shape); the host
//!   reshapes the flat vector internally. `in_len * 4` must fit in `memory`.
//! * `out_ptr` — where to write the output scores (little-endian `f32`).
//! * `out_cap` — **number of `f32` slots** available at `out_ptr`.
//! * returns `n` — the number of `f32` scores written (`n <= out_cap`), or a
//!   **negative value** on any error: no model loaded, `in_len <= 0`,
//!   `out_cap <= 0`, no `memory` export, an out-of-bounds region, or an input
//!   length that does not match the model. A guest should treat `n < 0` as
//!   "inference unavailable" and degrade gracefully.
//!
//! The guest never sees the model weights or the accelerator — it only asks for
//! a forward pass across this boundary.

// Guest linear-memory interop is inherently unsafe; this crate is compiled into
// WebAssembly modules where that is expected.
#![allow(unsafe_code)]
#![deny(missing_docs)]

use std::alloc::{alloc as sys_alloc, dealloc as sys_dealloc, Layout};

/// Allocate `len` bytes in the guest's linear memory and return a pointer, or
/// `0` on failure (non-positive length or layout error).
pub fn alloc_bytes(len: i32) -> i32 {
    if len <= 0 {
        return 0;
    }
    let layout = match Layout::from_size_align(len as usize, 1) {
        Ok(l) => l,
        Err(_) => return 0,
    };
    // SAFETY: layout has non-zero size (len > 0) and valid alignment (1).
    let ptr = unsafe { sys_alloc(layout) };
    ptr as i32
}

/// Free a buffer previously returned by [`alloc_bytes`]. No-op on a null pointer
/// or non-positive length.
pub fn dealloc_bytes(ptr: i32, len: i32) {
    if ptr == 0 || len <= 0 {
        return;
    }
    let layout = match Layout::from_size_align(len as usize, 1) {
        Ok(l) => l,
        Err(_) => return,
    };
    // SAFETY: ptr/len match a prior alloc_bytes allocation and layout.
    unsafe { sys_dealloc(ptr as *mut u8, layout) };
}

/// Pack a `(pointer, length)` pair into one `i64`: high 32 bits = pointer,
/// low 32 bits = length. This is the `process` return encoding — see the crate
/// docs for the bit layout.
///
/// ```
/// use kelvane_sdk::pack;
/// let packed = pack(0x1000, 42);
/// assert_eq!((packed >> 32) as u32, 0x1000); // high 32 bits: pointer
/// assert_eq!((packed & 0xFFFF_FFFF) as u32, 42); // low 32 bits: length
/// ```
#[inline]
pub fn pack(ptr: i32, len: i32) -> i64 {
    ((ptr as u32 as i64) << 32) | (len as u32 as i64)
}

/// Drive a module's byte-in / byte-out function against a host-provided input
/// buffer and return the packed `(out_ptr << 32) | out_len`.
///
/// This is the wasm32 entry point: pointers are guest linear-memory offsets, so
/// the i32↔pointer casts are lossless there. It is exercised end-to-end by the
/// runtime integration tests (which load real WASM). Empty input yields empty
/// output rather than trapping.
pub fn run<F>(in_ptr: i32, in_len: i32, f: F) -> i64
where
    F: FnOnce(&[u8]) -> Vec<u8>,
{
    let output = if in_ptr != 0 && in_len > 0 {
        // SAFETY: the host guarantees [in_ptr, in_ptr+in_len) is a buffer it just
        // allocated via `module_alloc` and wrote the input into.
        let input = unsafe { std::slice::from_raw_parts(in_ptr as *const u8, in_len as usize) };
        f(input)
    } else {
        f(&[])
    };
    let len = output.len() as i32;
    let out_ptr = alloc_bytes(len);
    if out_ptr != 0 {
        // SAFETY: out_ptr points to `len` freshly allocated bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(output.as_ptr(), out_ptr as *mut u8, output.len());
        }
    }
    pack(out_ptr, len)
}

/// Generate the WASM ABI exports for a module from its byte-in / byte-out
/// function.
///
/// Emits `#[unsafe(no_mangle)]` `process`, `module_alloc`, and `module_dealloc`
/// in the calling crate so the symbols are present in the final `cdylib`
/// (defining them here in the library would risk dead-code elimination). The
/// handler takes the input bytes and returns the output bytes; all pointer
/// plumbing is handled for you.
///
/// ```no_run
/// use kelvane_sdk::export_module;
///
/// fn handle(input: &[u8]) -> Vec<u8> {
///     // Echo the input straight back out.
///     input.to_vec()
/// }
///
/// export_module!(handle);
/// ```
#[macro_export]
macro_rules! export_module {
    ($handler:path) => {
        /// Host entry point: `(in_ptr, in_len) -> (out_ptr << 32 | out_len)`.
        #[unsafe(no_mangle)]
        pub extern "C" fn process(in_ptr: i32, in_len: i32) -> i64 {
            $crate::run(in_ptr, in_len, $handler)
        }

        /// Allocate `len` bytes of guest memory for the host to fill.
        #[unsafe(no_mangle)]
        pub extern "C" fn module_alloc(len: i32) -> i32 {
            $crate::alloc_bytes(len)
        }

        /// Free a buffer previously returned by `module_alloc`.
        #[unsafe(no_mangle)]
        pub extern "C" fn module_dealloc(ptr: i32, len: i32) {
            $crate::dealloc_bytes(ptr, len)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: `run`/`alloc_bytes` cast i32↔pointer, which is lossless only under
    // wasm32 (32-bit pointers); they can't be round-tripped on a 64-bit host, so
    // they are validated end-to-end by the runtime integration tests against real
    // WASM. Here we test the pointer-free packing.

    #[test]
    fn pack_splits_into_high_ptr_low_len() {
        let packed = pack(0x1234, 0x56);
        assert_eq!((packed >> 32) as u32, 0x1234);
        assert_eq!((packed & 0xFFFF_FFFF) as u32, 0x56);
    }

    #[test]
    fn pack_handles_full_width() {
        let packed = pack(0x7FFF_FFFF, 0x00AB_CDEF);
        assert_eq!((packed >> 32) as u32, 0x7FFF_FFFF);
        assert_eq!((packed & 0xFFFF_FFFF) as u32, 0x00AB_CDEF);
    }
}
