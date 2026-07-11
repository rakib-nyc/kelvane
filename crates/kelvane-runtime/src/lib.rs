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

//! Kelvane runtime — a safety-first host for untrusted, hot-swappable
//! WebAssembly modules.
//!
//! # What it is
//!
//! Kelvane runs an untrusted `.wasm` guest and lets it request a neural-network
//! inference from a model the **host** owns. Every [`ModuleRuntime::invoke`]
//! executes in a **fresh** sandbox store carrying a hard memory ceiling, a
//! per-call CPU fuel budget, and **zero ambient authority** (no filesystem, no
//! network, no stdio). Modules can be [hot-swapped](ModuleRuntime::hot_swap) at
//! runtime, and reach inference through the `kelvane::infer` host import (CPU via
//! `tract`, or CUDA via ONNX Runtime behind the `cuda` feature).
//!
//! # Trust model (one paragraph)
//!
//! **Trusted:** the host process, the ONNX model, and the WebAssembly engine
//! (wasmtime). **Untrusted:** the guest module and the bytes it is given.
//! Kelvane's job is to let the untrusted guest run and ask for an inference
//! without being able to touch the host, the model weights, the accelerator, or
//! anything outside its own capped linear memory and fuel budget. It is *not* a
//! protection against side channels or against bugs in the engine itself — see
//! `SECURITY.md`.
//!
//! # Minimal end-to-end example
//!
//! ```no_run
//! use kelvane_runtime::{ExecutionLimits, ModuleRuntime};
//! use std::path::Path;
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut rt = ModuleRuntime::new(ExecutionLimits::default())?;
//! // Host owns the model; the guest never sees the weights.
//! rt.load_model(Path::new("models/policy.onnx"), &[1, 4, 11, 11])?;
//! rt.load_module(Path::new("policy_module.wasm"), "policy")?;
//!
//! // Drive the guest: bytes in, bytes out, fully sandboxed.
//! let decision = rt.invoke("policy", br#"{"data":[0.0]}"#)?;
//! println!("{} bytes back", decision.len());
//! # Ok(())
//! # }
//! ```
//!
//! See [`ModuleRuntime`] for the entry point and [`inference`] for backends. The
//! guest side of the ABI lives in the `kelvane-sdk` crate.
//!
//! # Stability
//!
//! Pre-1.0: the item-level classification and the semver promise are documented
//! in the crate README's "API stability" section. Treat anything reached through
//! [`internals`] as private and unstable.

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod host;
pub mod inference;

pub use host::{ExecutionLimits, ModuleRuntime};
pub use inference::{load_model, Model};

/// Internal, **unstable** surface exposed only for the in-repo fuzz targets and
/// gated behind the off-by-default `internals` Cargo feature.
///
/// These are the pure decode functions at the guest→host trust boundary
/// (`decode_output_region`, `bytes_to_f32`). They are **not** part of the public
/// API and are **not** covered by semver — do not depend on this module.
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod internals {
    /// Forwarding wrapper for the crate-internal `decode_output_region` (fuzz only).
    pub fn decode_output_region(packed: i64, mem_size: usize) -> anyhow::Result<(usize, usize)> {
        crate::host::decode_output_region(packed, mem_size)
    }

    /// Forwarding wrapper for the crate-internal `bytes_to_f32` (fuzz only).
    pub fn bytes_to_f32(buf: &[u8]) -> Vec<f32> {
        crate::host::bytes_to_f32(buf)
    }
}
