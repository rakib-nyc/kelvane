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
//! Every invocation runs in a **fresh** sandbox store carrying a hard memory
//! ceiling, a per-call CPU fuel budget, and **zero ambient authority** (no
//! filesystem, no network, no stdio). Modules can be **hot-swapped** at runtime,
//! and are offered **host-owned neural inference** (CPU via `tract`, or CUDA via
//! ONNX Runtime behind the `cuda` feature) through the `kelvane::infer` import.
//!
//! See [`ModuleRuntime`] for the entry point and [`inference`] for backends.

#![deny(unsafe_code)]

pub mod host;
pub mod inference;

pub use host::{ExecutionLimits, ModuleRuntime};
pub use inference::{load_model, Model};
