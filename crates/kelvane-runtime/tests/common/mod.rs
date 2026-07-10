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

//! Shared helpers for the runtime integration tests.
//!
//! Missing artifacts are a hard **failure**, never a skip: a green test run must
//! mean the sandbox was actually exercised (see the Phase 2 report). WASM
//! modules are written inline as WAT and materialized to temp files; ONNX models
//! are committed fixtures under `tests/models/`.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use kelvane_runtime::{ExecutionLimits, ModuleRuntime};

static CTR: AtomicUsize = AtomicUsize::new(0);

/// Materialize a WAT (or binary WASM) module to a unique temp file and return
/// its path. `wasmtime::Module::new` accepts the text format, so hostile modules
/// can be authored inline.
pub fn module_file(name: &str, text: &str) -> PathBuf {
    let n = CTR.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("kelvane_{}_{}_{}.wat", std::process::id(), name, n));
    std::fs::write(&p, text).expect("write temp module file");
    p
}

/// Path to a committed ONNX fixture. **Panics** (test failure) if absent.
pub fn fixture(name: &str) -> PathBuf {
    let candidates = [
        format!("tests/models/{name}"),
        format!("crates/kelvane-runtime/tests/models/{name}"),
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    panic!(
        "required ONNX fixture '{name}' not found (looked in {candidates:?}); \
         regenerate with crates/kelvane-runtime/tests/models/generate.py"
    );
}

/// Path to a built example-module `.wasm`. **Panics** (test failure) if absent,
/// so CI must build the wasm modules before the test step.
pub fn module_wasm(name: &str) -> PathBuf {
    for base in [
        "../../target/wasm32-wasip1/release",
        "target/wasm32-wasip1/release",
    ] {
        let p = PathBuf::from(format!("{base}/{name}.wasm"));
        if p.exists() {
            return p;
        }
    }
    panic!(
        "required wasm module '{name}.wasm' not built; run: \
         cargo build --release --target wasm32-wasip1 -p policy-module -p scripted-module"
    );
}

/// A runtime with default limits.
pub fn default_rt() -> ModuleRuntime {
    ModuleRuntime::new(ExecutionLimits::default()).expect("construct runtime")
}

/// A JSON observation `{"data":[...]}` of `n` zero features.
pub fn zeros_observation(n: usize) -> String {
    let data = vec![0.0_f32; n];
    format!("{{\"data\":{}}}", serde_json::to_string(&data).unwrap())
}
