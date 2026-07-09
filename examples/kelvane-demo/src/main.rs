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

//! Kelvane end-to-end demo.
//!
//! Loads a trained ONNX policy and runs it inside the sandboxed runtime, then
//! shows (a) inference in the sandbox, (b) a module being stopped when it would
//! exceed the memory cap, and (c) hot-swapping a module at runtime. Self-
//! contained and offline; neutral toy input only.

use std::path::PathBuf;

use anyhow::{Context, Result};
use kelvane_runtime::{ExecutionLimits, ModuleRuntime};

fn first_existing(rel: &str) -> Option<PathBuf> {
    for base in ["", "../../", "../"] {
        let p = PathBuf::from(format!("{base}{rel}"));
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// A neutral toy observation: a small gradient across a 4x11x11 feature window.
fn toy_input() -> String {
    let mut data = Vec::with_capacity(4 * 11 * 11);
    for c in 0..4 {
        for i in 0..11 {
            for j in 0..11 {
                data.push(((c + i + j) as f32) / 40.0);
            }
        }
    }
    format!("{{\"data\":{}}}", serde_json::to_string(&data).unwrap())
}

fn main() -> Result<()> {
    let policy = first_existing("target/wasm32-wasip1/release/policy_module.wasm")
        .context("policy_module.wasm not built (cargo build --target wasm32-wasip1 --release -p policy-module)")?;
    let scripted = first_existing("target/wasm32-wasip1/release/scripted_module.wasm")
        .context("scripted_module.wasm not built")?;
    let model = first_existing("models/grid_policy.onnx")
        .context("grid_policy.onnx not found (train + export with kelvane-marl first)")?;

    println!("\n=== Kelvane demo ===\n");

    // --- (a) Inference in the sandbox ------------------------------------
    let mut rt = ModuleRuntime::new(ExecutionLimits::default())?;
    rt.load_model(&model, &[1, 4, 11, 11])?;
    println!(
        "[a] inference backend: {}",
        rt.model_backend().unwrap_or("none")
    );
    rt.load_module(&policy, "policy")?;
    let out = rt.invoke("policy", toy_input().as_bytes())?;
    let decision: serde_json::Value = serde_json::from_slice(&out)?;
    println!(
        "[a] policy ran in-sandbox -> action {}, confidence {:.3}",
        decision["action"], decision["confidence"]
    );

    // --- (b) Memory cap stops an oversized module ------------------------
    let tiny = ExecutionLimits {
        max_memory_bytes: 16 * 1024,
        ..ExecutionLimits::default()
    };
    let mut capped = ModuleRuntime::new(tiny)?;
    capped.load_module(&policy, "policy")?;
    match capped.invoke("policy", toy_input().as_bytes()) {
        Ok(_) => println!("[b] UNEXPECTED: module ran under a 16 KB cap"),
        Err(e) => println!("[b] module stopped by the memory cap, as expected: {e}"),
    }

    // --- (c) Hot-swap a module at runtime --------------------------------
    let mut swap = ModuleRuntime::new(ExecutionLimits::default())?;
    swap.load_model(&model, &[1, 4, 11, 11])?;
    swap.load_module(&scripted, "slot")?;
    let before: serde_json::Value = serde_json::from_slice(&swap.invoke("slot", b"{}")?)?;
    println!(
        "[c] before swap -> module \"{}\"",
        before["module"].as_str().unwrap_or("?")
    );
    swap.hot_swap("slot", &policy)?;
    let after: serde_json::Value =
        serde_json::from_slice(&swap.invoke("slot", toy_input().as_bytes())?)?;
    println!(
        "[c] after hot-swap -> module \"{}\", action {}",
        after["module"].as_str().unwrap_or("?"),
        after["action"]
    );

    println!("\n=== demo complete ===\n");
    Ok(())
}
