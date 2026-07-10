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

//! End-to-end sandboxed-call latency benchmark.
//!
//! Unlike `bench.rs` (which times only host-side `Model::run`), this measures a
//! **full** `ModuleRuntime::invoke` call: fresh sandbox store construction,
//! WASI + inference linker setup, instantiation, the guest `module_alloc` +
//! input copy, the `process` ABI round trip (including the guest's call back
//! into the host's `kelvane::infer`), reading the packed output, `module_dealloc`,
//! and store teardown. This is the latency a caller actually pays per decision.
//!
//! Run: `cargo run --release --example bench_e2e -p kelvane-runtime`
//!  or: `cargo run --release --features cuda --example bench_e2e -p kelvane-runtime`

use std::path::PathBuf;
use std::time::Instant;

use kelvane_runtime::{ExecutionLimits, ModuleRuntime};

fn find(rel: &str) -> Option<PathBuf> {
    for base in ["../../", "", "../"] {
        let p = PathBuf::from(format!("{base}{rel}"));
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// A neutral toy observation matching the model's 4x11x11 input window.
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

/// Nearest-rank percentile over an ascending-sorted slice.
fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

fn main() -> anyhow::Result<()> {
    let model = find("models/grid_policy.onnx").ok_or_else(|| {
        anyhow::anyhow!("models/grid_policy.onnx not found; train + export first")
    })?;
    let wasm = find("target/wasm32-wasip1/release/policy_module.wasm").ok_or_else(|| {
        anyhow::anyhow!(
            "policy_module.wasm not built (cargo build --release \
             --target wasm32-wasip1 -p policy-module)"
        )
    })?;

    let mut rt = ModuleRuntime::new(ExecutionLimits::default())?;
    rt.load_model(&model, &[1, 4, 11, 11])?;
    rt.load_module(&wasm, "policy")?;
    println!("backend: {}", rt.model_backend().unwrap_or("none"));
    println!("path:    full ModuleRuntime::invoke (store + ABI + infer + teardown)");

    let input = toy_input();
    let bytes = input.as_bytes();

    // Sanity: confirm the module actually produced a decision before timing.
    let first = rt.invoke("policy", bytes)?;
    let decision: serde_json::Value = serde_json::from_slice(&first)?;
    if decision["module"] != "policy" || !decision["action"].is_number() {
        anyhow::bail!("unexpected module output: {decision}");
    }

    // Warm-up.
    for _ in 0..50 {
        let _ = rt.invoke("policy", bytes)?;
    }

    let iters = 2000;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = rt.invoke("policy", bytes)?;
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let median = samples[samples.len() / 2];
    println!("iters:  {iters}");
    println!("min:    {:.1} us", samples[0]);
    println!("mean:   {mean:.1} us");
    println!("median: {median:.1} us");
    println!("p50:    {:.1} us", pct(&samples, 50.0));
    println!("p95:    {:.1} us", pct(&samples, 95.0));
    println!("p99:    {:.1} us", pct(&samples, 99.0));
    println!("max:    {:.1} us", samples[samples.len() - 1]);
    println!("throughput (mean): {:.0} calls/s", 1e6 / mean);
    Ok(())
}
