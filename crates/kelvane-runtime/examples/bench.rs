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

//! Measure inference latency of the active backend (CPU by default, CUDA with
//! `--features cuda`). Prints mean / median / p95 over many iterations.
//!
//! Run: `cargo run --release --example bench -p kelvane-runtime`
//!  or: `cargo run --release --features cuda --example bench -p kelvane-runtime`

use std::path::PathBuf;
use std::time::Instant;

use kelvane_runtime::inference::{load_model, Model};

fn find(rel: &str) -> Option<PathBuf> {
    for base in ["../../", "", "../"] {
        let p = PathBuf::from(format!("{base}{rel}"));
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn main() -> anyhow::Result<()> {
    let path = find("models/grid_policy.onnx").ok_or_else(|| {
        anyhow::anyhow!("models/grid_policy.onnx not found; train + export first")
    })?;
    let model: Model = load_model(&path, &[1, 4, 11, 11])?;
    println!("backend: {}", model.backend());

    let input = vec![0.05_f32; model.input_len()];
    // warm-up
    for _ in 0..50 {
        let _ = model.run(&input)?;
    }
    let iters = 2000;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = model.run(&input)?;
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let median = samples[samples.len() / 2];
    let p95 = samples[(samples.len() as f64 * 0.95) as usize];
    println!("iters: {iters}");
    println!("mean:   {mean:.1} us");
    println!("median: {median:.1} us");
    println!("p95:    {p95:.1} us");
    println!("throughput: {:.0} inferences/s", 1e6 / mean);
    Ok(())
}
