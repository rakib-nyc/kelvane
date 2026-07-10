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

//! Example Kelvane module: runs a neural policy inside the sandbox.
//!
//! It reads a JSON observation `{"data": [f32, ...]}`, ships the feature vector
//! to the host's inference capability (`kelvane::infer`), argmaxes the returned
//! scores, and returns a JSON decision. All memory plumbing lives in
//! [`kelvane_sdk`]; this module only supplies the decision logic.

// `#[unsafe(no_mangle)]` exports (from `export_module!`) and the host import
// trip the unsafe_code lint; WebAssembly modules expect this.
#![allow(unsafe_code)]

use kelvane_sdk::export_module;
use serde_json::json;

/// Fixed model input length: 4 channels x 11 x 11.
/// Upper bound on the number of action scores this module reads back. The guest
/// is shape-agnostic — it forwards however many features it is given and accepts
/// up to this many scores — so one module works across models of different input
/// and output sizes (the host owns the actual shape).
const MAX_ACTIONS: usize = 64;

#[link(wasm_import_module = "kelvane")]
extern "C" {
    /// Host inference call: reads `in_len` f32 at `in_ptr`, runs the loaded
    /// model, writes up to `out_cap` f32 scores at `out_ptr`. Returns the number
    /// of scores written, or a negative value on error.
    fn infer(in_ptr: i32, in_len: i32, out_ptr: i32, out_cap: i32) -> i32;
}

fn handle(input: &[u8]) -> Vec<u8> {
    let value: serde_json::Value = match serde_json::from_slice(input) {
        Ok(v) => v,
        Err(_) => return fallback("invalid_input"),
    };
    // Forward the feature vector as-is; its length must match the loaded model's
    // input size, which the host enforces (a mismatch comes back as an error).
    let features: Vec<f32> = value["data"]
        .as_array()
        .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
        .unwrap_or_default();
    if features.is_empty() {
        return fallback("invalid_input");
    }

    let mut scores = [0f32; MAX_ACTIONS];
    let n = unsafe {
        infer(
            features.as_ptr() as i32,
            features.len() as i32,
            scores.as_mut_ptr() as i32,
            scores.len() as i32,
        )
    };
    if n <= 0 {
        return fallback("inference_unavailable");
    }
    let scores = &scores[..n as usize];

    // Softmax for a calibrated confidence; argmax for the chosen action.
    let max = scores.iter().copied().fold(f32::MIN, f32::max);
    let exp: Vec<f32> = scores.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let mut best = 0usize;
    for i in 1..scores.len() {
        if scores[i] > scores[best] {
            best = i;
        }
    }
    let confidence = if sum > 0.0 { exp[best] / sum } else { 0.0 };

    serde_json::to_vec(&json!({
        "module": "policy",
        "action": best,
        "confidence": confidence,
    }))
    .unwrap_or_else(|_| b"{}".to_vec())
}

fn fallback(note: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "module": "policy",
        "action": 0,
        "confidence": 0.0,
        "note": note,
    }))
    .unwrap_or_else(|_| b"{}".to_vec())
}

export_module!(handle);
