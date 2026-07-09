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

//! End-to-end integration test over the public API: load a model + module and
//! run inference in the sandbox. Skips gracefully if the artifacts aren't built.

use std::path::PathBuf;

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

#[test]
fn policy_runs_in_sandbox() {
    let (Some(wasm), Some(model)) = (
        find("target/wasm32-wasip1/release/policy_module.wasm"),
        find("models/grid_policy.onnx"),
    ) else {
        println!("skip: policy module and/or model not present");
        return;
    };
    let mut rt = ModuleRuntime::new(ExecutionLimits::default()).unwrap();
    rt.load_model(&model, &[1, 4, 11, 11]).unwrap();
    rt.load_module(&wasm, "policy").unwrap();

    let data: Vec<f32> = vec![0.05; 4 * 11 * 11];
    let input = format!("{{\"data\":{}}}", serde_json::to_string(&data).unwrap());
    let out = rt.invoke("policy", input.as_bytes()).unwrap();
    let decision: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(decision["module"], "policy");
    assert!(decision["action"].is_number());
    assert!(decision["confidence"].as_f64().unwrap() >= 0.0);
}
