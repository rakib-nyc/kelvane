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
//! run inference in the sandbox. Uses a committed ONNX fixture and the built
//! `policy_module.wasm`; a missing artifact is a hard failure, never a skip.

mod common;

use common::{default_rt, fixture, module_wasm, zeros_observation};

#[test]
fn policy_runs_in_sandbox() {
    let mut rt = default_rt();
    rt.load_model(&fixture("policy_4x11x11.onnx"), &[1, 4, 11, 11])
        .unwrap();
    rt.load_module(&module_wasm("policy_module"), "policy")
        .unwrap();

    let input = zeros_observation(4 * 11 * 11);
    let out = rt.invoke("policy", input.as_bytes()).unwrap();
    let decision: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(decision["module"], "policy");
    assert!(decision["action"].is_number());
    assert!(decision["confidence"].as_f64().unwrap() >= 0.0);
}
