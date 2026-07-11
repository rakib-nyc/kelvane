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

//! Generality test: run a model that is NOT the gridworld policy end-to-end
//! through the sandbox.
//!
//! The model is a small CNN trained on the scikit-learn `digits` dataset (8x8
//! handwritten digits, 10 classes) — a different domain and a different input
//! shape (`[1, 1, 8, 8]`) than the gridworld policy (`[1, 4, 11, 11]`). It is a
//! real trained classifier, so a correct prediction is a *sensible* output, not
//! noise. See `tests/models/generate_digits.py` for provenance and license.
//!
//! It runs through the same generalized `policy_module.wasm` guest: the guest
//! forwards a flat feature vector to `kelvane::infer` and argmaxes the returned
//! scores, so its `"action"` field is the predicted digit class.

mod common;

use common::{default_rt, fixture, module_wasm};

#[test]
fn digits_cnn_classifies_through_the_sandbox() {
    // A real, trained, non-gridworld model of a different input shape.
    let mut rt = default_rt();
    rt.load_model(&fixture("digits_cnn.onnx"), &[1, 1, 8, 8])
        .unwrap();
    rt.load_module(&module_wasm("policy_module"), "clf")
        .unwrap();

    let samples: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture("digits_samples.json")).unwrap()).unwrap();
    let samples = samples["samples"].as_array().expect("samples array");
    assert!(!samples.is_empty(), "fixture must contain sample digits");

    let mut correct = 0usize;
    for s in samples {
        let data = &s["data"]; // flat 64 f32, normalized
        let label = s["label"].as_u64().expect("label");
        let input = format!("{{\"data\":{data}}}");

        let out = rt.invoke("clf", input.as_bytes()).unwrap();
        let decision: serde_json::Value = serde_json::from_slice(&out).unwrap();

        // Output must be well-formed and a valid class in 0..=9.
        let predicted = decision["action"].as_u64().expect("action is a number");
        assert!(
            predicted <= 9,
            "class {predicted} out of range for 10-way digits"
        );
        if predicted == label {
            correct += 1;
        }
    }

    // A trained classifier gets the held-out samples right; random would be ~10%.
    // Require a clear majority so the test proves real classification, not chance,
    // while tolerating an occasional miss.
    let n = samples.len();
    assert!(
        correct * 3 >= n * 2,
        "expected a trained classifier to get >= 2/3 of {n} digits right, got {correct}"
    );
}
