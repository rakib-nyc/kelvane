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

//! Example Kelvane module: a fixed scripted response, no inference.
//!
//! Useful as a lightweight contrast for the hot-swap demo (swap this for the
//! neural `policy-module` at runtime) and as a minimal reference module.

// `#[unsafe(no_mangle)]` exports (from `export_module!`) trip the unsafe_code lint.
#![allow(unsafe_code)]

use kelvane_sdk::export_module;
use serde_json::json;

fn handle(_input: &[u8]) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "module": "scripted",
        "action": 0,
        "confidence": 1.0,
    }))
    .unwrap_or_else(|_| b"{}".to_vec())
}

export_module!(handle);
