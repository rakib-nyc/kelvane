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

//! Host-owned neural-network inference backends.
//!
//! The runtime loads an ONNX model once and runs it on demand, exposing it to
//! sandboxed WebAssembly modules through the `kelvane::infer` host import (see
//! [`crate::host`]). The default backend is pure-Rust [`tract`](tract_onnx);
//! enabling the `cuda` feature adds an ONNX Runtime CUDA backend that is used
//! automatically when CUDA hardware is present and falls back to CPU otherwise.

use anyhow::Result;
use std::path::Path;

/// A loaded model with a fixed input shape, runnable on the selected backend.
pub struct Model {
    inner: Backend,
    input_len: usize,
    backend: &'static str,
}

// A model is loaded once and stored behind an `Arc`, so the size gap between the
// CPU and (ONNX Runtime session) GPU variants is irrelevant — not worth an extra
// box on the benchmarked CPU inference path.
#[cfg_attr(feature = "cuda", allow(clippy::large_enum_variant))]
enum Backend {
    Cpu(cpu::CpuModel),
    #[cfg(feature = "cuda")]
    Gpu(gpu::GpuModel),
}

/// Load an ONNX model for inference with the given fixed input shape (e.g.
/// `[1, 4, 11, 11]`). With the `cuda` feature the CUDA backend is tried first
/// and transparently falls back to CPU if unavailable.
pub fn load_model(path: &Path, input_shape: &[usize]) -> Result<Model> {
    let input_len: usize = input_shape.iter().product();

    #[cfg(feature = "cuda")]
    {
        match gpu::GpuModel::load(path, input_shape) {
            Ok(g) => {
                return Ok(Model {
                    inner: Backend::Gpu(g),
                    input_len,
                    backend: "cuda(onnxruntime)",
                });
            }
            Err(e) => {
                tracing::warn!("CUDA backend unavailable ({e}); falling back to CPU");
            }
        }
    }

    let c = cpu::CpuModel::load(path, input_shape)?;
    Ok(Model {
        inner: Backend::Cpu(c),
        input_len,
        backend: "cpu(tract)",
    })
}

impl Model {
    /// Number of `f32` values the model expects as input.
    pub fn input_len(&self) -> usize {
        self.input_len
    }

    /// Human-readable name of the active backend.
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// Run inference on a flat `input_len()` input, returning the output values.
    pub fn run(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_len {
            anyhow::bail!(
                "model input must be {} values, got {}",
                self.input_len,
                input.len()
            );
        }
        match &self.inner {
            Backend::Cpu(m) => m.run(input),
            #[cfg(feature = "cuda")]
            Backend::Gpu(m) => m.run(input),
        }
    }
}

// --------------------------------------------------------------------------
mod cpu {
    use anyhow::Result;
    use std::path::Path;
    use tract_onnx::prelude::*;

    pub struct CpuModel {
        plan: TypedRunnableModel<TypedModel>,
        shape: Vec<usize>,
    }

    impl CpuModel {
        pub fn load(path: &Path, shape: &[usize]) -> Result<Self> {
            let plan = tract_onnx::onnx()
                .model_for_path(path)?
                .with_input_fact(0, f32::fact(shape).into())?
                .into_optimized()?
                .into_runnable()?;
            Ok(Self {
                plan,
                shape: shape.to_vec(),
            })
        }

        pub fn run(&self, input: &[f32]) -> Result<Vec<f32>> {
            let arr = tract_ndarray::ArrayD::from_shape_vec(self.shape.clone(), input.to_vec())?;
            let tensor: Tensor = arr.into();
            let outputs = self.plan.run(tvec!(tensor.into()))?;
            Ok(outputs[0].as_slice::<f32>()?.to_vec())
        }
    }
}

// --------------------------------------------------------------------------
#[cfg(feature = "cuda")]
mod gpu {
    use anyhow::Result;
    use std::path::Path;
    use std::sync::Mutex;

    // ONNX Runtime's `Session::run` takes `&mut self`, so we guard the session
    // with a Mutex to keep `Model::run(&self)` uniform across backends.
    pub struct GpuModel {
        session: Mutex<ort::session::Session>,
        shape: Vec<i64>,
    }

    // ort 2.0's `Error<R>` embeds the (non-`Send`/`Sync`) builder handle it
    // failed on, so it can't cross a `?` into `anyhow::Error` directly; funnel
    // every ort error through its `Display` string instead.
    fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
        anyhow::anyhow!("ort: {e}")
    }

    impl GpuModel {
        pub fn load(path: &Path, shape: &[usize]) -> Result<Self> {
            let session = ort::session::Session::builder()
                .map_err(ort_err)?
                .with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default().build(),
                ])
                .map_err(ort_err)?
                .commit_from_file(path)
                .map_err(ort_err)?;
            Ok(Self {
                session: Mutex::new(session),
                shape: shape.iter().map(|&d| d as i64).collect(),
            })
        }

        pub fn run(&self, input: &[f32]) -> Result<Vec<f32>> {
            let tensor = ort::value::Tensor::from_array((self.shape.clone(), input.to_vec()))
                .map_err(ort_err)?;
            let mut session = self
                .session
                .lock()
                .map_err(|_| anyhow::anyhow!("inference session mutex poisoned"))?;
            let outputs = session.run(ort::inputs![tensor]).map_err(ort_err)?;
            let (_shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
            Ok(data.to_vec())
        }
    }
}
