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

//! The module host: loads, sandboxes, and invokes untrusted WebAssembly
//! modules, and exposes host-owned neural inference to them.

use crate::inference::{self, Model};
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tracing::info;
use wasmtime::*;
use wasmtime_wasi::p1::WasiP1Ctx;
use wasmtime_wasi::WasiCtxBuilder;

/// Per-invocation resource limits.
#[derive(Debug, Clone)]
pub struct ExecutionLimits {
    /// Hard cap on a single module's WASM linear memory (enforced by the runtime).
    pub max_memory_bytes: usize,
    /// CPU fuel budget per `process()` call (~ms of compute).
    pub fuel_per_call: u64,
    /// Maximum number of concurrently loaded modules.
    pub max_instances: usize,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: 64 * 1024 * 1024, // 64 MB
            fuel_per_call: 200_000_000,
            max_instances: 32,
        }
    }
}

/// Largest output payload the host will read back from a module (guards against
/// a buggy or malicious module returning an absurd length).
const MAX_OUTPUT_BYTES: usize = 1024 * 1024;

/// Decode a guest-returned packed `i64` (`ptr << 32 | len`) into an
/// `(offset, len)` region, enforcing the max-output cap and validating that the
/// region lies fully inside the guest's `mem_size`-byte linear memory.
///
/// This is the trust boundary for a module's return value — every field is
/// attacker-controlled — so it is a pure function with no I/O, exercised both by
/// [`ModuleRuntime::invoke`] and by the `abi_decode` fuzz target.
#[doc(hidden)]
pub fn decode_output_region(packed: i64, mem_size: usize) -> Result<(usize, usize)> {
    let out_ptr = ((packed >> 32) & 0xFFFF_FFFF) as usize;
    let out_len = (packed & 0xFFFF_FFFF) as usize;
    if out_len > MAX_OUTPUT_BYTES {
        anyhow::bail!("output length {out_len} exceeds cap {MAX_OUTPUT_BYTES}");
    }
    let end = out_ptr
        .checked_add(out_len)
        .ok_or_else(|| anyhow::anyhow!("output region overflow: {out_ptr} + {out_len}"))?;
    if end > mem_size {
        anyhow::bail!("output region out of bounds ({end} > {mem_size})");
    }
    Ok((out_ptr, out_len))
}

/// Decode a little-endian `f32` vector from raw guest bytes, ignoring a trailing
/// partial (`< 4`-byte) remainder. Pure and total — never panics on any input —
/// which the `infer_decode` fuzz target asserts. Used to marshal the guest's
/// inference request out of its linear memory.
#[doc(hidden)]
pub fn bytes_to_f32(buf: &[u8]) -> Vec<f32> {
    buf.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Per-`Store` state: the WASI context plus the memory limiter.
struct StoreState {
    wasi: WasiP1Ctx,
    limits: StoreLimits,
}

/// A runtime that executes untrusted, hot-swappable WebAssembly modules under a
/// per-invocation compute/authority budget, optionally serving them host-owned
/// neural inference.
pub struct ModuleRuntime {
    engine: Engine,
    limits: ExecutionLimits,
    modules: HashMap<String, LoadedModule>,
    model: Option<Arc<Model>>,
}

struct LoadedModule {
    module: Module,
    call_count: u64,
}

impl ModuleRuntime {
    /// Create a runtime with fuel metering enabled.
    pub fn new(limits: ExecutionLimits) -> Result<Self> {
        let mut config = Config::new();
        config.consume_fuel(true);
        config.cranelift_opt_level(OptLevel::Speed);
        Ok(Self {
            engine: Engine::new(&config)?,
            limits,
            modules: HashMap::new(),
            model: None,
        })
    }

    /// Load an ONNX model and make it available to modules via `kelvane::infer`.
    pub fn load_model(&mut self, onnx_path: &Path, input_shape: &[usize]) -> Result<()> {
        let model = inference::load_model(onnx_path, input_shape)?;
        info!(backend = model.backend(), "Loaded model for inference");
        self.model = Some(Arc::new(model));
        Ok(())
    }

    /// Name of the active inference backend, if a model is loaded.
    pub fn model_backend(&self) -> Option<&'static str> {
        self.model.as_ref().map(|m| m.backend())
    }

    /// Compile and register a WASM module under `id`.
    pub fn load_module(&mut self, wasm_path: &Path, id: &str) -> Result<()> {
        if self.modules.len() >= self.limits.max_instances {
            anyhow::bail!(
                "module limit reached ({}/{})",
                self.modules.len(),
                self.limits.max_instances
            );
        }
        let module = self.compile(wasm_path, id)?;
        self.modules.insert(
            id.to_string(),
            LoadedModule {
                module,
                call_count: 0,
            },
        );
        Ok(())
    }

    /// Replace a loaded module in place, preserving its call statistics. Because
    /// every `invoke` instantiates fresh from the stored module, the next call
    /// transparently runs the new module — no runtime restart.
    pub fn hot_swap(&mut self, id: &str, wasm_path: &Path) -> Result<()> {
        // Compile first so a bad module can't corrupt a live slot.
        let module = self.compile(wasm_path, id)?;
        let slot = self
            .modules
            .get_mut(id)
            .ok_or_else(|| anyhow::anyhow!("module {id} not loaded; cannot hot-swap"))?;
        slot.module = module;
        info!(module = id, "Hot-swapped module");
        Ok(())
    }

    fn compile(&self, wasm_path: &Path, id: &str) -> Result<Module> {
        let bytes = std::fs::read(wasm_path)?;
        let module = Module::new(&self.engine, &bytes)?;
        info!(
            module = id,
            size_kb = bytes.len() / 1024,
            "Compiled WASM module"
        );
        Ok(module)
    }

    /// Build a fresh, sandboxed store for one invocation.
    ///
    /// Capability posture: **no ambient authority**. The guest gets no
    /// filesystem, no inherited stdio, and no network. Memory is hard-capped at
    /// `max_memory_bytes` via a [`ResourceLimiter`], and CPU at `fuel_per_call`.
    fn new_store(&self) -> Result<Store<StoreState>> {
        let wasi = WasiCtxBuilder::new().build_p1();
        let limits = StoreLimitsBuilder::new()
            .memory_size(self.limits.max_memory_bytes)
            .instances(1)
            .memories(1)
            .build();
        let mut store = Store::new(&self.engine, StoreState { wasi, limits });
        store.limiter(|s| &mut s.limits);
        store.set_fuel(self.limits.fuel_per_call)?;
        Ok(store)
    }

    /// Invoke a module's `process` function with `input` bytes and return the
    /// output bytes, enforcing the per-call memory/CPU sandbox.
    pub fn invoke(&mut self, id: &str, input: &[u8]) -> Result<Vec<u8>> {
        let module = {
            let slot = self
                .modules
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("module {id} not found"))?;
            slot.module.clone()
        };

        let mut store = self.new_store()?;
        let mut linker = Linker::new(&self.engine);
        wasmtime_wasi::p1::add_to_linker_sync(&mut linker, |s: &mut StoreState| &mut s.wasi)?;

        // Host-owned inference capability. Modules that don't import it are
        // unaffected; a module that imports it with no model loaded gets -1 and
        // is expected to degrade gracefully.
        let model = self.model.clone();
        linker.func_wrap(
            "kelvane",
            "infer",
            move |mut caller: Caller<'_, StoreState>,
                  in_ptr: i32,
                  in_len: i32,
                  out_ptr: i32,
                  out_cap: i32|
                  -> i32 {
                let model = match &model {
                    Some(m) => m.clone(),
                    None => return -1,
                };
                if in_len <= 0 || out_cap <= 0 {
                    return -1;
                }
                let mem = match caller.get_export("memory") {
                    Some(Extern::Memory(m)) => m,
                    _ => return -1,
                };
                // Bound the host-side allocation by the guest's actual memory:
                // `in_len` is attacker-controlled, so without this a guest could
                // request `in_len * 4` ~= 8 GiB and OOM the host before the read
                // ever fails. You can never read more bytes than the guest holds.
                let need = (in_len as usize).saturating_mul(4);
                if need > mem.data_size(&caller) {
                    return -1;
                }
                let mut buf = vec![0u8; need];
                if mem.read(&caller, in_ptr as usize, &mut buf).is_err() {
                    return -1;
                }
                let input: Vec<f32> = bytes_to_f32(&buf);
                let out = match model.run(&input) {
                    Ok(o) => o,
                    Err(_) => return -1,
                };
                let n = out.len().min(out_cap as usize);
                let mut out_bytes = Vec::with_capacity(n * 4);
                for v in &out[..n] {
                    out_bytes.extend_from_slice(&v.to_le_bytes());
                }
                if mem
                    .write(&mut caller, out_ptr as usize, &out_bytes)
                    .is_err()
                {
                    return -1;
                }
                n as i32
            },
        )?;

        let instance = linker.instantiate(&mut store, &module)?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow::anyhow!("module {id} has no `memory` export"))?;
        let alloc_fn = instance.get_typed_func::<i32, i32>(&mut store, "module_alloc")?;
        let process_fn = instance.get_typed_func::<(i32, i32), i64>(&mut store, "process")?;

        if input.len() > i32::MAX as usize {
            anyhow::bail!("input too large: {} bytes", input.len());
        }
        let in_ptr = alloc_fn.call(&mut store, input.len() as i32)?;
        if in_ptr == 0 {
            anyhow::bail!("guest module_alloc returned null for {} bytes", input.len());
        }
        memory.write(&mut store, in_ptr as usize, input)?;

        let packed = process_fn.call(&mut store, (in_ptr, input.len() as i32))?;
        let mem_size = memory.data_size(&store);
        let (out_ptr, out_len) = decode_output_region(packed, mem_size)
            .map_err(|e| anyhow::anyhow!("module {id}: {e}"))?;
        let mut result = vec![0u8; out_len];
        memory.read(&store, out_ptr, &mut result)?;

        if let Ok(dealloc_fn) =
            instance.get_typed_func::<(i32, i32), ()>(&mut store, "module_dealloc")
        {
            let _ = dealloc_fn.call(&mut store, (in_ptr, input.len() as i32));
        }

        let fuel_left = store.get_fuel()?;
        let used = self.limits.fuel_per_call.saturating_sub(fuel_left);
        if let Some(slot) = self.modules.get_mut(id) {
            slot.call_count += 1;
            info!(
                module = id,
                fuel_used = used,
                call = slot.call_count,
                "Invocation complete"
            );
        }
        Ok(result)
    }

    /// Number of times a module has been invoked.
    pub fn call_count(&self, id: &str) -> Option<u64> {
        self.modules.get(id).map(|m| m.call_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Artifacts are REQUIRED: a missing one fails the test (never a silent skip),
    // so a green run always means the sandbox was actually exercised. CI builds
    // the wasm modules before testing; the ONNX model is a committed fixture.
    fn module_wasm(name: &str) -> PathBuf {
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

    fn model_path() -> PathBuf {
        for base in ["tests/models", "crates/kelvane-runtime/tests/models"] {
            let p = PathBuf::from(format!("{base}/policy_4x11x11.onnx"));
            if p.exists() {
                return p;
            }
        }
        panic!(
            "required ONNX fixture 'policy_4x11x11.onnx' not found; \
             regenerate with crates/kelvane-runtime/tests/models/generate.py"
        );
    }

    #[test]
    fn scripted_module_roundtrips() {
        let wasm = module_wasm("scripted_module");
        let mut rt = ModuleRuntime::new(ExecutionLimits::default()).unwrap();
        rt.load_module(&wasm, "scripted").unwrap();
        let out = rt.invoke("scripted", b"{}").unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["module"], "scripted");
        assert_eq!(rt.call_count("scripted"), Some(1));
    }

    #[test]
    fn memory_limit_blocks_oversized_module() {
        let wasm = module_wasm("scripted_module");
        // Cap memory below the module's initial linear memory so instantiation is
        // rejected by the ResourceLimiter.
        let limits = ExecutionLimits {
            max_memory_bytes: 16 * 1024,
            ..ExecutionLimits::default()
        };
        let mut rt = ModuleRuntime::new(limits).unwrap();
        rt.load_module(&wasm, "tiny").unwrap();
        let result = rt.invoke("tiny", b"{}");
        assert!(
            result.is_err(),
            "expected memory-limit rejection, got {result:?}"
        );
    }

    #[test]
    fn hot_swap_replaces_module() {
        let scripted = module_wasm("scripted_module");
        let policy = module_wasm("policy_module");
        let model = model_path();
        let mut rt = ModuleRuntime::new(ExecutionLimits::default()).unwrap();
        rt.load_model(&model, &[1, 4, 11, 11]).unwrap();
        rt.load_module(&scripted, "slot").unwrap();

        let before: serde_json::Value =
            serde_json::from_slice(&rt.invoke("slot", b"{}").unwrap()).unwrap();
        assert_eq!(before["module"], "scripted");

        rt.hot_swap("slot", &policy).unwrap();
        let obs = observation_json(&vec![0.05_f32; 484]);
        let after: serde_json::Value =
            serde_json::from_slice(&rt.invoke("slot", obs.as_bytes()).unwrap()).unwrap();
        assert_eq!(after["module"], "policy");
        assert_eq!(rt.call_count("slot"), Some(2));
    }

    #[test]
    fn policy_module_runs_model() {
        let wasm = module_wasm("policy_module");
        let model = model_path();
        let mut rt = ModuleRuntime::new(ExecutionLimits::default()).unwrap();
        rt.load_model(&model, &[1, 4, 11, 11]).unwrap();
        rt.load_module(&wasm, "policy").unwrap();
        let obs = observation_json(&vec![0.05_f32; 484]);
        let out: serde_json::Value =
            serde_json::from_slice(&rt.invoke("policy", obs.as_bytes()).unwrap()).unwrap();
        assert_eq!(out["module"], "policy");
        assert!(out["action"].is_number());
    }

    fn observation_json(data: &[f32]) -> String {
        format!("{{\"data\":{}}}", serde_json::to_string(data).unwrap())
    }

    // --- pure decode helpers (the ABI trust boundary; also fuzzed) ----------

    #[test]
    fn decode_output_rejects_over_cap() {
        assert!(decode_output_region((MAX_OUTPUT_BYTES as i64) + 1, usize::MAX).is_err());
    }

    #[test]
    fn decode_output_rejects_out_of_bounds() {
        // ptr=0, len=100 but memory only 10 bytes.
        assert!(decode_output_region(100, 10).is_err());
    }

    #[test]
    fn decode_output_accepts_in_bounds() {
        // ptr=8, len=4, memory=64 -> Ok(8,4).
        let packed = ((8_i64) << 32) | 4;
        assert_eq!(decode_output_region(packed, 64).unwrap(), (8, 4));
    }

    #[test]
    fn decode_output_max_pointer_is_oob_not_panic() {
        let packed = ((0xFFFF_FFFF_i64) << 32) | (1024 * 1024);
        assert!(decode_output_region(packed, 64 * 1024).is_err());
    }

    #[test]
    fn bytes_to_f32_ignores_partial_tail_and_never_panics() {
        assert_eq!(bytes_to_f32(&[]).len(), 0);
        assert_eq!(bytes_to_f32(&[1, 2, 3]).len(), 0); // < 4 bytes
        assert_eq!(bytes_to_f32(&[0, 0, 0, 0, 1]).len(), 1); // 5 bytes -> 1 f32
        assert_eq!(bytes_to_f32(&0.0_f32.to_le_bytes()), vec![0.0]);
    }
}
