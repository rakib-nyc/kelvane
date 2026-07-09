# kelvane-runtime

A safety-first runtime for **untrusted, hot-swappable WebAssembly modules** with
a per-invocation compute/authority budget and **host-owned neural inference**.

Every invocation runs in a **fresh sandbox store** carrying:

- a hard **memory ceiling** (enforced by a `ResourceLimiter`),
- a per-call **CPU fuel budget**,
- a **zero-ambient-authority** WASI context (no filesystem, no network, no stdio),
- and access to a **host-owned inference** capability (`kelvane::infer`).

Modules can be **hot-swapped** at runtime — the next invocation transparently
runs the new module, no restart.

## Inference backends

- **CPU (default):** pure-Rust [`tract`](https://crates.io/crates/tract-onnx).
- **CUDA (optional):** ONNX Runtime's CUDA execution provider behind the `cuda`
  Cargo feature, with automatic fallback to CPU when CUDA hardware is absent.

```bash
cargo build --release                       # CPU backend
cargo build --release --features cuda        # + CUDA backend (needs the CUDA toolchain)
```

## Usage

```rust
use kelvane_runtime::{ExecutionLimits, ModuleRuntime};

let mut rt = ModuleRuntime::new(ExecutionLimits::default())?;
rt.load_model(std::path::Path::new("models/grid_policy.onnx"), &[1, 4, 11, 11])?;
rt.load_module(std::path::Path::new("policy_module.wasm"), "policy")?;
let out = rt.invoke("policy", br#"{"data":[/* features */]}"#)?;
```

## License

Apache License 2.0. See the repository `LICENSE` and `NOTICE`.
