# Changelog

All notable changes to Kelvane are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and Kelvane adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Threat model** (`SECURITY.md`): what the sandbox does and does not protect
  against, each protection tied to a mechanism and a test; trust boundary,
  assumptions, and responsible-disclosure guidance.
- **Generality test**: a small CNN trained on the scikit-learn `digits` dataset
  (input shape `[1,1,8,8]`, ~97% accuracy) classified end-to-end through the
  sandbox, demonstrating Kelvane runs models beyond its own gridworld toy.
- **`no_ambient_filesystem_authority`** adversarial test: a guest's WASI
  filesystem call is denied (no preopens), backing the zero-authority claim.
- Full rustdoc on the public API with doctests, and a documented, conservative
  API-stability statement in the README.

### Changed
- **API surface made intentional.** The pure ABI-decode functions
  (`decode_output_region`, `bytes_to_f32`), previously `pub #[doc(hidden)]` for
  fuzzing, are now `pub(crate)` and reachable only via the off-by-default
  `internals` feature (fuzz-only, no semver guarantee).
- `#![deny(missing_docs)]` on `kelvane-runtime` and `kelvane-sdk`; `cargo test
  --doc` enforced in CI.

### Security
- **Upgraded `wasmtime` / `wasmtime-wasi` 29 → 46.0.1**, resolving 19 RustSec
  advisories against the old engine — including two critical sandbox escapes
  (RUSTSEC-2026-0095, RUSTSEC-2026-0096) and a HIGH WASI `path_open` permission
  bypass (RUSTSEC-2026-0149). `cargo audit` is now clean (only a transitive
  `paste` *unmaintained* warning remains) and is a blocking CI gate. The
  zero-ambient-authority WASI context and all 29 adversarial containment cases
  (memory cap, fuel trap, OOB, infer-DoS bound) were re-verified on the new
  engine.

### Changed
- Relicensed from PolyForm Noncommercial 1.0.0 to Apache License 2.0.
- **MSRV raised 1.88 → 1.94.0** (required by wasmtime 46 / cranelift 0.133,
  edition 2024). Determined empirically; the MSRV CI row pins exactly 1.94.0.
- Benchmarks re-measured on wasmtime 46: end-to-end CPU per-call latency moved
  from ~161 µs to ~175 µs (≈9% higher engine instantiation cost); inference-only
  (~16 µs CPU) is unchanged. See the README.

## [0.1.0] — 2026-07-08

### Added
- **`kelvane-runtime`** — a runtime for untrusted, hot-swappable WebAssembly
  modules with a per-call memory ceiling, a per-call CPU fuel budget, a
  zero-ambient-authority WASI context, per-call fresh-store execution, module
  hot-swap, and a host-owned neural-inference capability. CPU backend via
  `tract`; optional CUDA backend via ONNX Runtime behind the `cuda` feature with
  automatic CPU fallback.
- **`kelvane-sdk`** — the guest SDK / ABI: a length-delimited packed
  pointer+length return, host-driven allocation, and an `export_module!` macro.
- **`kelvane-marl`** — a multi-agent RL reference: a cooperative gridworld, a
  vectorized (GPU-capable) environment, MAPPO and QMIX trainers, and ONNX policy
  export (FP32 + INT8).
- **`examples/kelvane-demo`** — an end-to-end demo: train → export → run a policy
  in the sandbox, hit the memory cap, and hot-swap a module.
- Unit + integration tests (including the memory-limit and hot-swap tests),
  pytest tests for a training smoke run and ONNX export, and a CI workflow.
