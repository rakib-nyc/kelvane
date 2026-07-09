# Changelog

All notable changes to Kelvane are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and Kelvane adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Relicensed from PolyForm Noncommercial 1.0.0 to Apache License 2.0.

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
