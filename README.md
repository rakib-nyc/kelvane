# Kelvane

[![CI](https://github.com/rakib-nyc/kelvane/actions/workflows/ci.yml/badge.svg)](https://github.com/rakib-nyc/kelvane/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-79%25-green)](#supported-platforms--toolchains)
[![MSRV](https://img.shields.io/badge/MSRV-1.94.0-blue)](#supported-platforms--toolchains)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Run neural policies you don't fully trust — safely, and swap them live.**

Kelvane is a small, honest toolkit for **safety-first neural inference on
WebAssembly**. It runs untrusted `.wasm` modules inside a sandbox that enforces a
**per-invocation compute and authority budget**, gives them **host-owned neural
inference** (CPU or CUDA), and lets you **hot-swap** a module at runtime — plus a
compact, reproducible **multi-agent reinforcement learning** reference to produce
policies to run.

> Author: **Muhammad Rakibul Islam**. License: Apache-2.0.
> No networking. No security primitives. General-purpose research infrastructure
> built from publicly known techniques.

## Why this exists

Running a neural policy — especially one you didn't write or can't fully audit —
means running someone else's code. Kelvane's contribution is a **safety-first
place to run it**: each decision executes in a fresh sandbox with a hard memory
ceiling, a per-call CPU budget, and **zero ambient authority** (no filesystem, no
network, no stdio), and the model itself is owned by the host — the module only
gets to ask for an inference, never touches the weights or the accelerator
directly. You can replace the running module mid-flight without a restart. The
included RL framework is a self-contained way to **train, export, and then run** a
policy through that boundary end to end.

## Components

| Component | What it does |
|---|---|
| **`kelvane-runtime`** (Rust) | Loads untrusted, hot-swappable WebAssembly modules; enforces a per-call memory ceiling, CPU fuel budget, and zero-ambient-authority sandbox; serves host-owned inference (CPU via `tract`, optional CUDA via ONNX Runtime). |
| **`kelvane-sdk`** (Rust) | The guest ABI: a length-delimited packed pointer+length return, host-driven allocation, and an `export_module!` macro. |
| **`kelvane-marl`** (Python) | A cooperative gridworld, a GPU-capable vectorized environment, MAPPO and QMIX trainers, and ONNX policy export. |

## 60-second quickstart

```bash
# 1) train a policy and export it to ONNX (GPU used automatically if present)
cd kelvane-marl
pip install numpy torch gymnasium pettingzoo onnx onnxruntime
python -m kelvane_marl.mappo --updates 120 --envs 64
python -m kelvane_marl.export_onnx
cd ..

# 2) build the example modules for the wasm32-wasip1 platform
rustup target add wasm32-wasip1
cargo build --release --target wasm32-wasip1 -p policy-module -p scripted-module

# 3) run the end-to-end demo: inference in the sandbox, the memory cap, hot-swap
cargo run --release -p kelvane-demo
```

## Architecture

```
        kelvane-marl (Python)                      kelvane-runtime (Rust)
   ┌───────────────────────────┐            ┌──────────────────────────────────┐
   │ gridworld → MAPPO / QMIX  │  ONNX      │  sandbox (fresh store per call)   │
   │ vectorized env (GPU)      │ ───────►   │   • memory ceiling                │
   │ export FP32 / INT8        │  model     │   • CPU fuel budget               │
   └───────────────────────────┘            │   • zero ambient authority        │
                                            │   • hot-swap                      │
   input bytes ──► module_alloc ──► process │   host-owned inference:           │
        ▲                              │     │     kelvane::infer ──► CPU / CUDA │
        └──────── output bytes ◄───────┘     └──────────────────────────────────┘
```

The guest ships bytes in and gets bytes out through the `kelvane-sdk` ABI; when a
module needs a neural decision it calls the host's `kelvane::infer`, which runs
the ONNX model on the selected backend and returns the scores.

## The train → export → sandbox demo

`cargo run --release -p kelvane-demo` shows the three core behaviors:

- **(a) inference in the sandbox** — the exported policy runs inside a WASM module
  and returns a decision;
- **(b) the memory cap** — a module run under a 16 KB memory ceiling is stopped
  before it can exceed it;
- **(c) hot-swap** — a scripted module is swapped for the neural policy at
  runtime, and the next call transparently runs the new one.

## How it compares

Kelvane is a thin, opinionated layer on top of **Wasmtime**, not a replacement
for it, and not a general plugin framework.

- **Wasmtime** is the WebAssembly engine Kelvane is built on. Kelvane adds the
  opinionated policy: a *per-invocation* memory + CPU budget, a fresh store per
  call, a host-owned inference capability, and safe hot-swap.
- **extism** is a general plugin system for many host languages. Kelvane is
  narrower on purpose: it focuses on running *inference policies* under a
  per-call compute/authority budget rather than being a broad plugin ABI.
- **Wassette** is a WebAssembly-based runtime for agent tools. Kelvane does not
  aim to be an agent/tool runtime; it is the smaller building block that runs a
  single untrusted policy safely, with the model owned by the host.

If you need a full plugin framework or tool runtime, use those. If you want a
minimal, auditable place to run a neural policy under a hard per-call budget with
host-owned inference, that is what Kelvane is.

## Benchmarks

Two things are measured, because they answer different questions:

- **End-to-end** (`bench_e2e`) — a full `ModuleRuntime::invoke`: fresh sandbox
  store, WASI + inference linker setup, instantiation, the guest `module_alloc` +
  input copy, the `process` ABI round trip (including the guest's call back into
  host `kelvane::infer`), reading the packed output, `module_dealloc`, and store
  teardown. **This is the latency a caller actually pays per decision.**
- **Inference-only** (`bench`) — just the host-side `Model::run` (one ONNX
  forward pass), for comparison. It is *not* the per-call cost.

Exported `[1,4,11,11]` policy, 2000 iterations after 50 warm-up iterations.

**Environment.** AMD Ryzen 7 3700X (8C/16T) · NVIDIA RTX 2080 SUPER (driver
610.62, CUDA 12.x) · Ubuntu 24.04.2 on WSL2 (kernel 6.18, Windows 10) · rustc
1.94.1 · wasmtime 46.0.1 · tract-onnx 0.21.12 · ort 2.0.0-rc.12 (ONNX Runtime,
CUDA EP). All figures are from this one machine.

| Path | Backend | median | mean | p95 | p99 | throughput |
|---|---|---|---|---|---|---|
| **End-to-end** | CPU (`tract`) | **175 µs** | 184 µs | 227 µs | ~290 µs | ~5,400 calls/s |
| **End-to-end** | GPU (CUDA/ORT) | 345 µs | 353 µs | 400 µs | 447 µs | ~2,830 calls/s |
| Inference-only | CPU (`tract`) | 16.3 µs | 17.0 µs | 21.5 µs | ~32 µs | ~58,900 inf/s |
| Inference-only | GPU (CUDA/ORT) | 158 µs | 164 µs | 192 µs | 224 µs | ~6,080 inf/s |

Two honest takeaways for *this* small model on *this* machine:

1. **The full sandboxed call is ~11× the inference number** (175 µs vs 16.3 µs on
   CPU). Per-call store construction, instantiation, and the ABI round trip — not
   inference — dominate. (On wasmtime 46 this end-to-end cost is ~9% higher than
   on the old wasmtime 29 — the engine upgrade that fixed the sandbox-escape CVEs
   moved per-call instantiation from ~161 µs to ~175 µs; inference is unchanged.)
2. **The GPU path is real but slower here — about 10× slower than CPU** (158 µs vs
   16.3 µs inference-only). Expected for a tiny `[1,4,11,11]→7` model: kernel
   launch and host⇄device transfer swamp the compute, and `tract`'s optimized CPU
   path wins. CUDA is present for larger models, not as a speed-up for this one.

The `cuda` feature depends on ONNX Runtime, which `ort` fetches over TLS at build
time; that pulls in `openssl-sys`, so a CUDA build needs the OpenSSL development
headers and `pkg-config` (e.g. `apt-get install libssl-dev pkg-config` on
Debian/Ubuntu) plus, at run time, the CUDA runtime and cuDNN 9 on the library
path. Reproduce with:

```bash
cargo run --release --example bench_e2e -p kelvane-runtime            # CPU, end-to-end
cargo run --release --example bench      -p kelvane-runtime            # CPU, inference-only
cargo run --release --features cuda --example bench_e2e -p kelvane-runtime   # GPU
```

These are not general performance claims — measure your own model and hardware.

## Runs models beyond its own toy

Kelvane is not tied to the gridworld policy it ships. The generality test
(`crates/kelvane-runtime/tests/generality.rs`) loads a small **CNN trained on the
scikit-learn `digits` dataset** (8×8 handwritten digits, 10 classes — a different
domain and a different input shape, `[1,1,8,8]`, than the `[1,4,11,11]` policy)
and classifies held-out digits **end to end through the sandbox**, via the same
generic guest module. The model (~97% test accuracy) and its provenance/license
are documented in `crates/kelvane-runtime/tests/models/generate_digits.py`.

**Current generality boundary:** the input shape is fixed at load time and the
batch dimension is 1; the example guest reads a flat feature vector and argmaxes
up to 64 output scores. Models within those limits (fixed shape, batch-1, ≤64
outputs) run unchanged; anything needing dynamic shapes or batching does not yet.

## API stability

Kelvane is **pre-1.0** (`0.x`). "Stable" below means we will not change it
without bumping the minor version (`0.x → 0.(x+1)`) and noting it in the
changelog; "experimental" means it may change more freely. When in doubt we mark
things experimental — under-promising is deliberate.

**`kelvane-runtime`**

- *Stable:* `ModuleRuntime` and its methods (`new`, `load_model`, `load_module`,
  `invoke`, `hot_swap`, `call_count`, `model_backend`); `ExecutionLimits` and its
  fields; `inference::load_model`; `Model` and its methods (`run`, `input_len`,
  `backend`).
- *Not covered by semver:* the **string values** returned by `model_backend` /
  `Model::backend` (e.g. `"cpu(tract)"`) are informational — do not parse them.
  The `internals` feature is private (fuzz-only). The `cuda` backend-selection /
  fallback behavior may change.
- *Known limitation (not a stability promise):* `load_model` takes a fixed input
  shape; auto-detection may be added later as a **companion** API, without
  breaking the existing signature.

**`kelvane-sdk`** — the guest ABI

- *Stable:* the `export_module!` macro and its byte-in/byte-out handler contract;
  `pack`; the packed-`i64` return layout; the `module_alloc` / `process` /
  `module_dealloc` export signatures; the `kelvane::infer` import signature and
  its little-endian flat-`f32` tensor convention. Guests built against these keep
  working. The full contract is in the crate's rustdoc.
- *Experimental:* `alloc_bytes`, `dealloc_bytes`, and `run` are the low-level
  plumbing behind `export_module!` — prefer the macro.

The JSON shape of the **example** modules' output
(`{"module":...,"action":...}`) is an example convention, **not** part of the
ABI, and is not stable.

## Security

Kelvane's threat model — what the sandbox does and does **not** protect against,
each claim tied to a mechanism and a test — is documented in
[`SECURITY.md`](SECURITY.md), along with the trust boundary, the assumptions the
guarantees rest on, and how to report a vulnerability. In short: it defends the
**host** against an untrusted **guest** (memory cap, fuel/DoS bounds, zero
ambient authority, per-call isolation, host-owned model, a patched engine); it
does **not** protect against side channels, a malicious host, bugs in the engine
itself, or the safety of the model's own decisions.

## Supported platforms & toolchains

CI builds and runs the **full** test suite on every cell below. WASM modules are
built and the ONNX fixtures are committed **before** the test step, so nothing is
skipped — a green cell means the sandbox was actually exercised on that platform
(not merely compiled). All cells build with `--locked`.

| OS (runner) | Rust stable | Rust 1.88.0 (MSRV) |
|---|---|---|
| Linux — `ubuntu-latest`, x86-64 | ✅ | ✅ |
| macOS — `macos-latest`, arm64 | ✅ | ✅ |
| Windows — `windows-latest`, x86-64 | ✅ | ✅ |

- **MSRV: 1.94.0**, determined empirically — the dependency tree's floor is
  `wasmtime`/`wasmtime-wasi` 46 and `cranelift` 0.133 (edition 2024, rustc
  1.94.0). 1.93 and below fail to build; the MSRV CI row pins exactly 1.94.0.
  `rust-version` in `Cargo.toml` matches. (Raised from 1.88 by the wasmtime
  upgrade that resolved the sandbox-escape advisories.)
- **wasm target:** the example guest modules compile to `wasm32-wasip1`
  (`rustup target add wasm32-wasip1`). The guest code is gated to `wasm32`, so a
  host-side `cargo build`/`clippy --workspace` is link-clean on every OS.
- **CUDA backend** (`cuda` feature): a separate, **non-blocking, build-only**
  Linux job (the runners have no GPU) — it compiles and lint-checks the backend
  but does not run it. Building it needs `libssl-dev` + `pkg-config`; running it
  needs the CUDA runtime + cuDNN 9 on the library path.
- **Reproducibility:** `Cargo.lock` is committed and authoritative; CI builds
  `--locked` so an out-of-sync lock fails loudly. `ort` is pinned to
  `=2.0.0-rc.12`. Python deps in `kelvane-marl` are fully pinned.
- **Toolchain versions** behind the benchmark numbers are listed in the
  Benchmarks *Environment* line above and match what CI uses.

### Supply-chain status (honest)

- **Licenses** (`cargo deny`): the whole dependency tree is permissive and
  Apache-2.0-compatible (Apache-2.0/MIT/BSD/ISC/Zlib/Unicode-3.0/Unlicense/
  BlueOak). The only copyleft entries (`ittapi` GPL-2.0, `r-efi` LGPL-2.1) are
  the copyleft arm of dual/multi-licensed crates; the permissive arm is selected,
  so **no copyleft obligation** and **no GPL/AGPL surprise**.
- **Advisories** (`cargo audit`, a **blocking** CI job): **no known
  vulnerabilities.** The earlier 19 RustSec advisories against `wasmtime` /
  `wasmtime-wasi` 29 — including two critical sandbox escapes
  (RUSTSEC-2026-0095/0096) and a HIGH WASI `path_open` bypass (RUSTSEC-2026-0149)
  — were **resolved by upgrading to `wasmtime` 46.0.1**. The only remaining
  finding is one *unmaintained* advisory for `paste` (a transitive proc-macro
  dependency), which is not a vulnerability.

## License

Kelvane is open source under the Apache License 2.0. You may use, modify, and
distribute it, including commercially, provided you retain the copyright notice
and NOTICE and state significant changes, per the license. Copyright 2026
Muhammad Rakibul Islam. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Roadmap

- Additional inference backends behind Cargo features.
- A richer example-module gallery.
- Optional WIT / component-model interface for the guest ABI.
- More reproducible RL reference scenarios.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). All Rust checks (`fmt`, `clippy -D
warnings`, `build`, `test`) and the Python `pytest` suite must pass.

## Contact

Questions about Kelvane? rakib.islam@rutgers.edu

Author: Muhammad Rakibul Islam — https://github.com/rakib-nyc
