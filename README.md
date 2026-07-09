# Kelvane

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

Measured on this contributor's hardware (AMD Ryzen 7 CPU; NVIDIA RTX 2080 Super).
Inference of the exported `[1,4,11,11]` policy, 2000 iterations after warm-up.

| Backend | Median | Throughput |
|---|---|---|
| CPU (`tract`) | **16.6 µs** | ~58,000 inferences/s |

A **CUDA backend** is available behind the `cuda` Cargo feature (ONNX Runtime's
CUDA execution provider) with automatic fallback to CPU. It was **not built in
this environment** — the ONNX Runtime toolchain was unavailable here (the
binary-download path requires the OpenSSL development libraries) — so **no CUDA
numbers are reported**; they would be fabricated otherwise. On a machine with the
ONNX Runtime CUDA toolchain, build and measure with:

```bash
cargo run --release --features cuda --example bench -p kelvane-runtime
```

Numbers are illustrative of *this* small model on *this* machine; run
`cargo run --release --example bench -p kelvane-runtime` to measure your own.
Do not treat these as general performance claims.

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
