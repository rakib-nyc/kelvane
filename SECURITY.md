# Kelvane Security & Threat Model

This document states, as precisely and honestly as possible, what Kelvane's
sandbox **does** and **does not** protect against. Every "protects against" claim
is tied to a specific mechanism **and** a test in the suite. Where a protection
is real but not covered by a dedicated test, it is labeled as such — an
overstated guarantee is worse than none.

Kelvane's job: run an **untrusted** WebAssembly guest and let it request a
neural-network inference from a **host-owned** model, without letting the guest
touch the host, the weights, the accelerator, or anything outside its own capped
linear memory and fuel budget.

## Trust boundary

**Trusted** (assumed correct and non-malicious):

- the host process and the code embedding `ModuleRuntime`;
- the ONNX model the host loads;
- the WebAssembly engine (wasmtime + Cranelift) and the inference libraries
  (`tract`, and `ort`/ONNX Runtime under the `cuda` feature).

**Untrusted** (treated as adversarial):

- the guest `.wasm` module;
- the input bytes passed to `invoke`;
- the guest's `process` return value and its `kelvane::infer` arguments.

The guest reaches the host **only** through (a) its own exports
`process` / `module_alloc` / `module_dealloc`, (b) the `kelvane::infer` import,
and (c) WASI preview1 imports that are wired to a context granting nothing.
Everything crossing that boundary is size- and bounds-checked by the host.

## What Kelvane protects against

Each row: the threat, the mechanism, and the test(s) that demonstrate it. Tests
live in `crates/kelvane-runtime/tests/adversarial.rs` (integration) and
`crates/kelvane-runtime/src/host.rs` (`#[cfg(test)]` units).

| Threat | Mechanism | Test(s) |
|---|---|---|
| Guest exhausts host memory | per-store `StoreLimits::memory_size` cap (default 64 MiB) | `oversized_initial_memory_is_rejected`, `memory_grow_loop_stays_capped`, `memory_limit_blocks_oversized_module` |
| Guest burns unbounded CPU / hangs | wasmtime fuel metering (`consume_fuel` + `set_fuel` per call) | `infinite_loop_hits_fuel_limit_not_hang`, `fuel_boundary_brackets_success_and_trap`, `deep_recursion_traps_not_crash` |
| Guest reads/writes host memory via its ABI return | `decode_output_region`: 1 MiB output cap + linear-memory bounds check | `output_len_over_cap_is_error`, `output_region_out_of_bounds_is_error`, `garbage_packed_return_is_error`, `max_pointer_stays_in_bounds_check`, `decode_output_*` units |
| Guest OOMs the host via a huge `infer` `in_len` | host read buffer bounded by the guest's own memory size | `infer_huge_in_len_is_rejected_without_giant_alloc` |
| Guest sends hostile `infer` arguments | bounds/shape/length checks return `-1`, never fault | `infer_input_too_short/long_returns_negative`, `infer_in_ptr/out_ptr_out_of_bounds_returns_negative`, `infer_out_cap_smaller_than_output_truncates`, `infer_with_no_model_returns_negative` |
| Malformed / trapping / missing-export module | instantiation + typed-function checks; a trap surfaces as `Err` | `missing_process_export_is_error`, `missing_memory_export_is_error`, `wrong_alloc_signature_is_error`, `alloc_returns_null_is_error`, `process_trap_is_error_not_crash` |
| State leaking between calls | a **fresh** `Store` + instance per `invoke` | `per_call_state_is_isolated` |
| Guest accessing the filesystem | WASI context grants **no** preopened directories | `no_ambient_filesystem_authority` |
| Malformed / wrong-shape model file | `tract` load/shape validation errors surface as `Err` | `non_onnx_file_fails_to_load`, `declared_shape_mismatch_fails` |
| Known engine sandbox-escape CVEs | pinned to **wasmtime 46** (the CVEs against 29 are fixed) | `cargo audit` CI gate (blocking; currently clean) |

**Protections that hold by construction, but have no dedicated adversarial
test** (stated honestly, not as verified guarantees):

- **The guest cannot read the model weights or touch the accelerator.** There is
  no host import that exposes them — the only imports are WASI (granting nothing)
  and `kelvane::infer`, which returns scores. This is *verified by construction*
  (no such mechanism exists to attack), not by a test that "tries and fails."
- **No network and no stdio ambient authority.** The WASI context grants none
  (deny-by-default `WasiCtxBuilder::new`). Only the **filesystem** denial is
  covered by a test; network and stdio denial rest on the same builder default
  but are not separately exercised.

## What Kelvane does NOT protect against

Be adversarial when reading this — these are real gaps, stated plainly:

- **Microarchitectural / side-channel attacks.** Spectre-class speculation, cache
  or port timing, power/EM. wasmtime's sandbox is a *memory-safety* boundary, not
  a side-channel boundary. Out of scope.
- **Timing.** A guest can measure how long an `infer` takes; Kelvane makes no
  constant-time guarantee.
- **A malicious host.** Kelvane protects the host *from* the guest, never the
  guest from the host. The host can read guest memory, feed any input, and see
  all output. Guest-supplied secrets are not protected from the host.
- **Bugs in wasmtime, tract, or ort themselves.** A 0-day in the engine or the
  inference libraries is not something Kelvane can prevent; it is only *mitigated*
  by keeping them patched (the `cargo audit` gate), and by the trust assumption
  below.
- **A malicious model file.** `load_model` runs `tract`/`ort` over the `.onnx`;
  the model is **trusted** in this threat model. A model crafted to exploit a
  parser/codegen bug in those libraries is out of scope. Only *malformed but not
  malicious* models are tested (they error out).
- **Correctness or safety of the model's decisions.** Kelvane runs the model; it
  does not judge whether the output action is sensible or safe.
- **Resource use below the caps.** A guest that stays under the memory and fuel
  caps but does useless work still consumes the budget you granted it. Caps bound
  the damage; they do not make computation free.
- **Content of the allowed channels.** The guest's output bytes and its `infer`
  requests are legitimate channels; Kelvane checks their **size and bounds**, not
  their **content**. A guest can encode arbitrary data in its output.
- **Host-level resource limits beyond the per-call store** — file descriptors,
  threads, or many runtimes in one process. Kelvane caps a single call's memory
  and CPU, not the whole host.
- **Physical attacks** — rowhammer, cold-boot, hardware fault injection.

## Assumptions that must hold for the guarantees

1. **A patched engine.** wasmtime/tract/ort are at non-CVE versions. The
   `cargo audit` CI gate enforces this at build time; a fork that pins a
   vulnerable engine loses the sandbox guarantees.
2. **Sane limits.** The host uses reasonable `ExecutionLimits`. The defaults
   (64 MiB, ~200M fuel, 32 modules) are sane; a host that sets `fuel_per_call`
   or `max_memory_bytes` absurdly high weakens the DoS bounds accordingly.
3. **No extra capabilities granted.** The runtime grants the guest no WASI
   authority. A modified host that adds preopened directories, network, or stdio
   breaks the zero-ambient-authority property.
4. **A trusted model.** The `.onnx` passed to `load_model` is host-chosen and
   trusted; it is not adversarial input.
5. **A sound sandbox on the target.** The wasmtime sandbox is assumed correct for
   the build target/architecture in use.

## Reporting a vulnerability

Please report suspected vulnerabilities **privately**, not via a public issue:
open a GitHub private security advisory on `rakib-nyc/kelvane`, or email the
maintainer (see the README contact). Include a minimal reproducer if possible.
There is no formal SLA — this is a research project — but security reports are
prioritized over other work.
