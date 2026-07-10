# Contributing to Kelvane

Thanks for your interest. Kelvane is general-purpose AI infrastructure and
research tooling; contributions that keep it small, honest, and reproducible are
very welcome.

## Ground rules

- **Scope.** Kelvane is intentionally general-purpose. Please keep it that way —
  no networking and no security-primitive (key-exchange / signing) code belongs
  in this project.
- **Honesty.** Benchmarks must be measured on the contributor's own hardware and
  labeled as such. Do not add fabricated numbers.
- **License.** By contributing you agree your contribution is licensed under the
  project's Apache License 2.0 (see `LICENSE`).

## Developer workflow

```bash
# Rust
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release
cargo build --release --target wasm32-wasip1 -p policy-module -p scripted-module
cargo test --workspace

# Python (kelvane-marl)
cd kelvane-marl
pytest -q
```

The runtime's artifact-dependent tests **fail** (never skip) if their inputs are
missing, so a green run always means the sandbox was actually exercised. Build
the wasm example modules first (as above); the small ONNX models they load are
committed fixtures under `crates/kelvane-runtime/tests/models/` (regenerate with
that directory's `generate.py`, which needs only `onnx`). Adversarial and
fuzz-decode coverage lives in `crates/kelvane-runtime/tests/adversarial.rs` and
`fuzz/` (the latter needs nightly + `cargo-fuzz`; it is its own workspace and is
not part of the default build).

All checks must pass before a pull request is merged. GPU-only code paths are
gated behind the `cuda` Cargo feature and a runtime hardware check so the default
suite runs anywhere.

## Style

- Rust: `rustfmt` defaults, `clippy -D warnings` clean.
- Python: keep modules small and importable; prefer clear names over cleverness.
- Commits: short, imperative subject lines.
