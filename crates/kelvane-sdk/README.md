# kelvane-sdk

The guest SDK and host/guest ABI for **Kelvane WebAssembly modules**.

A module supplies a byte-in / byte-out function and invokes `export_module!`;
the SDK generates the ABI exports and hides the linear-memory plumbing.

## ABI

- `module_alloc(len) -> ptr` — host asks the guest to allocate an input buffer.
- `process(in_ptr, in_len) -> i64` — returns the output **length-delimited**:
  high 32 bits = output pointer, low 32 bits = output length (`(ptr << 32) | len`).
- `module_dealloc(ptr, len)` — free a buffer.

The packed pointer+length return avoids fixed offsets and NUL-termination
guessing; the host reads exactly `len` bytes.

## Example

```rust
use kelvane_sdk::export_module;

fn handle(input: &[u8]) -> Vec<u8> {
    // interpret `input`, produce output bytes
    input.to_vec()
}

export_module!(handle);
```

## License

Apache License 2.0. See the repository `LICENSE` and `NOTICE`.
