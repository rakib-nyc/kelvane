# Copyright 2026 Muhammad Rakibul Islam
# Licensed under the Apache License, Version 2.0 (the "License").
"""Regenerate the small ONNX fixtures used by the Rust runtime test suite.

These are tiny, deterministic, synthetic models (Flatten + Gemm) — NOT trained
policies. They exist so the adversarial / variable-shape tests can load and run
real models of several input shapes through the sandbox without depending on the
Python training pipeline. Committed to the repo so `cargo test` is hermetic.

Regenerate:  python -m pip install "onnx>=1.16" && python generate.py
Only dependency is `onnx` (no torch).
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_flat_gemm(path: str, in_shape: list[int], out_dim: int, seed: int) -> None:
    """A model that flattens `in_shape` and applies one Gemm -> [batch, out_dim]."""
    rng = np.random.default_rng(seed)
    in_flat = int(np.prod(in_shape[1:]))  # everything but the batch dim
    w = (rng.standard_normal((in_flat, out_dim)) * 0.1).astype(np.float32)
    b = (rng.standard_normal((out_dim,)) * 0.1).astype(np.float32)

    obs = helper.make_tensor_value_info("observation", TensorProto.FLOAT, in_shape)
    scores = helper.make_tensor_value_info(
        "scores", TensorProto.FLOAT, [in_shape[0], out_dim]
    )
    nodes = [
        helper.make_node("Flatten", ["observation"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "W", "B"], ["scores"], transB=0),
    ]
    graph = helper.make_graph(
        nodes,
        "kelvane_test_model",
        [obs],
        [scores],
        initializer=[
            numpy_helper.from_array(w, name="W"),
            numpy_helper.from_array(b, name="B"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9  # broadly compatible with tract 0.21 / onnxruntime
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote {path}  ({in_shape} -> [{in_shape[0]}, {out_dim}])")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    # Same shape as the real grid policy (4x11x11 -> 7 actions), so the existing
    # runtime tests can use a synthetic stand-in.
    make_flat_gemm(os.path.join(here, "policy_4x11x11.onnx"), [1, 4, 11, 11], 7, seed=1)
    # Two deliberately different shapes for the variable-shape tests.
    make_flat_gemm(os.path.join(here, "mlp_16.onnx"), [1, 16], 3, seed=2)
    make_flat_gemm(os.path.join(here, "img_3x8x8.onnx"), [1, 3, 8, 8], 5, seed=3)


if __name__ == "__main__":
    main()
