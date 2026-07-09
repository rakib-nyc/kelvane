# Copyright 2026 Muhammad Rakibul Islam
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export the trained policy to ONNX (FP32 + INT8), verified against PyTorch."""

import os

import numpy as np
import torch

from kelvane_marl.mappo import Actor
from kelvane_marl.paths import LOGS_DIR, MODELS_DIR, ensure_dirs

FP32 = os.path.join(MODELS_DIR, "grid_policy.onnx")
INT8 = os.path.join(MODELS_DIR, "grid_policy.int8.onnx")
ACTOR = os.path.join(LOGS_DIR, "actor.pt")
OPSET = 17


def export_fp32(actor):
    torch.onnx.export(
        actor,
        torch.randn(1, 4, 11, 11),
        FP32,
        input_names=["observation"],
        output_names=["scores"],
        dynamic_axes={"observation": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )
    print(f"[fp32] exported {FP32} (opset {OPSET})")


def quantize_int8():
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        print(f"[int8] onnxruntime.quantization unavailable ({e}); skipping")
        return False
    quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8)
    print(f"[int8] wrote {INT8}")
    return True


def verify(actor, path, tol):
    try:
        import onnxruntime as ort
    except ImportError as e:
        print(f"[verify] onnxruntime unavailable ({e}); skipping {path}")
        return
    x = torch.randn(3, 4, 11, 11)
    with torch.no_grad():
        ref = actor(x).numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(["scores"], {"observation": x.numpy().astype(np.float32)})[0]
    max_diff = float(np.max(np.abs(ref - out)))
    argmax_match = bool(np.all(ref.argmax(1) == out.argmax(1)))
    status = "OK" if max_diff <= tol else "DIVERGENT"
    print(f"[verify] {path}: max|delta|={max_diff:.5f} (tol {tol}) argmax_match={argmax_match} -> {status}")


def main():
    ensure_dirs()
    actor = Actor()
    if os.path.exists(ACTOR):
        actor.load_state_dict(torch.load(ACTOR, map_location="cpu"))
        print(f"loaded weights from {ACTOR}")
    else:
        print(f"WARNING: {ACTOR} not found; exporting a randomly-initialized policy")
    actor.eval()

    export_fp32(actor)
    verify(actor, FP32, tol=1e-4)
    if quantize_int8():
        verify(actor, INT8, tol=5e-1)

    for p in (FP32, INT8):
        if os.path.exists(p):
            print(f"[size] {p}: {os.path.getsize(p) / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
