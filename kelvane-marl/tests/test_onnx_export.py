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

"""Test: ONNX export produces a valid FP32 policy that runs under onnxruntime."""
import os

import numpy as np

from kelvane_marl import export_onnx
from kelvane_marl.paths import MODELS_DIR


def test_onnx_export_runs():
    export_onnx.main()
    fp32 = os.path.join(MODELS_DIR, "grid_policy.onnx")
    assert os.path.exists(fp32)

    import onnxruntime as ort
    sess = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
    out = sess.run(["scores"], {"observation": np.zeros((1, 4, 11, 11), dtype=np.float32)})[0]
    assert out.shape == (1, 7)
