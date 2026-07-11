# Copyright 2026 Muhammad Rakibul Islam
# Licensed under the Apache License, Version 2.0 (the "License").
"""Train a tiny CNN image classifier and export it to ONNX for the generality test.

This is a REAL trained classifier (not a random fixture): it demonstrates that
Kelvane runs a model outside its own gridworld toy — a different domain
(handwritten-digit image classification) and a different input shape
([1, 1, 8, 8]) — end to end through the sandbox.

Dataset provenance / license
----------------------------
`sklearn.datasets.load_digits()` = the UCI ML "Optical Recognition of Handwritten
Digits" dataset (Alpaydin & Kaynak), 1797 samples of 8x8 grayscale digits, 0-9.
Bundled with scikit-learn (BSD-3-Clause); the underlying UCI dataset is CC BY 4.0.
No network access is required to load it, which keeps CI offline.

Regenerate:
    python -m pip install "torch" "scikit-learn" "onnx" \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple
    python generate_digits.py
Outputs (committed): digits_cnn.onnx  and  digits_samples.json
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))


class DigitsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 8x8 -> 4x4
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 4x4 -> 2x2
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 10),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    d = load_digits()
    x = (d.images / 16.0).astype(np.float32)          # normalize 0..16 -> 0..1
    x = x.reshape(-1, 1, 8, 8)
    y = d.target.astype(np.int64)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=SEED)

    model = DigitsCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    xtr_t, ytr_t = torch.from_numpy(xtr), torch.from_numpy(ytr)
    model.train()
    batch = 64
    n = xtr_t.shape[0]
    for epoch in range(120):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(xte)).argmax(1).numpy()
    acc = float((pred == yte).mean())
    print(f"test accuracy: {acc:.3f} on {len(yte)} held-out digits")

    onnx_path = os.path.join(HERE, "digits_cnn.onnx")
    torch.onnx.export(
        model,
        torch.zeros(1, 1, 8, 8),
        onnx_path,
        input_names=["input"],
        output_names=["scores"],
        opset_version=13,
        dynamic_axes=None,  # fixed batch-1, fixed [1,1,8,8]
        dynamo=False,  # legacy TorchScript exporter — no onnxscript, tract-loadable
    )
    print(f"wrote {onnx_path}")

    # A handful of held-out samples with known labels, flattened to 64 f32.
    samples = []
    for i in range(6):
        samples.append({"data": xte[i].reshape(-1).tolist(), "label": int(yte[i])})
    samples_path = os.path.join(HERE, "digits_samples.json")
    with open(samples_path, "w") as f:
        json.dump({"note": "sklearn digits held-out test samples, normalized /16",
                   "input_shape": [1, 1, 8, 8], "samples": samples}, f)
    print(f"wrote {samples_path} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
