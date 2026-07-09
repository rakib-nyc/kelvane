# kelvane-marl

A compact, reproducible **multi-agent reinforcement learning** reference.

- A **cooperative gridworld** (`env_grid.py`) with neutral entities only:
  agents, resources, goals, and obstacles.
- A **GPU-capable vectorized environment** (`env_vec.py`) that runs many copies
  as batched tensors on the GPU (or CPU) so neural compute is batched.
- A **MAPPO trainer** (`mappo.py`) — centralized critic, decentralized actors,
  per-agent GAE, clipped PPO objective.
- A **QMIX trainer** (`qmix.py`) — recurrent per-agent Q-networks with a
  monotonic mixing network.
- **ONNX export** (`export_onnx.py`) — FP32 and INT8, verified against PyTorch.
- **Evaluation** (`evaluate.py`) — trained policy vs a random baseline, plus a
  team-size scale sweep (the same policy on larger teams, no retraining).

## Quickstart

```bash
pip install numpy torch gymnasium pettingzoo onnx onnxruntime

# train (GPU used automatically if available), export, evaluate
python -m kelvane_marl.mappo --updates 200 --envs 64
python -m kelvane_marl.export_onnx
python -m kelvane_marl.evaluate --episodes 20

# environment / QMIX smoke checks
python -m kelvane_marl.env_grid
python -m kelvane_marl.qmix --episodes 20
```

The exported policy (`models/grid_policy.onnx`) is the artifact the
`kelvane-runtime` demo loads and runs inside the sandbox.

## Observation encoding

Each agent sees an `11x11x4` egocentric window (channels: agents, resources,
obstacles, goals). The goal channel also carries an **objective beacon** — the
bearing to the nearest un-reached goal, projected onto the window border — so the
policy has a directional signal even when the goal is beyond the window. The
encoding is identical in the single-env, vectorized-env, and exported paths.

## License

Apache License 2.0. See the repository `LICENSE` and `NOTICE`.
