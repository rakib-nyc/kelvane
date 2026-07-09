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

"""Vectorized MAPPO trainer (GPU-capable).

Centralized Training, Decentralized Execution: each agent acts from its local
window via a shared ``Actor``; a ``CentralizedCritic`` sees the global state for
a low-variance baseline. Real PPO — per-agent GAE(lambda) advantages against the
shared critic, a clipped surrogate objective, an entropy bonus, and minibatch
epochs. Runs K parallel environments on the GPU (or CPU) so the neural compute
is batched.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from kelvane_marl.env_vec import VecGridWorld
from kelvane_marl.paths import LOGS_DIR, ensure_dirs

N_CH = 4
N_AGENTS = 8
LR = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.005
VALUE_COEF = 0.5
EPOCHS = 4
MAX_GRAD_NORM = 0.5


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(16 * 11 * 11, 64), nn.ReLU(), nn.Linear(64, 7))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.fc(self.conv(x))


class CentralizedCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(4, 16, 5, stride=2), nn.ReLU(), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(16 * 48 * 48, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.fc(self.conv(x))


def train(updates, num_envs=64, rollout=32, minibatch=4096, curriculum=True, seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = VecGridWorld(num_envs=num_envs, n_agents=N_AGENTS, device=str(device), seed=seed)
    K, N, T = num_envs, N_AGENTS, rollout

    actor = Actor().to(device)
    critic = CentralizedCritic().to(device)
    a_opt = torch.optim.Adam(actor.parameters(), lr=LR)
    c_opt = torch.optim.Adam(critic.parameters(), lr=LR)

    print(f"MAPPO on {device}: {K} envs x {N} agents, rollout {T} (batch {K*N*T}) for {updates} updates")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    curve, ema = [], None
    obs = env.reset()
    t0 = time.time()

    for u in range(updates):
        if curriculum:
            env.obs_noise = float(np.random.uniform(0.0, 0.04))

        obs_b = torch.empty(T, K * N, N_CH, 11, 11, device=device)
        act_b = torch.empty(T, K * N, dtype=torch.long, device=device)
        logp_b = torch.empty(T, K * N, device=device)
        rew_b = torch.empty(T, K, N, device=device)
        val_b = torch.empty(T, K, device=device)
        done_b = torch.empty(T, K, device=device)
        state_b = torch.empty(T, K, N_CH, 100, 100, device=device)

        with torch.no_grad():
            for t in range(T):
                flat = obs.view(K * N, N_CH, 11, 11)
                dist = Categorical(logits=actor(flat))
                acts = dist.sample()
                state = env.global_state()
                val = critic(state).squeeze(-1)
                obs_b[t], act_b[t], logp_b[t] = flat, acts, dist.log_prob(acts)
                val_b[t], state_b[t] = val, state
                obs, rew, done = env.step(acts.view(K, N))
                rew_b[t], done_b[t] = rew, done.float()
            last_v = critic(env.global_state()).squeeze(-1)

        adv = torch.empty(T, K, N, device=device)
        gae = torch.zeros(K, N, device=device)
        for t in reversed(range(T)):
            next_v = last_v if t == T - 1 else val_b[t + 1]
            nonterm = (1.0 - done_b[t])[:, None]
            delta = rew_b[t] + GAMMA * next_v[:, None] * nonterm - val_b[t][:, None]
            gae = delta + GAMMA * GAE_LAMBDA * nonterm * gae
            adv[t] = gae
        returns = adv + val_b[:, :, None]

        obs_f = obs_b.reshape(T * K * N, N_CH, 11, 11)
        act_f = act_b.reshape(-1)
        logp_f = logp_b.reshape(-1)
        adv_f = adv.reshape(-1)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        states_f = state_b.reshape(T * K, N_CH, 100, 100)
        ret_f = returns.mean(dim=2).reshape(-1)

        n = obs_f.shape[0]
        idx = torch.randperm(n, device=device)
        ent_acc, nb = 0.0, 0
        for _ in range(EPOCHS):
            for s in range(0, n, minibatch):
                mb = idx[s:s + minibatch]
                dist = Categorical(logits=actor(obs_f[mb]))
                nlp = dist.log_prob(act_f[mb])
                ent = dist.entropy().mean()
                ratio = torch.exp(nlp - logp_f[mb])
                a = adv_f[mb]
                s1, s2 = ratio * a, torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * a
                pl = -torch.min(s1, s2).mean() - ENTROPY_COEF * ent
                a_opt.zero_grad(); pl.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM); a_opt.step()
                ent_acc += ent.item(); nb += 1
            v = critic(states_f).squeeze(-1)
            vl = VALUE_COEF * F.mse_loss(v, ret_f)
            c_opt.zero_grad(); vl.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM); c_opt.step()

        ret = float(rew_b.sum(dim=0).mean())
        ema = ret if ema is None else 0.9 * ema + 0.1 * ret
        curve.append({"update": u, "return": round(ret, 3), "ema": round(ema, 3),
                      "entropy": round(ent_acc / max(1, nb), 3)})
        if u % 20 == 0 or u == updates - 1:
            ups = (u + 1) / (time.time() - t0)
            print(f"  update {u:03d}  return={ret:8.2f}  ema={ema:8.2f}  "
                  f"entropy={ent_acc/max(1,nb):.3f}  [{ups:.1f} upd/s]")

    ensure_dirs()
    torch.save(actor.state_dict(), os.path.join(LOGS_DIR, "actor.pt"))
    with open(os.path.join(LOGS_DIR, "curve.json"), "w") as f:
        json.dump(curve, f)
    print(f"Saved actor to {os.path.join(LOGS_DIR, 'actor.pt')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--no-curriculum", action="store_true")
    a = p.parse_args()
    train(a.updates, num_envs=a.envs, rollout=a.rollout, minibatch=a.minibatch,
          curriculum=not a.no_curriculum)
