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

"""Vectorized cooperative gridworld — batched torch tensors, GPU-capable.

Runs K environment copies simultaneously as batched tensors on the selected
device (GPU when available, else CPU). Observation construction (including the
objective beacon), dynamics, and reward are fully vectorized. The observation
encoding matches ``env_grid.py`` so a policy trained here transfers to the
single-env evaluator, the ONNX export, and the Kelvane runtime.

Shapes (K envs, N agents, G goals, R resources):
  reset()  -> obs (K, N, 4, 11, 11)
  step(a)  -> obs, reward (K, N), done (K,)   [auto-resets finished envs]
  global_state() -> (K, 4, 100, 100)
"""

import torch

WINDOW = 11
HALF = 5
GRID = 100
N_CH = 4
CH_AGENT, CH_RESOURCE, CH_OBSTACLE, CH_GOAL = 0, 1, 2, 3

STEP_COST = -0.01
GOAL_BONUS = 10.0
RESOURCE_BONUS = 2.0
COMPLETION_BONUS = 20.0
REACH_RADIUS = 2.0
COLLECT_RADIUS = 1.5

# action -> (dx, dy); 0 stay,1 up,2 down,3 left,4 right,5 collect,6 wait
_DELTA = torch.tensor(
    [[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0], [0, 0], [0, 0]], dtype=torch.float32
)


class VecGridWorld:
    def __init__(self, num_envs=64, n_agents=8, n_goals=3, n_resources=5,
                 n_obstacles=40, max_steps=200, device="cpu", obs_noise=0.0, seed=0):
        self.K = num_envs
        self.N = n_agents
        self.G = n_goals
        self.R = n_resources
        self.O = n_obstacles
        self.max_steps = max_steps
        self.device = torch.device(device)
        self.obs_noise = float(obs_noise)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.delta = _DELTA.to(self.device)
        self._init = False
        self.reset()

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.K, dtype=torch.bool, device=self.device)
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return self._obs()
        m = idx.numel()
        total = self.N + self.G + self.R + self.O
        scores = torch.rand(m, GRID * GRID, generator=self.gen, device=self.device)
        flat = scores.topk(total, dim=1).indices
        cells = torch.stack([flat // GRID, flat % GRID], dim=-1).float()

        if not self._init:
            self.agent_pos = torch.zeros(self.K, self.N, 2, device=self.device)
            self.goal_pos = torch.zeros(self.K, self.G, 2, device=self.device)
            self.resource_pos = torch.zeros(self.K, self.R, 2, device=self.device)
            self.obstacle_pos = torch.zeros(self.K, self.O, 2, device=self.device)
            self.reached = torch.zeros(self.K, self.G, dtype=torch.bool, device=self.device)
            self.collected = torch.zeros(self.K, self.R, dtype=torch.bool, device=self.device)
            self.steps = torch.zeros(self.K, dtype=torch.long, device=self.device)
            self._init = True

        a, b, c = self.N, self.N + self.G, self.N + self.G + self.R
        self.agent_pos[idx] = cells[:, :a]
        self.goal_pos[idx] = cells[:, a:b]
        self.resource_pos[idx] = cells[:, b:c]
        self.obstacle_pos[idx] = cells[:, c:]
        self.reached[idx] = False
        self.collected[idx] = False
        self.steps[idx] = 0
        return self._obs()

    def _scatter(self, obs, ch, entities, mask=None):
        rel = entities[:, None, :, :] - self.agent_pos[:, :, None, :]
        di = rel[..., 0] + HALF
        dj = rel[..., 1] + HALF
        inwin = (di >= 0) & (di < WINDOW) & (dj >= 0) & (dj < WINDOW)
        if mask is not None:
            inwin = inwin & mask[:, None, :]
        flat = (di.clamp(0, WINDOW - 1) * WINDOW + dj.clamp(0, WINDOW - 1)).long()
        obs[:, :, ch, :].scatter_add_(2, flat, inwin.float())

    def _obs(self):
        K, N = self.K, self.N
        obs = torch.zeros(K, N, N_CH, WINDOW * WINDOW, device=self.device)
        self._scatter(obs, CH_AGENT, self.agent_pos)
        self._scatter(obs, CH_RESOURCE, self.resource_pos, mask=~self.collected)
        self._scatter(obs, CH_OBSTACLE, self.obstacle_pos)
        self._scatter(obs, CH_GOAL, self.goal_pos, mask=~self.reached)
        obs[:, :, CH_AGENT, HALF * WINDOW + HALF] += 1.0

        nt, has = self._nearest_open_goal()
        vec = nt - self.agent_pos
        norm = vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        unit = vec / norm
        for r in range(2, HALF + 1):
            bi = (HALF + unit[..., 0] * r).round().clamp(0, WINDOW - 1)
            bj = (HALF + unit[..., 1] * r).round().clamp(0, WINDOW - 1)
            idx = (bi * WINDOW + bj).long().unsqueeze(-1)
            obs[:, :, CH_GOAL, :].scatter_add_(2, idx, has.float().unsqueeze(-1))
        obs = obs.clamp(0.0, 1.0).view(K, N, N_CH, WINDOW, WINDOW)
        if self.obs_noise > 0.0:
            obs = (obs + torch.randn(obs.shape, generator=self.gen, device=self.device)
                   * self.obs_noise).clamp(0.0, 1.0)
        return obs

    def _nearest_open_goal(self):
        d = torch.cdist(self.agent_pos, self.goal_pos)
        d = d.masked_fill(self.reached[:, None, :], float("inf"))
        has = torch.isfinite(d).any(dim=-1)
        d = d.masked_fill(~torch.isfinite(d), 1e9)
        nidx = d.argmin(dim=-1)
        nt = torch.gather(self.goal_pos, 1, nidx.unsqueeze(-1).expand(-1, -1, 2))
        return nt, has

    def global_state(self):
        s = torch.zeros(self.K, N_CH, GRID, GRID, device=self.device)

        def mark(ch, pos, keep=None):
            p = pos.long().clamp(0, GRID - 1)
            k = torch.arange(self.K, device=self.device)[:, None].expand(-1, p.shape[1])
            if keep is not None:
                kf, p0, p1 = k[keep], p[..., 0][keep], p[..., 1][keep]
            else:
                kf, p0, p1 = k.reshape(-1), p[..., 0].reshape(-1), p[..., 1].reshape(-1)
            s[kf, ch, p0, p1] = 1.0

        mark(CH_AGENT, self.agent_pos)
        mark(CH_RESOURCE, self.resource_pos, keep=~self.collected)
        mark(CH_OBSTACLE, self.obstacle_pos)
        mark(CH_GOAL, self.goal_pos, keep=~self.reached)
        return s

    def step(self, actions):
        self.steps += 1
        move = self.delta[actions]
        nxt = (self.agent_pos + move).clamp(0, GRID - 1)
        blocked = (nxt[:, :, None, :] == self.obstacle_pos[:, None, :, :]).all(-1).any(-1)
        self.agent_pos = torch.where(blocked[..., None], self.agent_pos, nxt)

        reward = self._reward()

        # resource gathering
        dr = torch.cdist(self.agent_pos, self.resource_pos)
        rhit = (dr <= COLLECT_RADIUS).any(dim=1) & (~self.collected)
        if rhit.any():
            for ri in range(self.R):
                sel = rhit[:, ri]
                if sel.any():
                    kk = sel.nonzero(as_tuple=True)[0]
                    ai = dr[:, :, ri].min(dim=1).indices[kk]
                    reward[kk, ai] += RESOURCE_BONUS
            self.collected = self.collected | rhit

        # goal servicing
        dg = torch.cdist(self.agent_pos, self.goal_pos)
        ghit = (dg <= REACH_RADIUS).any(dim=1) & (~self.reached)
        if ghit.any():
            for gi in range(self.G):
                sel = ghit[:, gi]
                if sel.any():
                    kk = sel.nonzero(as_tuple=True)[0]
                    ai = dg[:, :, gi].min(dim=1).indices[kk]
                    reward[kk, ai] += GOAL_BONUS
            self.reached = self.reached | ghit

        all_done = self.reached.all(dim=1)
        reward += (all_done.float() * COMPLETION_BONUS)[:, None]
        done = all_done | (self.steps >= self.max_steps)
        obs = self._obs()
        if done.any():
            self.reset(mask=done)
            obs = self._obs()
        return obs, reward, done

    def _reward(self):
        r = torch.full((self.K, self.N), STEP_COST, device=self.device)
        d = torch.cdist(self.agent_pos, self.goal_pos)
        d = d.masked_fill(self.reached[:, None, :], float("inf"))
        nd = d.min(dim=-1).values
        nd = torch.where(torch.isfinite(nd), nd, torch.zeros_like(nd))
        r = r - nd / GRID
        return r


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = VecGridWorld(num_envs=8, device=dev, obs_noise=0.02)
    obs = env.reset()
    print("device", dev, "obs", tuple(obs.shape), "state", tuple(env.global_state().shape))
    total = torch.zeros(env.K, env.N, device=env.device)
    for _ in range(50):
        acts = torch.randint(0, 7, (env.K, env.N), device=env.device)
        obs, rew, done = env.step(acts)
        total += rew
    print("mean reward/agent over 50 steps:", round(total.mean().item(), 3))
    print("vectorized gridworld OK")
