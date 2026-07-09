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

"""Cooperative gridworld (PettingZoo ParallelEnv).

A scaled cooperative task with neutral entities only: a team of **agents** must
reach a set of **goals** while gathering **resources** and avoiding
**obstacles**, using only a local egocentric observation window. A global state
is exposed for a centralized critic (CTDE).

Observation channels (11x11 egocentric window):
  0 = agents, 1 = resources, 2 = obstacles, 3 = goals (+ objective beacon)

The objective beacon projects the bearing to the nearest un-reached goal onto
the goal channel's window border, so the policy always has a "go this way"
signal even when the goal is beyond the sensing window.
"""

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

CH_AGENT, CH_RESOURCE, CH_OBSTACLE, CH_GOAL = 0, 1, 2, 3
N_CHANNELS = 4
WINDOW = 11
HALF = WINDOW // 2

STAY, UP, DOWN, LEFT, RIGHT, COLLECT, WAIT = range(7)
N_ACTIONS = 7

STEP_COST = -0.01
GOAL_BONUS = 10.0
RESOURCE_BONUS = 2.0
COMPLETION_BONUS = 20.0
REACH_RADIUS = 2.0
COLLECT_RADIUS = 1.5


class GridWorld(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "kelvane_gridworld_v1"}

    def __init__(self, n_agents=8, grid_size=100, n_goals=3, n_resources=5,
                 n_obstacles=40, max_steps=200, seed=None, obs_noise=0.0):
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_goals = n_goals
        self.n_resources = n_resources
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps
        # obs_noise: stddev of Gaussian noise added to observations, modelling
        # imperfect sensors (defaults to 0 = perfect observations).
        self.obs_noise = float(obs_noise)

        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents[:]
        self._rng = np.random.default_rng(seed)

        self._obs_space = spaces.Box(low=0.0, high=1.0, shape=(WINDOW, WINDOW, N_CHANNELS), dtype=np.float32)
        self._act_space = spaces.Discrete(N_ACTIONS)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._act_space

    # -- lifecycle ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.steps = 0

        cells = self._sample_unique_cells(
            self.n_agents + self.n_goals + self.n_resources + self.n_obstacles
        )
        it = iter(cells)
        self.positions = {a: np.array(next(it), dtype=np.int32) for a in self.agents}
        self.goals = np.array([next(it) for _ in range(self.n_goals)], dtype=np.int32)
        self.resources = np.array([next(it) for _ in range(self.n_resources)], dtype=np.int32)
        self.obstacles = np.array([next(it) for _ in range(self.n_obstacles)], dtype=np.int32)
        self.reached = np.zeros(self.n_goals, dtype=bool)
        self.collected = np.zeros(self.n_resources, dtype=bool)

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        self.steps += 1
        rewards = {}
        for agent in self.agents:
            act = int(actions[agent])
            self._apply_move(agent, act)
            rewards[agent] = self._reward(agent, act)

        # Resource gathering.
        for ri in range(self.n_resources):
            if self.collected[ri]:
                continue
            for agent in self.agents:
                if np.linalg.norm(self.positions[agent] - self.resources[ri]) <= COLLECT_RADIUS:
                    self.collected[ri] = True
                    rewards[agent] += RESOURCE_BONUS
                    break

        # Goal servicing.
        for gi in range(self.n_goals):
            if self.reached[gi]:
                continue
            for agent in self.agents:
                if np.linalg.norm(self.positions[agent] - self.goals[gi]) <= REACH_RADIUS:
                    self.reached[gi] = True
                    rewards[agent] += GOAL_BONUS
                    break

        all_done = bool(self.reached.all())
        if all_done:
            for agent in self.agents:
                rewards[agent] += COMPLETION_BONUS

        truncate = self.steps >= self.max_steps
        terminated = {a: all_done for a in self.agents}
        truncated = {a: truncate for a in self.agents}
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {"reached": int(self.reached.sum())} for a in self.agents}
        if all_done or truncate:
            self.agents = []
        return observations, rewards, terminated, truncated, infos

    # -- state / observation ------------------------------------------------
    def state(self):
        grid = np.zeros((self.grid_size, self.grid_size, N_CHANNELS), dtype=np.float32)
        for a in self.possible_agents:
            if a in self.positions:
                x, y = self.positions[a]
                grid[x, y, CH_AGENT] = 1.0
        for ri, (x, y) in enumerate(self.resources):
            if not self.collected[ri]:
                grid[x, y, CH_RESOURCE] = 1.0
        for (x, y) in self.obstacles:
            grid[x, y, CH_OBSTACLE] = 1.0
        for gi, (x, y) in enumerate(self.goals):
            if not self.reached[gi]:
                grid[x, y, CH_GOAL] = 1.0
        return grid

    def _get_obs(self, agent):
        obs = np.zeros((WINDOW, WINDOW, N_CHANNELS), dtype=np.float32)
        if agent not in self.positions:
            return obs
        ax, ay = self.positions[agent]

        def plot(px, py, ch):
            dx, dy = px - ax + HALF, py - ay + HALF
            if 0 <= dx < WINDOW and 0 <= dy < WINDOW:
                obs[dx, dy, ch] = 1.0

        for other in self.possible_agents:
            if other in self.positions and other != agent:
                plot(self.positions[other][0], self.positions[other][1], CH_AGENT)
        for ri, (x, y) in enumerate(self.resources):
            if not self.collected[ri]:
                plot(x, y, CH_RESOURCE)
        for (x, y) in self.obstacles:
            plot(x, y, CH_OBSTACLE)
        for gi, (x, y) in enumerate(self.goals):
            if not self.reached[gi]:
                plot(x, y, CH_GOAL)
        obs[HALF, HALF, CH_AGENT] = 1.0

        # Objective beacon: ray toward the nearest un-reached goal.
        open_idx = np.where(~self.reached)[0]
        if len(open_idx) > 0:
            gs = self.goals[open_idx]
            d = np.linalg.norm(gs - self.positions[agent], axis=1)
            vec = gs[int(np.argmin(d))] - self.positions[agent]
            norm = np.linalg.norm(vec)
            if norm > 0:
                ux, uy = vec / norm
                for r in range(2, HALF + 1):
                    bx = min(max(int(round(HALF + ux * r)), 0), WINDOW - 1)
                    by = min(max(int(round(HALF + uy * r)), 0), WINDOW - 1)
                    obs[bx, by, CH_GOAL] = 1.0

        if self.obs_noise > 0.0:
            noise = self._rng.normal(0.0, self.obs_noise, obs.shape).astype(np.float32)
            obs = np.clip(obs + noise, 0.0, 1.0)
        return obs

    # -- dynamics -----------------------------------------------------------
    def _apply_move(self, agent, act):
        pos = self.positions[agent]
        nxt = pos.copy()
        if act == UP:
            nxt[1] = min(self.grid_size - 1, pos[1] + 1)
        elif act == DOWN:
            nxt[1] = max(0, pos[1] - 1)
        elif act == LEFT:
            nxt[0] = max(0, pos[0] - 1)
        elif act == RIGHT:
            nxt[0] = min(self.grid_size - 1, pos[0] + 1)
        # STAY / COLLECT / WAIT do not move.
        if not any(np.array_equal(nxt, o) for o in self.obstacles):
            self.positions[agent] = nxt

    def _reward(self, agent, act):
        r = STEP_COST
        pos = self.positions[agent]
        open_goals = self.goals[~self.reached]
        if len(open_goals) > 0:
            dist = np.min(np.linalg.norm(open_goals - pos, axis=1))
            r += -dist / self.grid_size
        return r

    def _sample_unique_cells(self, k):
        idx = self._rng.choice(self.grid_size * self.grid_size, size=k, replace=False)
        return [(int(i // self.grid_size), int(i % self.grid_size)) for i in idx]


if __name__ == "__main__":
    env = GridWorld(seed=0)
    obs, _ = env.reset(seed=0)
    a0 = env.possible_agents[0]
    print(f"agents={len(env.possible_agents)} obs={obs[a0].shape} state={env.state().shape}")
    total = {a: 0.0 for a in env.possible_agents}
    steps = 0
    while env.agents:
        acts = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, term, trunc, infos = env.step(acts)
        for a, rw in rewards.items():
            total[a] += rw
        steps += 1
    print(f"ran {steps} steps; goals reached={int(env.reached.sum())}/{env.n_goals}; "
          f"mean_return={np.mean(list(total.values())):.2f}")
    print("gridworld smoke OK")
