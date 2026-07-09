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

"""Smoke test: a short vectorized MAPPO run saves a policy checkpoint."""
import os

from kelvane_marl import mappo
from kelvane_marl.paths import LOGS_DIR


def test_mappo_smoke():
    mappo.train(updates=3, num_envs=8, rollout=8, minibatch=512, curriculum=False)
    assert os.path.exists(os.path.join(LOGS_DIR, "actor.pt"))


def test_gridworld_runs():
    from kelvane_marl.env_grid import GridWorld
    env = GridWorld(seed=0)
    obs, _ = env.reset(seed=0)
    a0 = env.possible_agents[0]
    assert obs[a0].shape == (11, 11, 4)
    acts = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, term, trunc, _ = env.step(acts)
    assert len(rewards) == env.n_agents
