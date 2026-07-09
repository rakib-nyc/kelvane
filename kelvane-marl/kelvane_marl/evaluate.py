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

"""Evaluate the trained policy against a random baseline on the gridworld.

Reports goal-completion rate, reach rate, team return, and steps, plus a scale
sweep (the same policy deployed on larger teams with no retraining).
"""

import argparse
import json
import os

import numpy as np
import torch

from kelvane_marl.env_grid import GridWorld
from kelvane_marl.mappo import Actor
from kelvane_marl.paths import LOGS_DIR, ensure_dirs


def obs_batch(obs_dict, agents):
    arr = np.stack([obs_dict[a] for a in agents])
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.as_tensor(arr, dtype=torch.float32)


def greedy(actor, obs_dict, agents):
    with torch.no_grad():
        acts = torch.argmax(actor(obs_batch(obs_dict, agents)), dim=1)
    return {a: int(acts[i].item()) for i, a in enumerate(agents)}


def run_episode(env, policy_fn):
    obs, _ = env.reset()
    agents = env.possible_agents[:]
    total, steps = 0.0, 0
    while env.agents:
        acts = policy_fn(env, obs, agents)
        obs, rewards, _t, _tr, _ = env.step(acts)
        total += float(np.mean([rewards[a] for a in agents]))
        steps += 1
    reached = int(env.reached.sum())
    return {"return": total, "reach_rate": reached / env.n_goals,
            "success": 1.0 if bool(env.reached.all()) else 0.0, "steps": float(steps)}


def summarize(rs):
    return {k: round(float(np.mean([r[k] for r in rs])), 4)
            for k in ["return", "reach_rate", "success", "steps"]}


def evaluate(policy_fn, n_agents, episodes, seed_base, **kw):
    out = []
    for e in range(episodes):
        env = GridWorld(n_agents=n_agents, seed=seed_base + e, **kw)
        out.append(run_episode(env, policy_fn))
    return summarize(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    actor = Actor()
    actor.load_state_dict(torch.load(os.path.join(LOGS_DIR, "actor.pt"), map_location="cpu"))
    actor.eval()

    def trained(_e, obs, agents):
        return greedy(actor, obs, agents)

    def random_policy(env, _obs, agents):
        return {a: env.action_space(a).sample() for a in agents}

    print(f"Evaluating over {args.episodes} episodes...")
    baseline = evaluate(random_policy, 8, args.episodes, 1000)
    trained_8 = evaluate(trained, 8, args.episodes, 1000)
    scale = {str(n): evaluate(trained, n, args.episodes, 2000) for n in (8, 16, 32)}

    metrics = {
        "episodes": args.episodes,
        "baseline_random_8": baseline,
        "trained_8": trained_8,
        "scale_sweep_trained": scale,
    }
    ensure_dirs()
    with open(os.path.join(LOGS_DIR, "eval.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
