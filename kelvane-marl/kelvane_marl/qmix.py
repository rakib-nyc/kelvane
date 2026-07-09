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

"""QMIX trainer for the cooperative gridworld (discrete value factorization).

Each agent learns a recurrent Q-network (DRQN) over its local observation; a
monotonic mixing network — whose weights come from hypernetworks conditioned on
the global state — combines the per-agent Q values into a single Q_tot, so
argmax over the joint action equals per-agent argmax (decentralized execution).
A runnable reference, not a full training run.
"""

import argparse
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kelvane_marl.env_grid import GridWorld
from kelvane_marl.paths import LOGS_DIR, ensure_dirs

N_AGENTS = 8
OBS_DIM = 11 * 11 * 4
STATE_DIM = 100 * 100 * 4
N_ACTIONS = 7
RNN_HIDDEN = 64
MIXER_EMBED = 32
LR = 5e-4
GAMMA = 0.99
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.995
SYNC_INTERVAL = 10
BUFFER_CAP = 500
BATCH = 8
MAX_STEPS = 30


class AgentDRQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, RNN_HIDDEN)
        self.rnn = nn.GRUCell(RNN_HIDDEN, RNN_HIDDEN)
        self.fc_out = nn.Linear(RNN_HIDDEN, N_ACTIONS)

    def init_hidden(self, batch):
        return torch.zeros(batch, RNN_HIDDEN)

    def forward(self, obs, hidden):
        h = self.rnn(F.relu(self.fc1(obs)), hidden)
        return self.fc_out(h), h


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hw1 = nn.Linear(STATE_DIM, N_AGENTS * MIXER_EMBED)
        self.hb1 = nn.Linear(STATE_DIM, MIXER_EMBED)
        self.hw2 = nn.Linear(STATE_DIM, MIXER_EMBED)
        self.hb2 = nn.Sequential(nn.Linear(STATE_DIM, MIXER_EMBED), nn.ReLU(), nn.Linear(MIXER_EMBED, 1))

    def forward(self, qs, state):
        b = qs.size(0)
        w1 = torch.abs(self.hw1(state)).view(b, N_AGENTS, MIXER_EMBED)
        b1 = self.hb1(state).view(b, 1, MIXER_EMBED)
        hidden = F.elu(torch.bmm(qs.view(b, 1, N_AGENTS), w1) + b1)
        w2 = torch.abs(self.hw2(state)).view(b, MIXER_EMBED, 1)
        b2 = self.hb2(state).view(b, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(b, 1)


def flatten_obs(obs_dict, agents):
    return torch.FloatTensor(np.stack([obs_dict[a].reshape(-1) for a in agents]))


def train(episodes=20, seed=0):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    env = GridWorld(n_agents=N_AGENTS)
    agent, frozen = AgentDRQN(), AgentDRQN()
    frozen.load_state_dict(agent.state_dict())
    mixer, tmixer = Mixer(), Mixer()
    tmixer.load_state_dict(mixer.state_dict())
    opt = torch.optim.Adam(list(agent.parameters()) + list(mixer.parameters()), lr=LR)
    buf = deque(maxlen=BUFFER_CAP)
    eps, gstep = EPS_START, 0

    print("QMIX training (reference run)...")
    for ep in range(episodes):
        obs_d, _ = env.reset()
        agents = env.possible_agents[:]
        hidden = agent.init_hidden(N_AGENTS)
        ep_r = 0.0
        for _ in range(MAX_STEPS):
            obs = flatten_obs(obs_d, agents)
            state = torch.FloatTensor(env.state().reshape(-1))
            q, hidden = agent(obs, hidden)
            acts = {}
            for i, a in enumerate(agents):
                acts[a] = random.randrange(N_ACTIONS) if random.random() < eps else int(torch.argmax(q[i]))
            nobs_d, rewards, term, trunc, _ = env.step(acts)
            team_r = float(np.mean([rewards[a] for a in agents]))
            done = all(term.values()) or all(trunc.values())
            nobs = flatten_obs(nobs_d, agents)
            nstate = torch.FloatTensor(env.state().reshape(-1))
            av = torch.LongTensor([acts[a] for a in agents])
            buf.append((obs, state, av, team_r, nobs, nstate, done))
            obs_d = nobs_d
            ep_r += team_r

            if len(buf) >= BATCH:
                batch = random.sample(buf, BATCH)
                bo = torch.stack([b[0] for b in batch])
                bs = torch.stack([b[1] for b in batch])
                ba = torch.stack([b[2] for b in batch])
                br = torch.FloatTensor([b[3] for b in batch])
                bno = torch.stack([b[4] for b in batch])
                bns = torch.stack([b[5] for b in batch])
                bd = torch.FloatTensor([float(b[6]) for b in batch])
                h0 = agent.init_hidden(BATCH * N_AGENTS)
                qa, _ = agent(bo.view(-1, OBS_DIM), h0)
                qa = qa.view(BATCH, N_AGENTS, N_ACTIONS)
                chosen = qa.gather(2, ba.unsqueeze(2)).squeeze(2)
                qtot = mixer(chosen, bs)
                with torch.no_grad():
                    tqa, _ = frozen(bno.view(-1, OBS_DIM), h0)
                    tqa = tqa.view(BATCH, N_AGENTS, N_ACTIONS)
                    ttot = tmixer(tqa.max(2).values, bns)
                    y = br.unsqueeze(1) + GAMMA * (1 - bd.unsqueeze(1)) * ttot
                loss = F.mse_loss(qtot, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(agent.parameters()) + list(mixer.parameters()), 10.0)
                opt.step()
                gstep += 1
                if gstep % SYNC_INTERVAL == 0:
                    frozen.load_state_dict(agent.state_dict())
                    tmixer.load_state_dict(mixer.state_dict())
            if done:
                break
        eps = max(EPS_END, eps * EPS_DECAY)
        print(f"  episode {ep:02d}  team_return={ep_r:7.2f}  eps={eps:.3f}")

    ensure_dirs()
    torch.save({"agent": agent.state_dict(), "mixer": mixer.state_dict()},
               os.path.join(LOGS_DIR, "qmix.pt"))
    print(f"Saved checkpoint to {os.path.join(LOGS_DIR, 'qmix.pt')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()
    train(args.episodes)
