"""DQN 예제. 이미지 관측 + 이산 행동. 가장 기초적인 형태만 구현했다.

    python dqn_agent.py
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from cell_arena import ActionSpec, Config, Events, Observation, ObsSpec, StudentAgent, load_config, make_env

NUM_ACTIONS = 18


class FrameStack:
    """최근 k 개 관측을 쌓는다. 다른 세포의 속도는 관측에 없어서 여러 프레임이 필요하다.

    push(x): (B, ...) → (B, k, ...). 초기화된 원소는 첫 프레임을 k 번 복제한다.
    peek(x): push 와 같지만 저장하지 않는다 (학습 루프에서 final_obs 로 다음 상태 만들 때).
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.frames: torch.Tensor | None = None
        self.fresh: torch.Tensor | None = None

    def reset(self, done: np.ndarray) -> None:
        if self.frames is None or len(done) != len(self.frames):
            self.frames = None
            return
        self.fresh |= torch.as_tensor(done, device=self.fresh.device)

    def push(self, x: torch.Tensor) -> torch.Tensor:
        if self.frames is None or self.frames.shape[0] != x.shape[0] or self.frames.device != x.device:
            self.frames = x.unsqueeze(1).repeat_interleave(self.k, dim=1)
            self.fresh = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        else:
            self.frames = self.peek(x)
            self.fresh[:] = False
        return self.frames

    def peek(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.frames[:, 1:], x.unsqueeze(1)], dim=1)
        out[self.fresh] = x[self.fresh].unsqueeze(1)
        return out


class Encoder(nn.Module):
    """이미지 (B, C, R, R) + 내 상태 (B, 3) → 특징 (B, 256)."""

    def __init__(self, in_channels: int, resolution: int, dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        n = self.conv(torch.zeros(1, in_channels, resolution, resolution)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n + 3, dim), nn.ReLU())

    def forward(self, img: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.conv(img.float()), vec], dim=1))


class QNet(nn.Module):
    def __init__(self, in_channels: int, resolution: int) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, resolution)
        self.q = nn.Linear(256, NUM_ACTIONS)

    def forward(self, img: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return self.q(self.encoder(img, vec))


class DQNAgent(StudentAgent):
    name = "dqn"
    color = (90, 160, 250)
    weights = "dqn.pt"

    obs_spec = ObsSpec(mode="image", resolution=64)
    action_spec = ActionSpec(mode="discrete")

    def setup(self) -> None:
        k = self.cfg.get("frame_stack", 4)
        self.frames = FrameStack(k)
        self.q = QNet(k * 5, self.obs_spec.resolution).to(self.device)
        self.epsilon = 0.05  # 학습 루프가 바꾼다

    def _features(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.as_tensor(obs.image, device=self.device)
        s = torch.as_tensor(obs.self_state, device=self.device)
        vec = torch.stack([torch.log(s[:, 0].clamp(min=1) / 100), s[:, 3], s[:, 4]], dim=1)
        return img, vec

    # 이미지는 (B, k·5, R, R) uint8 로 쌓고, 내 상태는 [log(크기/100), v_x, v_y]
    def preprocess(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        img, vec = self._features(obs)
        return self.frames.push(img).flatten(1, 2), vec

    def peek(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        img, vec = self._features(obs)
        return self.frames.peek(img).flatten(1, 2), vec

    # ε-greedy
    def policy(self, x: tuple[torch.Tensor, torch.Tensor], explore: bool) -> np.ndarray:
        action = self.q(*x).argmax(dim=1)
        if explore:
            rand = torch.rand(len(action), device=action.device) < self.epsilon
            action[rand] = torch.randint(NUM_ACTIONS, (int(rand.sum()),), device=action.device)
        return action.cpu().numpy()

    def reset(self, done: np.ndarray) -> None:
        self.frames.reset(done)

    def reward(self, events: Events, obs: Observation) -> np.ndarray:
        return (events.size_after - events.size_before) / 100.0 - 1.0 * events.died + 5.0 * events.won


class ReplayBuffer:
    """전이를 CPU 에 원형으로 저장한다. 이미지는 uint8 그대로."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: dict[str, torch.Tensor] = {}
        self.pos = 0
        self.size = 0

    def add(self, **batch: torch.Tensor) -> None:
        n = len(next(iter(batch.values())))
        if not self.data:
            self.data = {k: torch.zeros((self.capacity, *v.shape[1:]), dtype=v.dtype) for k, v in batch.items()}
        idx = (self.pos + torch.arange(n)) % self.capacity
        for k, v in batch.items():
            self.data[k][idx] = v.cpu()
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, n: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = torch.randint(self.size, (n,))
        return {k: v[idx].to(device) for k, v in self.data.items()}


def train(cfg: Config) -> None:
    agent = DQNAgent(cfg)
    env = make_env(cfg, agent)
    run = wandb.init(project="cell-arena", name=agent.name, config=cfg.to_dict())

    gamma = cfg.get("gamma", 0.99)
    batch_size = cfg.get("batch_size", 256)
    learning_starts = cfg.get("learning_starts", 50_000)  # 샘플 수
    target_update = cfg.get("target_update", 20_000)      # 샘플 수
    eps_start, eps_end = cfg.get("eps_start", 1.0), cfg.get("eps_end", 0.05)
    eps_decay = cfg.get("eps_decay", 1_000_000)           # 샘플 수

    target_q = copy.deepcopy(agent.q)
    optimizer = torch.optim.Adam(agent.q.parameters(), lr=cfg.get("lr", 1e-4))
    buffer = ReplayBuffer(cfg.get("buffer_size", 20_000))

    ep_reward = np.zeros(cfg.num_envs)
    recent_returns: list[float] = []
    log_every = 10_000
    next_log = log_every
    next_target = target_update
    start_time = time.time()
    loss = torch.tensor(0.0)

    obs = env.reset()
    agent.reset(np.ones(cfg.num_envs, dtype=bool))
    samples = 0
    while samples < cfg.total_samples:
        agent.epsilon = max(eps_end, eps_start - (eps_start - eps_end) * samples / eps_decay)
        # act() 대신 preprocess → policy 를 직접 불러 x 를 버퍼에 넣는다
        with torch.no_grad():
            x = agent.preprocess(obs)
            action = agent.policy(x, explore=True)
        out = env.step(action)
        reward = agent.reward(out.events, out.final_obs)
        with torch.no_grad():
            x_next = agent.peek(out.final_obs)
        samples += cfg.num_envs

        buffer.add(
            img=x[0], vec=x[1], action=torch.as_tensor(action),
            reward=torch.as_tensor(reward, dtype=torch.float32),
            next_img=x_next[0], next_vec=x_next[1],
            terminated=torch.as_tensor(out.terminated, dtype=torch.float32),
        )

        if samples >= learning_starts:
            b = buffer.sample(batch_size, agent.device)
            q = agent.q(b["img"], b["vec"]).gather(1, b["action"].unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                # truncated 는 terminated 가 아니므로 부트스트랩한다
                target = b["reward"] + gamma * (1 - b["terminated"]) * target_q(b["next_img"], b["next_vec"]).max(1).values
            loss = F.smooth_l1_loss(q, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.q.parameters(), 10.0)
            optimizer.step()

        if samples >= next_target:
            next_target += target_update
            target_q.load_state_dict(agent.q.state_dict())

        episode_done = out.terminated | out.truncated
        ep_reward += reward
        recent_returns.extend(ep_reward[episode_done].tolist())
        recent_returns[:-200] = []
        ep_reward[episode_done] = 0.0

        if samples >= next_log:
            next_log += log_every
            run.log({
                "train/reward_step": float(reward.mean()),
                "train/episode_return": float(np.mean(recent_returns)) if recent_returns else float("nan"),
                "train/size": float(out.final_obs.self_state[:, 0].mean()),
                "train/deaths": int(out.events.died.sum()),
                "train/samples_per_sec": samples / (time.time() - start_time),
                "train/loss": float(loss),
                "train/epsilon": agent.epsilon,
            }, step=samples)

        agent.reset(episode_done | out.events.died)
        obs = out.obs

    agent.save()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    train(load_config(parser.parse_args().config))
