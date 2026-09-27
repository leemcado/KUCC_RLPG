"""Dueling DQN 예제. dqn_agent.py 와 QNet 만 다름. 사용법과 API 는 README.md 참고."""

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

from cell_arena import (
    IMAGE_CHANNELS,
    ActionSpec,
    Config,
    Events,
    Observation,
    ObsSpec,
    StudentAgent,
    load_config,
    make_env,
)

NUM_ACTIONS = 18


class FrameStack:
    """최근 k 프레임 스택. (B, ...) -> (B, k, ...)"""

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

    # push 와 같지만 저장하지 않음. final_obs 로 다음 상태를 만들 때 사용
    def peek(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.frames[:, 1:], x.unsqueeze(1)], dim=1)
        out[self.fresh] = x[self.fresh].unsqueeze(1)  # 방금 리셋된 원소는 x 로 채움
        return out


class DuelingQNet(nn.Module):
    def __init__(self, in_channels: int, resolution: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        n = self.conv(torch.zeros(1, in_channels, resolution, resolution)).shape[1]
        self.v = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 1))
        self.a = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, NUM_ACTIONS))

    # Q = V + A - mean(A)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.float())
        a = self.a(h)
        return self.v(h) + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
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


class DuelingDQNAgent(StudentAgent):
    # 0. 기본명세
    name = "dueling_dqn"
    color = (250, 160, 90)  # (R, G, B)
    weights = "dueling_dqn.pt"

    # 1. 관측 / 액션 형태
    obs_spec = ObsSpec(mode="image", resolution=64)
    action_spec = ActionSpec(mode="discrete")

    # 2. 모델
    def setup(self) -> None:
        k = self.cfg.get("frame_stack", 4)
        self.frames = FrameStack(k)
        self.q = DuelingQNet(k * len(IMAGE_CHANNELS), self.obs_spec.resolution).to(self.device)
        self.epsilon = 0.05

    # 3. 관측(NumPy) -> 신경망 입력. 이미지 k 프레임 (B, k*6, R, R)
    def preprocess(self, obs: Observation) -> torch.Tensor:
        img = torch.as_tensor(obs.image, device=self.device)
        return self.frames.push(img).flatten(1, 2)

    # 4. 행동 선택. epsilon-greedy
    def policy(self, x: torch.Tensor, explore: bool) -> np.ndarray:
        action = self.q(x).argmax(dim=1)
        if explore:
            rand = torch.rand(len(action), device=action.device) < self.epsilon
            action[rand] = torch.randint(NUM_ACTIONS, (int(rand.sum()),), device=action.device)
        return action.cpu().numpy()

    # 프레임 스택 초기화. done=True 인 원소만
    def reset(self, done: np.ndarray) -> None:
        self.frames.reset(done)

    # 5. 보상
    def reward(self, events: Events, obs: Observation) -> np.ndarray:
        return (events.size_after - events.size_before) / 100.0 - 1.0 * events.died + 5.0 * events.won


def update(agent: DuelingDQNAgent, q_target: nn.Module, optimizer: torch.optim.Optimizer, b: dict[str, torch.Tensor], gamma: float) -> float:
    q = agent.q(b["x"]).gather(1, b["action"].unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        target = b["reward"] + gamma * (1 - b["terminated"]) * q_target(b["x_next"]).max(dim=1).values
    loss = F.smooth_l1_loss(q, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.q.parameters(), 10.0)
    optimizer.step()
    return float(loss)


# 6. 학습 루프
def train(cfg: Config) -> None:
    agent = DuelingDQNAgent(cfg)
    env = make_env(cfg, agent)
    run = wandb.init(project="cell-arena", name=agent.name, config=cfg.to_dict())
    q_target = copy.deepcopy(agent.q)
    optimizer = torch.optim.Adam(agent.q.parameters(), lr=cfg.get("lr", 1e-4))
    buffer = ReplayBuffer(cfg.get("buffer_size", 20_000))
    gamma = cfg.get("gamma", 0.99)
    batch_size = cfg.get("batch_size", 256)
    learning_starts = cfg.get("learning_starts", 50_000)  # 이하 단위는 샘플 수
    target_update = cfg.get("target_update", 20_000)
    eps_start, eps_end, eps_decay = cfg.get("eps_start", 1.0), cfg.get("eps_end", 0.05), cfg.get("eps_decay", 1_000_000)

    ep_reward = np.zeros(cfg.num_envs)
    recent_returns: list[float] = []
    log_every = 10_000
    next_log = log_every
    next_target = target_update
    loss = float("nan")
    start_time = time.time()

    obs = env.reset()
    agent.reset(np.ones(cfg.num_envs, dtype=bool))
    samples = 0
    while samples < cfg.total_samples:
        agent.epsilon = max(eps_end, eps_start - (eps_start - eps_end) * samples / eps_decay)
        with torch.no_grad():  # act() 와 같음. x 를 버퍼에 넣으려고 나눠서 호출
            x = agent.preprocess(obs)
            action = agent.policy(x, explore=True)
        out = env.step(action)
        reward = agent.reward(out.events, out.final_obs)
        samples += cfg.num_envs

        x_next = agent.frames.peek(torch.as_tensor(out.final_obs.image, device=agent.device)).flatten(1, 2)
        buffer.add(
            x=x, action=torch.as_tensor(action), reward=torch.as_tensor(reward, dtype=torch.float32),
            x_next=x_next, terminated=torch.as_tensor(out.terminated, dtype=torch.float32),
        )
        if samples >= learning_starts:
            loss = update(agent, q_target, optimizer, buffer.sample(batch_size, agent.device), gamma)
        if samples >= next_target:
            next_target += target_update
            q_target.load_state_dict(agent.q.state_dict())

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
                "train/loss": loss,
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
