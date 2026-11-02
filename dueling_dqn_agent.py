"""Dueling DQN 예제. dqn_agent.py 와 QNet 만 다름. 사용법과 API 는 README.md 참고."""

# agent.py 의 작성 영역을 채운 예제. TODO 4~6 은 update() 에 구현.

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


# 모델 작성 영역에서 사용하는 네트워크
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


# TODO 1 에서 사용하는 리플레이 버퍼
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
    # 0. 기본명세. name, weights 는 본인 이름으로
    name = "dueling_dqn"
    color = (250, 160, 90)  # (R, G, B)
    weights = "dueling_dqn.pt"  # 대결장은 이 파일을 불러옴

    # 1. 관측 / 액션 형태
    obs_spec = ObsSpec(mode="image", resolution=64)
    action_spec = ActionSpec(mode="discrete")

    # 2. 모델. 구조는 self.cfg 만으로 정해져야 함 (load 할 때 다시 호출됨)
    def setup(self) -> None:
        k = self.cfg.get("frame_stack", 4)
        self.frames = FrameStack(k)
        # ==================== 작성 시작: 모델 ====================
        self.q = DuelingQNet(k * len(IMAGE_CHANNELS), self.obs_spec.resolution).to(self.device)
        self.epsilon = 0.05
        # ==================== 작성 끝 ====================

    # 3. 관측(NumPy) -> 신경망 입력. 기본은 이미지 k 프레임 (B, k*6, R, R)
    def preprocess(self, obs: Observation) -> torch.Tensor:
        img = torch.as_tensor(obs.image, device=self.device)
        return self.frames.push(img).flatten(1, 2)

    # 4. 행동 선택. 대결에서는 explore=False
    def policy(self, x: torch.Tensor, explore: bool) -> np.ndarray:
        # ==================== 작성 시작: 행동 선택 ====================
        action = self.q(x).argmax(dim=1)
        if explore:
            rand = torch.rand(len(action), device=action.device) < self.epsilon
            action[rand] = torch.randint(NUM_ACTIONS, (int(rand.sum()),), device=action.device)
        return action.cpu().numpy()
        # ==================== 작성 끝 ====================

    # 프레임 스택 초기화. done=True 인 원소만
    def reset(self, done: np.ndarray) -> None:
        self.frames.reset(done)

    # 5. 보상
    def reward(self, events: Events, obs: Observation) -> np.ndarray:
        return (events.size_after - events.size_before) / 100.0 - 1.0 * events.died + 5.0 * events.won


def update(agent: DuelingDQNAgent, q_target: nn.Module, optimizer: torch.optim.Optimizer, b: dict[str, torch.Tensor], gamma: float) -> float:
    # TODO 4: 미니배치의 현재 Q(s, a) 추출 (배치 추출은 train 에서 수행)
    # ==================== 작성 시작: TODO 4 ====================
    q = agent.q(b["x"]).gather(1, b["action"].unsqueeze(1)).squeeze(1)
    # ==================== 작성 끝 ====================

    # TODO 5: no_grad 안에서 TD 타깃 계산
    # ==================== 작성 시작: TODO 5 ====================
    with torch.no_grad():
        target = b["reward"] + gamma * (1 - b["terminated"]) * q_target(b["x_next"]).max(dim=1).values
    # ==================== 작성 끝 ====================

    # TODO 6: loss 계산, 역전파, 파라미터 갱신
    # ==================== 작성 시작: TODO 6 ====================
    loss = F.smooth_l1_loss(q, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.q.parameters(), 10.0)
    optimizer.step()
    return float(loss)
    # ==================== 작성 끝 ====================


# 6. 학습 루프. TODO 번호는 agent.py 의 TODO 1~8 에 대응.
def train(cfg: Config) -> None:
    agent = DuelingDQNAgent(cfg)
    env = make_env(cfg, agent)
    run = wandb.init(project="cell-arena", name=agent.name, config=cfg.to_dict())
    # TODO 1: 학습 준비
    # ==================== 작성 시작: TODO 1 ====================
    q_target = copy.deepcopy(agent.q)
    optimizer = torch.optim.Adam(agent.q.parameters(), lr=cfg.get("lr", 1e-4))
    buffer = ReplayBuffer(cfg.get("buffer_size", 20_000))
    gamma = cfg.get("gamma", 0.99)
    batch_size = cfg.get("batch_size", 256)
    learning_starts = cfg.get("learning_starts", 50_000)  # 이하 단위는 샘플 수
    target_update = cfg.get("target_update", 20_000)
    eps_start, eps_end, eps_decay = cfg.get("eps_start", 1.0), cfg.get("eps_end", 0.05), cfg.get("eps_decay", 1_000_000)
    # ==================== 작성 끝 ====================

    ep_reward = np.zeros(cfg.num_envs)
    recent_returns: list[float] = []
    log_every = 10_000
    next_log = log_every
    # TODO 1 의 타깃 동기화 시점과 첫 업데이트 전 loss 초기화
    next_target = target_update
    loss = float("nan")
    start_time = time.time()

    obs = env.reset()
    agent.reset(np.ones(cfg.num_envs, dtype=bool))
    samples = 0
    while samples < cfg.total_samples:
        # TODO 2: 탐험률 갱신 + 현재 입력 x 확보
        # ==================== 작성 시작: TODO 2 ====================
        agent.epsilon = max(eps_end, eps_start - (eps_start - eps_end) * samples / eps_decay)
        with torch.no_grad():  # act() 와 같음. x 를 버퍼에 넣으려고 나눠서 호출
            x = agent.preprocess(obs)
            action = agent.policy(x, explore=True)
        # ==================== 작성 끝 ====================

        out = env.step(action)
        reward = agent.reward(out.events, out.final_obs)
        samples += cfg.num_envs

        # TODO 3: 다음 입력과 전이 저장
        # ==================== 작성 시작: TODO 3 ====================
        x_next = agent.frames.peek(torch.as_tensor(out.final_obs.image, device=agent.device)).flatten(1, 2)
        buffer.add(
            x=x, action=torch.as_tensor(action), reward=torch.as_tensor(reward, dtype=torch.float32),
            x_next=x_next, terminated=torch.as_tensor(out.terminated, dtype=torch.float32),
        )
        # ==================== 작성 끝 ====================

        # TODO 4~6: learning_starts 이후 미니배치 추출 -> update() 의 해당 단계 실행
        if samples >= learning_starts:
            loss = update(agent, q_target, optimizer, buffer.sample(batch_size, agent.device), gamma)

        # TODO 7: 타깃 네트워크 동기화
        # ==================== 작성 시작: TODO 7 ====================
        if samples >= next_target:
            next_target += target_update
            q_target.load_state_dict(agent.q.state_dict())
        # ==================== 작성 끝 ====================

        episode_done = out.terminated | out.truncated
        ep_reward += reward
        recent_returns.extend(ep_reward[episode_done].tolist())
        recent_returns[:-200] = []
        ep_reward[episode_done] = 0.0

        if samples >= next_log:
            next_log += log_every
            # TODO 8: loss 와 탐험률 로그
            run.log({
                "train/reward_step": float(reward.mean()),
                "train/episode_return": float(np.mean(recent_returns)) if recent_returns else float("nan"),
                "train/size": float(out.final_obs.self_state[:, 0].mean()),
                "train/deaths": int(out.events.died.sum()),
                "train/samples_per_sec": samples / (time.time() - start_time),
                # ==================== 작성 시작: TODO 8 ====================
                "train/loss": loss,
                "train/epsilon": agent.epsilon,
                # ==================== 작성 끝 ====================
            }, step=samples)

        agent.reset(episode_done | out.events.died)
        obs = out.obs

    agent.save()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    train(load_config(parser.parse_args().config))
