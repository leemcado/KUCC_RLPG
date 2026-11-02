"""Cell Arena 에이전트. 사용법과 API 는 README.md 참고."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from cell_arena import (
    ActionSpec,
    Config,
    Events,
    Observation,
    ObsSpec,
    StudentAgent,
    load_config,
    make_env,
)
from torch import nn

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
class QNet(nn.Module):
    def __init__(self, in_channels: int, resolution: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        n = self.conv(torch.zeros(1, in_channels, resolution, resolution)).shape[1]
        self.head = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, NUM_ACTIONS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x.float()))


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


class MyAgent(StudentAgent):
    # 0. 기본명세. name, weights 는 본인 이름으로
    name = "my_agent"
    color = (90, 160, 250)  # (R, G, B)
    weights = "my_agent.pt"  # 대결장은 이 파일을 불러옴

    # 1. 관측 / 액션 형태
    obs_spec = ObsSpec(mode="image", resolution=64)
    # obs_spec = ObsSpec(mode="state", max_objects=32)
    #   image: self_state (B, 5), image (B, 6, R, R)                R = resolution
    #   state: self_state (B, 5), objects (B, M, 8), mask (B, M)    M = max_objects

    action_spec = ActionSpec(mode="discrete")
    # action_spec = ActionSpec(mode="continuous")
    #   discrete:   (B,)    정수 0~17
    #   continuous: (B, 3)  float [theta, move, dash]

    # 2. 모델. 구조는 self.cfg 만으로 정해져야 함 (load 할 때 다시 호출됨)
    def setup(self) -> None:
        self.frames = FrameStack(self.cfg.get("frame_stack", 4))
        # DQN: 입력 (B, k*6, R, R) -> 행동별 Q값 (B, 18) 인 모델을 self.q 로 구성.
        # Dueling DQN: V (B, 1), A (B, 18) 을 Q = V + A - mean(A) 로 결합.
        # mean 은 각 샘플의 행동 축으로 계산. 모델은 self.device 에 배치.
        # ==================== 작성 시작: 모델 ====================
        raise NotImplementedError
        # ==================== 작성 끝 ====================

    # 3. 관측(NumPy) -> 신경망 입력. 기본은 이미지 k 프레임 (B, k*6, R, R)
    def preprocess(self, obs: Observation) -> torch.Tensor:
        img = torch.as_tensor(obs.image, device=self.device)
        return self.frames.push(img).flatten(1, 2)

    # 4. 행동 선택. 대결에서는 explore=False
    def policy(self, x: torch.Tensor, explore: bool) -> np.ndarray:
        # DQN: Q값이 가장 큰 행동 선택. explore=True 일 때만 epsilon 확률로 무작위 선택.
        # 반환: CPU NumPy 정수 배열 (B,), 값은 0~17. explore=False 에서는 탐험하지 않음.
        # ==================== 작성 시작: 행동 선택 ====================
        raise NotImplementedError
        # ==================== 작성 끝 ====================

    # 프레임 스택 초기화. done=True 인 원소만
    def reset(self, done: np.ndarray) -> None:
        self.frames.reset(done)

    # 5. 보상
    def reward(self, events: Events, obs: Observation) -> np.ndarray:
        return (events.size_after - events.size_before) / 100.0 - 1.0 * events.died + 5.0 * events.won


# 6. 학습 루프. 아래 TODO 1~8 은 DQN / Dueling DQN 실습 순서.
# PPO 를 구현할 때는 리플레이 버퍼와 TD 타깃 대신 ppo_agent.py 의 흐름 참고.
def train(cfg: Config) -> None:
    agent = MyAgent(cfg)
    env = make_env(cfg, agent)
    run = wandb.init(project="cell-arena", name=agent.name, config=cfg.to_dict())
    # TODO 1: 학습 준비
    # - agent.q 와 같은 초기 가중치의 별도 타깃 네트워크, 옵티마이저, 리플레이 버퍼 준비.
    # - 버퍼는 전이를 복사해 보관하고, 무작위 미니배치를 꺼낼 수 있게 구현.
    # - gamma, batch_size, learning_starts, target_update, epsilon 스케줄을 cfg 에서 읽기.
    # - learning_starts / target_update / eps_decay 의 단위는 루프 횟수가 아닌 샘플 수.
    # - 첫 업데이트 전 loss 는 미정(NaN). 타깃 동기화 시점도 초기화.
    # ==================== 작성 시작: TODO 1 ====================

    # ==================== 작성 끝 ====================

    ep_reward = np.zeros(cfg.num_envs)
    recent_returns: list[float] = []
    log_every = 10_000
    next_log = log_every
    start_time = time.time()

    obs = env.reset()
    agent.reset(np.ones(cfg.num_envs, dtype=bool))
    samples = 0
    while samples < cfg.total_samples:
        # TODO 2: 탐험률 갱신 + 현재 입력 x 확보
        # 아래 act 호출을 no_grad 안의 preprocess(obs) -> policy(x, explore=True) 로 대체.
        # x 는 (num_envs, k*6, R, R). act 와 preprocess 를 둘 다 호출하면 프레임이 두 번 쌓임.
        # ==================== 작성 시작: TODO 2 ====================
        action = agent.act(obs, explore=True)
        # ==================== 작성 끝 ====================

        out = env.step(action)
        reward = agent.reward(out.events, out.final_obs)
        samples += cfg.num_envs

        # TODO 3: 다음 입력과 전이 저장
        # out.final_obs.image 를 agent.device 의 텐서로 바꾸고 frames.peek 후 flatten(1, 2).
        # 이것이 x_next. preprocess 를 다시 호출하거나 리셋 후 out.obs 를 쓰지 않기.
        # (x, action, reward, x_next, terminated) 를 버퍼에 저장. 행동은 정수형,
        # reward 와 terminated 는 float32. 이미지 저장 시 불필요한 float32 변환은 피하기.
        # ==================== 작성 시작: TODO 3 ====================

        # ==================== 작성 끝 ====================

        # TODO 4: learning_starts 이후 미니배치를 뽑아 현재 Q(s, a) 추출
        # 배치를 agent.device 로 옮기고, 모델 출력 (N, 18) 에서 저장된 행동의 값만 선택.
        # 결과는 (N,). 여기서 N = batch_size 이며 num_envs 와 다를 수 있음.
        # ==================== 작성 시작: TODO 4 ====================

        # ==================== 작성 끝 ====================

        # TODO 5: no_grad 안에서 TD 타깃 계산
        # y = reward + gamma * (1 - terminated) * max_a Q_target(x_next, a)
        # truncated 는 부트스트랩을 막지 않음. died 도 terminated 와 구분.
        # ==================== 작성 시작: TODO 5 ====================

        # ==================== 작성 끝 ====================

        # TODO 6: Q(s, a) 와 y 사이 Huber loss -> 기울기 초기화 -> 역전파 -> 파라미터 갱신
        # 두 텐서 모두 (N,) 인지 확인해 의도치 않은 브로드캐스팅 방지.
        # 기울기 클리핑을 적용하고, 로그용 loss 는 계산 그래프에서 분리한 숫자로 보관.
        # ==================== 작성 시작: TODO 6 ====================

        # ==================== 작성 끝 ====================

        # TODO 7: target_update 샘플마다 학습 모델의 가중치를 타깃 네트워크로 복사
        # samples 가 다음 동기화 시점을 넘었는지 비교. num_envs 만큼 증가하므로
        # samples % target_update == 0 만 검사하면 동기화 시점을 놓칠 수 있음.
        # ==================== 작성 시작: TODO 7 ====================

        # ==================== 작성 끝 ====================

        episode_done = out.terminated | out.truncated
        ep_reward += reward
        recent_returns.extend(ep_reward[episode_done].tolist())
        recent_returns[:-200] = []
        ep_reward[episode_done] = 0.0

        if samples >= next_log:
            next_log += log_every
            # TODO 8: 아래 로그에 train/loss 와 train/epsilon 추가.
            # learning_starts 이후 loss 가 유한한지, epsilon 이 설정대로 줄어드는지 확인.
            run.log({
                "train/reward_step": float(reward.mean()),
                "train/episode_return": float(np.mean(recent_returns)) if recent_returns else float("nan"),
                "train/size": float(out.final_obs.self_state[:, 0].mean()),
                "train/deaths": int(out.events.died.sum()),
                "train/samples_per_sec": samples / (time.time() - start_time),
                # ==================== 작성 시작: TODO 8 ====================

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
