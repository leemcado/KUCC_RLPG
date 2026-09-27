"""Cell Arena 에이전트. 사용법과 API 는 README.md 참고."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from cell_arena import ActionSpec, Config, Events, Observation, ObsSpec, StudentAgent, load_config, make_env


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


class MyAgent(StudentAgent):
    # 0. 기본명세. name, weights 는 본인 이름으로
    name = "my_agent"
    color = (90, 160, 250)  # (R, G, B)
    weights = "my_agent.pt"  # 대결장은 이 파일을 불러옴

    # 1. 관측 / 액션 형태
    obs_spec = ObsSpec(mode="image", resolution=64)
    # obs_spec = ObsSpec(mode="state", max_objects=32)
    #   image: self_state (B, 5), image (B, 5, R, R)                R = resolution
    #   state: self_state (B, 5), objects (B, M, 7), mask (B, M)    M = max_objects

    action_spec = ActionSpec(mode="discrete")
    # action_spec = ActionSpec(mode="continuous")
    #   discrete:   (B,)    정수 0~17
    #   continuous: (B, 3)  float [theta, move, dash]

    # 2. 모델. 구조는 self.cfg 만으로 정해져야 함 (load 할 때 다시 호출됨)
    def setup(self) -> None:
        self.frames = FrameStack(self.cfg.get("frame_stack", 4))
        raise NotImplementedError

    # 3. 관측(NumPy) -> 신경망 입력. 기본은 이미지 k 프레임 (B, k*5, R, R)
    def preprocess(self, obs: Observation) -> torch.Tensor:
        img = torch.as_tensor(obs.image, device=self.device)
        return self.frames.push(img).flatten(1, 2)

    # 4. 행동 선택. 대결에서는 explore=False
    def policy(self, x: torch.Tensor, explore: bool) -> np.ndarray:
        raise NotImplementedError

    # 프레임 스택 초기화. done=True 인 원소만
    def reset(self, done: np.ndarray) -> None:
        self.frames.reset(done)

    # 5. 보상
    def reward(self, events: Events, obs: Observation) -> np.ndarray:
        return (events.size_after - events.size_before) / 100.0 - 1.0 * events.died + 5.0 * events.won


# 6. 학습 루프
def train(cfg: Config) -> None:
    agent = MyAgent(cfg)
    env = make_env(cfg, agent)
    run = wandb.init(project="cell-arena", name=agent.name, config=cfg.to_dict())
    # TODO: 옵티마이저, 버퍼

    ep_reward = np.zeros(cfg.num_envs)
    recent_returns: list[float] = []
    log_every = 10_000
    next_log = log_every
    start_time = time.time()

    obs = env.reset()
    agent.reset(np.ones(cfg.num_envs, dtype=bool))
    samples = 0
    while samples < cfg.total_samples:
        action = agent.act(obs, explore=True)
        out = env.step(action)
        reward = agent.reward(out.events, out.final_obs)
        samples += cfg.num_envs

        # TODO: 전이 저장, 업데이트. 다음 상태는 agent.frames.peek 로 만듦 (preprocess 를 또 부르면 프레임이 두 번 쌓임)

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
            }, step=samples)

        agent.reset(episode_done | out.events.died)
        obs = out.obs

    agent.save()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    train(load_config(parser.parse_args().config))
