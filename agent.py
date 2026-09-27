"""Cell Arena 에이전트. 사용법과 API 는 README.md 참고."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from cell_arena import ActionSpec, Config, Events, Observation, ObsSpec, StudentAgent, load_config, make_env


class MyAgent(StudentAgent):
    # 0. 이름·색·가중치 파일
    name = "my_agent"
    color = (90, 160, 250)  # (R, G, B)
    weights = "my_agent.pt"

    # 1. 관측 / 액션 형태
    obs_spec = ObsSpec(mode="state", max_objects=32)
    action_spec = ActionSpec(mode="discrete")

    # 2. 모델. 구조는 self.cfg 만으로 정해져야 한다 (load 할 때 다시 호출됨)
    def setup(self) -> None:
        raise NotImplementedError

    # 3. 관측(NumPy) → 신경망 입력
    def preprocess(self, obs: Observation) -> torch.Tensor:
        raise NotImplementedError

    # 4. 행동 선택. 대결에서는 explore=False
    def policy(self, x: torch.Tensor, explore: bool) -> np.ndarray:
        raise NotImplementedError

    # (선택) 프레임 스택·RNN 상태 초기화. done=True 인 원소만
    def reset(self, done: np.ndarray) -> None:
        pass

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

        # TODO: 전이 저장, 업데이트

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
