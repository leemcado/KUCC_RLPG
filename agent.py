"""Cell Arena 학생 에이전트 — 이 파일 하나가 학습 코드이자 제출물이다.

    학습   python agent.py                     (같은 폴더의 config.yaml)
           python agent.py --config dqn.yaml
    점검   cell-arena-check agent.py
    제출   이 파일(<이름>.py) + 가중치 파일(weights)

채울 곳 (TODO)
    ① 입출력 형태     obs_spec / action_spec
    ② 모델           setup()       인코더 + 알고리즘별 머리 (DQN / Dueling DQN / PPO)
    ③ 전처리·피처     preprocess()
    ④ 행동 선택       policy()
    ⑤ 보상           reward()
    ⑥ 학습 루프       train()       데이터 흐름(버퍼·업데이트 주기)과 로그(wandb)
고칠 수 없는 것: __init__, act, save, load — StudentAgent 가 정한다 (재정의하면 불러올 때 오류)
API 설명: README.md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from cell_arena import ActionSpec, Config, Events, Observation, ObsSpec, StudentAgent, load_config, make_env


class MyAgent(StudentAgent):
    # ⓪ 이름·색·가중치 파일
    name = "my_agent"
    color = (90, 160, 250)  # (R, G, B) 또는 "#5aa0fa"
    weights = "my_agent.pt"

    # ① 입출력 형태
    #   obs    state: self_state(B,5) + objects(B,M,7) + mask(B,M)  /  image: self_state(B,5) + image(B,5,R,R)
    #   action discrete(B,)∈[0,18)  /  multibinary(B,5)  /  continuous(B,3)[θ,move,dash]
    obs_spec = ObsSpec(mode="state", max_objects=32)
    action_spec = ActionSpec(mode="discrete")

    # ② 모델 — self.cfg, self.device 사용 가능. torch.nn.Module 속성은 자동 저장·복원된다
    def setup(self) -> None:
        # TODO: 인코더 + 알고리즘 머리 (DQN: Q / Dueling: V+A / PPO: 정책+가치)
        raise NotImplementedError("setup(): 인코더와 머리를 만든다")

    # ③ 전처리 — 관측(NumPy) → 신경망 입력(torch.Tensor). 스케일 제각각·속도 정보 없음 (README 참고)
    def preprocess(self, obs: Observation) -> torch.Tensor:
        raise NotImplementedError("preprocess(): 관측 → 신경망 입력")

    # ④ 행동 선택 — explore=True: 탐색, False: 대결에서 쓰는 행동
    def policy(self, x: torch.Tensor, explore: bool) -> np.ndarray:
        raise NotImplementedError("policy(): 신경망 입력 → 행동")

    # (선택) 기억 초기화 — 에피소드 시작·리스폰·대결 시작 때 done=True 인 배치 원소
    def reset(self, done: np.ndarray) -> None:
        pass

    # ⑤ 보상 — events 필드는 README 참고. 각 (B,)
    def reward(self, events: Events, obs: Observation) -> np.ndarray:
        # TODO: 설계. 아래는 출발점
        return (events.size_after - events.size_before) / 100.0 - 1.0 * events.died + 5.0 * events.won


# ⑥ 학습 루프 — 데이터 흐름과 로그를 직접 짠다
def train(cfg: Config) -> None:
    agent = MyAgent(cfg)
    env = make_env(cfg, agent)
    run = wandb.init(project="cell-arena", name=agent.name, config=cfg.to_dict())
    # TODO: 옵티마이저, 버퍼 (replay / rollout)

    # 기본 로깅 — 필요한 지표 자유롭게 추가
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

        # TODO: 전이 저장 (obs, action, reward, out.final_obs, terminated, truncated — 의미는 README)
        # TODO: 업데이트 (미니배치·타깃 네트워크 / GAE·에폭 등)

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
    parser = argparse.ArgumentParser(description="Cell Arena 학생 에이전트 학습")
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"), help="YAML 설정 파일")
    train(load_config(parser.parse_args().config))
