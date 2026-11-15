"""Cell Arena PPO 실습 스켈레톤. 사용법과 API 는 README.md 참고.

agent.py 의 PPO 판. 빈칸(작성 영역)을 채워 PPO 를 완성함.
정답 예제는 ppo_agent.py 이며, 같은 TODO 번호로 비교할 수 있음.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
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
from torch.distributions import Normal

import wandb

ACTION_DIM = 3  # continuous 액션 [theta, move, dash]


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


# 모델 작성 영역에서 사용하는 네트워크. 몸통(conv)을 공유하고 머리가 두 개
#   actor : 평균 mu (B, 3) + 학습되는 log_std (3,) -> 정규분포 pi(a|s)
#   critic: 상태 가치 V(s) (B,)
class ActorCritic(nn.Module):
    def __init__(self, in_channels: int, resolution: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        n = self.conv(torch.zeros(1, in_channels, resolution, resolution)).shape[1]
        self.mu = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, ACTION_DIM))
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM))
        self.v = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 1))
        with torch.no_grad():
            self.mu[-1].bias.copy_(torch.tensor([0.0, 1.0, 0.0]))  # move 는 켜진 채로 시작

    def forward(self, x: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        h = self.conv(x.float())
        return Normal(self.mu(h), self.log_std.exp()), self.v(h).squeeze(1)


class MyPPOAgent(StudentAgent):
    # 0. 기본명세. name, weights 는 본인 이름으로
    name = "my_ppo"
    color = (120, 220, 120)  # (R, G, B)
    weights = "my_ppo.pt"  # 대결장은 이 파일을 불러옴

    # 1. 관측 / 액션 형태. PPO 실습은 연속 액션
    obs_spec = ObsSpec(mode="image", resolution=64)
    action_spec = ActionSpec(mode="continuous")
    #   continuous: (B, 3)  float [theta, move, dash]. move, dash 는 0.5 초과면 켜짐

    # 2. 모델. 구조는 self.cfg 만으로 정해져야 함 (load 할 때 다시 호출됨)
    def setup(self) -> None:
        k = self.cfg.get("frame_stack", 4)
        self.frames = FrameStack(k)
        # 입력 (B, k*6, R, R) -> (정규분포, V (B,)) 를 내는 ActorCritic 을 self.net 으로 구성.
        # 입력 채널은 k 프레임 x 6 이미지 채널. 모델은 self.device 에 배치.
        # ==================== 작성 시작: 모델 ====================
        raise NotImplementedError
        # ==================== 작성 끝 ====================

    # 3. 관측(NumPy) -> 신경망 입력. 기본은 이미지 k 프레임 (B, k*6, R, R)
    def preprocess(self, obs: Observation) -> torch.Tensor:
        img = torch.as_tensor(obs.image, device=self.device)
        return self.frames.push(img).flatten(1, 2)

    # 4. 행동 선택. 대결에서는 explore=False
    def policy(self, x: torch.Tensor, explore: bool) -> np.ndarray:
        # self.net(x) 로 정규분포를 얻음. explore=True 면 분포에서 샘플, False 면 평균(mean).
        # 반환: CPU NumPy float 배열 (B, 3). epsilon 같은 별도 탐험은 없음 (분포 자체가 탐험).
        # ==================== 작성 시작: 행동 선택 ====================
        raise NotImplementedError
        # ==================== 작성 끝 ====================

    # 프레임 스택 초기화. done=True 인 원소만
    def reset(self, done: np.ndarray) -> None:
        self.frames.reset(done)

    # 5. 보상
    def reward(self, events: Events, obs: Observation) -> np.ndarray:
        return (events.size_after - events.size_before) / 100.0 - 1.0 * events.died + 5.0 * events.won


# 6-1. PPO 업데이트. rollout 하나(n_steps x num_envs 샘플)로 여러 epoch 학습
def update(
    agent: MyPPOAgent,
    optimizer: torch.optim.Optimizer,
    rollout: list[dict[str, torch.Tensor]],
    last_value: torch.Tensor,
    cfg: Config,
) -> dict[str, float]:
    gamma, lam, clip = cfg.get("gamma", 0.99), cfg.get("gae_lambda", 0.95), cfg.get("clip", 0.2)
    vf_coef, ent_coef = cfg.get("vf_coef", 0.5), cfg.get("ent_coef", 0.001)
    data = {k: torch.stack([step[k] for step in rollout]) for k in rollout[0]}  # (T, B, ...)

    # TODO 4: GAE 로 advantage 와 return 계산
    # 뒤에서부터 (t = T-1 ... 0) 한 칸씩:
    #   delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    #   gae_t   = delta_t + gamma * lam * (1 - done_t) * gae_{t+1}
    # 마지막 칸의 V(s_T) 는 last_value. 결과 adv 는 (T, B), ret = adv + value.
    # done 이 1 인 칸에서는 다음 에피소드의 값이 섞이지 않게 끊어야 함.
    # ==================== 작성 시작: TODO 4 ====================
    adv = torch.zeros_like(data["reward"])

    # ==================== 작성 끝 ====================
    data["adv"] = adv
    data["ret"] = adv + data["value"]
    flat = {k: v.flatten(0, 1) for k, v in data.items()}  # (T*B, ...)

    stats: dict[str, float] = {}
    for _ in range(cfg.get("epochs", 4)):
        for idx in torch.randperm(len(flat["adv"])).split(cfg.get("minibatch_size", 1024)):
            b = {k: v[idx].to(agent.device) for k, v in flat.items()}
            dist, value = agent.net(b["x"])

            # TODO 5: 확률 비율과 clipped surrogate 로 policy loss 계산
            # - new_logp: 저장된 행동의 log 확률을 행동 차원(3)으로 합함 -> (N,)
            # - ratio = exp(new_logp - b["logp"])  (옛 정책 대비 새 정책의 확률 비율)
            # - advantage 는 미니배치 안에서 평균 0, 표준편차 1 로 정규화
            # - surr1 = ratio * A, surr2 = clamp(ratio, 1-clip, 1+clip) * A
            # - policy_loss = -mean(min(surr1, surr2))
            # ==================== 작성 시작: TODO 5 ====================

            # ==================== 작성 끝 ====================

            # TODO 6: value loss + entropy 보너스 -> 총 loss -> 역전파 -> 파라미터 갱신
            # - value_loss = 0.5 * mean((ret - V)^2), entropy 는 행동 차원으로 합한 뒤 평균
            # - loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            # - 기울기 초기화 -> backward -> clip_grad_norm_(0.5) -> step
            # ==================== 작성 시작: TODO 6 ====================

            # ==================== 작성 끝 ====================

            # TODO 8 (1/2): 로그용 숫자 기록. 계산 그래프에서 분리한 float 로 보관.
            # policy_loss, value_loss, entropy 와 함께
            # approx_kl = mean((ratio - 1) - log(ratio)), clip_frac = mean(|ratio - 1| > clip).
            # ==================== 작성 시작: TODO 8 (1/2) ====================

            # ==================== 작성 끝 ====================
    return stats


# 6-2. 학습 루프. 아래 TODO 1~8 이 PPO 실습 순서 (TODO 4~6 은 update 안)
def train(cfg: Config) -> None:
    agent = MyPPOAgent(cfg)
    env = make_env(cfg, agent)
    run = wandb.init(project="cell-arena", name=agent.name, config=cfg.to_dict())
    # TODO 1: 학습 준비
    # - agent.net 의 파라미터를 학습하는 옵티마이저 (Adam, lr 은 cfg 에서).
    # - rollout 은 스텝마다 dict 하나를 쌓는 리스트. 리플레이 버퍼, 타깃 네트워크는 없음.
    # - n_steps (env 당 rollout 길이), gamma 를 cfg 에서 읽기.
    # - 첫 업데이트 전 로그용 stats 는 빈 dict.
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
        # TODO 2: 현재 입력 x, 행동, 그 행동의 log 확률, 가치 확보
        # 아래 act 호출을 no_grad 안의 preprocess(obs) -> agent.net(x) 로 대체.
        # dist.sample() 로 행동 (B, 3), log 확률은 행동 차원으로 합해 (B,), value 는 (B,).
        # act 와 preprocess 를 둘 다 호출하면 프레임이 두 번 쌓임.
        # ==================== 작성 시작: TODO 2 ====================
        action = agent.act(obs, explore=True)
        # ==================== 작성 끝 ====================

        out = env.step(torch.as_tensor(action).cpu().numpy())  # 텐서·NumPy 모두 허용
        reward = agent.reward(out.events, out.final_obs)
        samples += cfg.num_envs
        episode_done = out.terminated | out.truncated

        # TODO 3: 보상 보정과 rollout 저장
        # - max_steps 로 끊긴 env (truncated & ~terminated) 는 진짜 종료가 아니므로
        #   r += gamma * V(final_obs) 로 부트스트랩. V 는 frames.peek(final_obs.image) 로 구함.
        # - (x, action, logp, value, reward, done) 을 CPU 텐서로 rollout 에 append.
        #   done = terminated | truncated (float32). died 는 done 이 아님.
        # ==================== 작성 시작: TODO 3 ====================

        # ==================== 작성 끝 ====================

        ep_reward += reward
        recent_returns.extend(ep_reward[episode_done].tolist())
        recent_returns[:-200] = []
        ep_reward[episode_done] = 0.0

        if samples >= next_log:
            next_log += log_every
            # TODO 8 (2/2): 아래 로그에 update 가 돌려준 stats 추가.
            # train/policy_loss, train/value_loss, train/entropy, train/approx_kl, train/clip_frac
            run.log({
                "train/reward_step": float(reward.mean()),
                "train/episode_return": float(np.mean(recent_returns)) if recent_returns else float("nan"),
                "train/size": float(out.final_obs.self_state[:, 0].mean()),
                "train/deaths": int(out.events.died.sum()),
                "train/samples_per_sec": samples / (time.time() - start_time),
                # ==================== 작성 시작: TODO 8 (2/2) ====================

                # ==================== 작성 끝 ====================
            }, step=samples)

        agent.reset(episode_done | out.events.died)
        obs = out.obs

        # TODO 7: rollout 이 n_steps 만큼 차면 업데이트
        # - 마지막 다음 상태의 가치 last_value = V(frames.peek(obs.image)) 를 no_grad 로 계산 (B,).
        #   (reset 이후이므로 새 에피소드 첫 관측으로 채워짐)
        # - update(...) 결과를 stats 에 저장하고 rollout 을 비움 (on-policy: 다 쓰면 버림).
        # ==================== 작성 시작: TODO 7 ====================

        # ==================== 작성 끝 ====================

    agent.save()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    train(load_config(parser.parse_args().config))
