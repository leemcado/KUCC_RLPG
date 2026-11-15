"""PPO 예제. 사용법과 API 는 README.md 참고."""

# ppo_skeleton.py 의 작성 영역을 채운 예제. TODO 4~6, 8 (1/2) 는 update() 에 구현.

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.distributions import Normal

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
        self.mu = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 3))
        self.log_std = nn.Parameter(torch.zeros(3))
        self.v = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 1))
        with torch.no_grad():
            self.mu[-1].bias.copy_(torch.tensor([0.0, 1.0, 0.0]))  # move 는 켜진 채로 시작

    def forward(self, x: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        h = self.conv(x.float())
        return Normal(self.mu(h), self.log_std.exp()), self.v(h).squeeze(1)


class PPOAgent(StudentAgent):
    # 0. 기본명세
    name = "ppo"
    color = (120, 220, 120)  # (R, G, B)
    weights = "ppo.pt"

    # 1. 관측 / 액션 형태
    obs_spec = ObsSpec(mode="image", resolution=64)
    action_spec = ActionSpec(mode="continuous")

    # 2. 모델
    def setup(self) -> None:
        k = self.cfg.get("frame_stack", 4)
        self.frames = FrameStack(k)
        self.net = ActorCritic(k * len(IMAGE_CHANNELS), self.obs_spec.resolution).to(self.device)

    # 3. 관측(NumPy) -> 신경망 입력. 이미지 k 프레임 (B, k*6, R, R)
    def preprocess(self, obs: Observation) -> torch.Tensor:
        img = torch.as_tensor(obs.image, device=self.device)
        return self.frames.push(img).flatten(1, 2)

    # 4. 행동 선택. 탐색은 샘플, 대결은 평균
    def policy(self, x: torch.Tensor, explore: bool) -> np.ndarray:
        dist, _ = self.net(x)
        action = dist.sample() if explore else dist.mean
        return action.cpu().numpy()

    # 프레임 스택 초기화. done=True 인 원소만
    def reset(self, done: np.ndarray) -> None:
        self.frames.reset(done)

    # 5. 보상
    def reward(self, events: Events, obs: Observation) -> np.ndarray:
        return (events.size_after - events.size_before) / 100.0 - 1.0 * events.died + 5.0 * events.won


def update(agent: PPOAgent, optimizer: torch.optim.Optimizer, rollout: list[dict[str, torch.Tensor]], last_value: torch.Tensor, cfg: Config) -> dict[str, float]:
    gamma, lam, clip = cfg.get("gamma", 0.99), cfg.get("gae_lambda", 0.95), cfg.get("clip", 0.2)
    vf_coef, ent_coef = cfg.get("vf_coef", 0.5), cfg.get("ent_coef", 0.001)
    data = {k: torch.stack([step[k] for step in rollout]) for k in rollout[0]}  # (T, B, ...)

    # TODO 4: GAE 로 advantage 와 return 계산
    # ==================== 작성 시작: TODO 4 ====================
    adv = torch.zeros_like(data["reward"])
    gae, next_value = 0.0, last_value
    for t in reversed(range(len(rollout))):
        nonterminal = 1 - data["done"][t]
        delta = data["reward"][t] + gamma * next_value * nonterminal - data["value"][t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
        next_value = data["value"][t]
    # ==================== 작성 끝 ====================
    data["adv"] = adv
    data["ret"] = adv + data["value"]
    flat = {k: v.flatten(0, 1) for k, v in data.items()}

    stats: dict[str, float] = {}
    for _ in range(cfg.get("epochs", 4)):
        for idx in torch.randperm(len(flat["adv"])).split(cfg.get("minibatch_size", 1024)):
            b = {k: v[idx].to(agent.device) for k, v in flat.items()}
            dist, value = agent.net(b["x"])

            # TODO 5: 확률 비율과 clipped surrogate
            # ==================== 작성 시작: TODO 5 ====================
            new_logp = dist.log_prob(b["action"]).sum(1)
            ratio = (new_logp - b["logp"]).exp()
            adv_b = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
            surr1 = ratio * adv_b
            surr2 = ratio.clamp(1 - clip, 1 + clip) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            # ==================== 작성 끝 ====================

            # TODO 6: value loss + entropy -> 총 loss -> 역전파 -> 갱신
            # ==================== 작성 시작: TODO 6 ====================
            value_loss = 0.5 * (b["ret"] - value).pow(2).mean()
            entropy = dist.entropy().sum(1).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.net.parameters(), 0.5)
            optimizer.step()
            # ==================== 작성 끝 ====================

            # TODO 8 (1/2): 로그용 숫자 기록
            # ==================== 작성 시작: TODO 8 (1/2) ====================
            with torch.no_grad():
                log_ratio = new_logp - b["logp"]
                stats = {
                    "train/policy_loss": float(policy_loss),
                    "train/value_loss": float(value_loss),
                    "train/entropy": float(entropy),
                    "train/approx_kl": float(((ratio - 1) - log_ratio).mean()),
                    "train/clip_frac": float(((ratio - 1).abs() > clip).float().mean()),
                }
            # ==================== 작성 끝 ====================
    return stats


# 6. 학습 루프. TODO 번호는 ppo_skeleton.py 의 TODO 1~8 에 대응.
def train(cfg: Config) -> None:
    agent = PPOAgent(cfg)
    env = make_env(cfg, agent)
    run = wandb.init(project="cell-arena", name=agent.name, config=cfg.to_dict())
    # TODO 1: 학습 준비
    # ==================== 작성 시작: TODO 1 ====================
    optimizer = torch.optim.Adam(agent.net.parameters(), lr=cfg.get("lr", 3e-4))
    rollout: list[dict[str, torch.Tensor]] = []
    n_steps = cfg.get("n_steps", 128)  # env 당 rollout 길이
    gamma = cfg.get("gamma", 0.99)
    stats: dict[str, float] = {}
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
        # TODO 2: 현재 입력 x, 행동, log 확률, 가치 확보
        # ==================== 작성 시작: TODO 2 ====================
        with torch.no_grad():  # act() 와 같음. logp, value 가 필요해서 나눠서 호출
            x = agent.preprocess(obs)
            dist, value = agent.net(x)
            action = dist.sample()
            logp = dist.log_prob(action).sum(1)
        # ==================== 작성 끝 ====================

        out = env.step(action.cpu().numpy())
        reward = agent.reward(out.events, out.final_obs)
        samples += cfg.num_envs
        episode_done = out.terminated | out.truncated

        # TODO 3: 보상 보정과 rollout 저장
        # ==================== 작성 시작: TODO 3 ====================
        # max_steps 로 끊긴 env 는 마지막 상태의 가치로 부트스트랩
        r = torch.as_tensor(reward, dtype=torch.float32)
        cut = torch.as_tensor(out.truncated & ~out.terminated)
        if cut.any():
            with torch.no_grad():
                x_final = agent.frames.peek(torch.as_tensor(out.final_obs.image, device=agent.device)).flatten(1, 2)
                r[cut] += gamma * agent.net(x_final)[1].cpu()[cut]
        rollout.append({
            "x": x.cpu(), "action": action.cpu(), "logp": logp.cpu(),
            "value": value.cpu(), "reward": r, "done": torch.as_tensor(episode_done, dtype=torch.float32),
        })
        # ==================== 작성 끝 ====================

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
                # TODO 8 (2/2): update 가 돌려준 stats
                # ==================== 작성 시작: TODO 8 (2/2) ====================
                **stats,
                # ==================== 작성 끝 ====================
            }, step=samples)

        agent.reset(episode_done | out.events.died)
        obs = out.obs

        # TODO 7: rollout 이 n_steps 만큼 차면 업데이트
        # ==================== 작성 시작: TODO 7 ====================
        if len(rollout) == n_steps:
            with torch.no_grad():
                last_value = agent.net(agent.frames.peek(torch.as_tensor(obs.image, device=agent.device)).flatten(1, 2))[1].cpu()
            stats = update(agent, optimizer, rollout, last_value, cfg)
            rollout.clear()
        # ==================== 작성 끝 ====================

    agent.save()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    train(load_config(parser.parse_args().config))
