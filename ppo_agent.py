"""PPO 예제. 사용법과 API 는 README.md 참고."""

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
    data = {k: torch.stack([step[k] for step in rollout]) for k in rollout[0]}  # (T, B, ...)

    # GAE
    adv = torch.zeros_like(data["reward"])
    gae, next_value = 0.0, last_value
    for t in reversed(range(len(rollout))):
        nonterminal = 1 - data["done"][t]
        delta = data["reward"][t] + gamma * next_value * nonterminal - data["value"][t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
        next_value = data["value"][t]
    data["adv"] = adv
    data["ret"] = adv + data["value"]
    flat = {k: v.flatten(0, 1) for k, v in data.items()}

    for _ in range(cfg.get("epochs", 4)):
        for idx in torch.randperm(len(flat["adv"])).split(cfg.get("minibatch_size", 1024)):
            b = {k: v[idx].to(agent.device) for k, v in flat.items()}
            dist, value = agent.net(b["x"])
            ratio = (dist.log_prob(b["action"]).sum(1) - b["logp"]).exp()
            adv_b = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
            policy_loss = -torch.min(ratio * adv_b, ratio.clamp(1 - clip, 1 + clip) * adv_b).mean()
            value_loss = 0.5 * (b["ret"] - value).pow(2).mean()
            entropy = dist.entropy().sum(1).mean()
            loss = policy_loss + 0.5 * value_loss - cfg.get("ent_coef", 0.001) * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.net.parameters(), 0.5)
            optimizer.step()
    return {"train/policy_loss": float(policy_loss), "train/value_loss": float(value_loss), "train/entropy": float(entropy)}


# 6. 학습 루프
def train(cfg: Config) -> None:
    agent = PPOAgent(cfg)
    env = make_env(cfg, agent)
    run = wandb.init(project="cell-arena", name=agent.name, config=cfg.to_dict())
    optimizer = torch.optim.Adam(agent.net.parameters(), lr=cfg.get("lr", 3e-4))
    rollout: list[dict[str, torch.Tensor]] = []
    n_steps = cfg.get("n_steps", 128)  # env 당 rollout 길이
    gamma = cfg.get("gamma", 0.99)

    ep_reward = np.zeros(cfg.num_envs)
    recent_returns: list[float] = []
    log_every = 10_000
    next_log = log_every
    stats: dict[str, float] = {}
    start_time = time.time()

    obs = env.reset()
    agent.reset(np.ones(cfg.num_envs, dtype=bool))
    samples = 0
    while samples < cfg.total_samples:
        with torch.no_grad():  # act() 와 같음. logp, value 가 필요해서 나눠서 호출
            x = agent.preprocess(obs)
            dist, value = agent.net(x)
            action = dist.sample()
        out = env.step(action.cpu().numpy())
        reward = agent.reward(out.events, out.final_obs)
        samples += cfg.num_envs

        # max_steps 로 끊긴 env 는 마지막 상태의 가치로 부트스트랩
        r = torch.as_tensor(reward, dtype=torch.float32)
        cut = torch.as_tensor(out.truncated & ~out.terminated)
        if cut.any():
            with torch.no_grad():
                x_final = agent.frames.peek(torch.as_tensor(out.final_obs.image, device=agent.device)).flatten(1, 2)
                r[cut] += gamma * agent.net(x_final)[1].cpu()[cut]

        episode_done = out.terminated | out.truncated
        rollout.append({
            "x": x.cpu(), "action": action.cpu(), "logp": dist.log_prob(action).sum(1).cpu(),
            "value": value.cpu(), "reward": r, "done": torch.as_tensor(episode_done, dtype=torch.float32),
        })

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
                **stats,
            }, step=samples)

        agent.reset(episode_done | out.events.died)
        obs = out.obs

        if len(rollout) == n_steps:
            with torch.no_grad():
                last_value = agent.net(agent.frames.peek(torch.as_tensor(obs.image, device=agent.device)).flatten(1, 2))[1].cpu()
            stats = update(agent, optimizer, rollout, last_value, cfg)
            rollout.clear()

    agent.save()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    train(load_config(parser.parse_args().config))
