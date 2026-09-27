"""PPO 예제. 이미지 관측 + 연속 행동. 가장 기초적인 형태만 구현했다.

    python ppo_agent.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.distributions import Normal

from cell_arena import ActionSpec, Config, Events, Observation, ObsSpec, StudentAgent, load_config, make_env


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


class ActorCritic(nn.Module):
    """정책: [θ, move, dash] 의 가우시안 (평균은 신경망, 표준편차는 학습 파라미터). 가치: V(s)."""

    def __init__(self, in_channels: int, resolution: int) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, resolution)
        self.mu = nn.Linear(256, 3)
        self.log_std = nn.Parameter(torch.zeros(3))
        self.v = nn.Linear(256, 1)
        with torch.no_grad():
            self.mu.bias.copy_(torch.tensor([0.0, 1.0, 0.0]))  # 처음엔 주로 움직이고 가끔 돌진

    def forward(self, img: torch.Tensor, vec: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        h = self.encoder(img, vec)
        return Normal(self.mu(h), self.log_std.exp()), self.v(h).squeeze(1)


class PPOAgent(StudentAgent):
    name = "ppo"
    color = (120, 220, 120)
    weights = "ppo.pt"

    obs_spec = ObsSpec(mode="image", resolution=64)
    action_spec = ActionSpec(mode="continuous")

    def setup(self) -> None:
        k = self.cfg.get("frame_stack", 4)
        self.frames = FrameStack(k)
        self.net = ActorCritic(k * 5, self.obs_spec.resolution).to(self.device)

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

    # 탐색이면 샘플, 대결이면 평균
    def policy(self, x: tuple[torch.Tensor, torch.Tensor], explore: bool) -> np.ndarray:
        dist, _ = self.net(*x)
        action = dist.sample() if explore else dist.mean
        return action.cpu().numpy()

    def reset(self, done: np.ndarray) -> None:
        self.frames.reset(done)

    def reward(self, events: Events, obs: Observation) -> np.ndarray:
        return (events.size_after - events.size_before) / 100.0 - 1.0 * events.died + 5.0 * events.won


def train(cfg: Config) -> None:
    agent = PPOAgent(cfg)
    env = make_env(cfg, agent)
    run = wandb.init(project="cell-arena", name=agent.name, config=cfg.to_dict())

    gamma = cfg.get("gamma", 0.99)
    lam = cfg.get("gae_lambda", 0.95)
    n_steps = cfg.get("n_steps", 128)  # env 당 rollout 길이
    epochs = cfg.get("epochs", 4)
    minibatch_size = cfg.get("minibatch_size", 1024)
    clip = cfg.get("clip", 0.2)
    ent_coef = cfg.get("ent_coef", 0.001)
    optimizer = torch.optim.Adam(agent.net.parameters(), lr=cfg.get("lr", 3e-4))

    ep_reward = np.zeros(cfg.num_envs)
    recent_returns: list[float] = []
    log_every = 10_000
    next_log = log_every
    start_time = time.time()

    obs = env.reset()
    agent.reset(np.ones(cfg.num_envs, dtype=bool))
    samples = 0
    while samples < cfg.total_samples:
        # 1) rollout 수집
        roll: dict[str, list[torch.Tensor]] = {k: [] for k in ("img", "vec", "action", "logp", "value", "reward", "done")}
        for _ in range(n_steps):
            with torch.no_grad():
                x = agent.preprocess(obs)
                dist, value = agent.net(*x)
                action = dist.sample()
            out = env.step(action.cpu().numpy())
            reward = agent.reward(out.events, out.final_obs)
            samples += cfg.num_envs

            # max_steps 로 잘린 env 는 끝난 게 아니므로 마지막 상태의 가치로 부트스트랩한다
            r = torch.as_tensor(reward, dtype=torch.float32)
            cut = torch.as_tensor(out.truncated & ~out.terminated)
            if cut.any():
                with torch.no_grad():
                    _, v_final = agent.net(*agent.peek(out.final_obs))
                r[cut] += gamma * v_final.cpu()[cut]
            episode_done = out.terminated | out.truncated

            roll["img"].append(x[0].cpu())
            roll["vec"].append(x[1].cpu())
            roll["action"].append(action.cpu())
            roll["logp"].append(dist.log_prob(action).sum(1).cpu())
            roll["value"].append(value.cpu())
            roll["reward"].append(r)
            roll["done"].append(torch.as_tensor(episode_done, dtype=torch.float32))

            ep_reward += reward
            recent_returns.extend(ep_reward[episode_done].tolist())
            recent_returns[:-200] = []
            ep_reward[episode_done] = 0.0

            agent.reset(episode_done | out.events.died)
            obs = out.obs

        # 2) GAE
        with torch.no_grad():
            _, last_value = agent.net(*agent.peek(obs))
        data = {k: torch.stack(v) for k, v in roll.items()}  # (T, B, ...)
        adv = torch.zeros_like(data["reward"])
        gae = torch.zeros(cfg.num_envs)
        next_value = last_value.cpu()
        for t in reversed(range(n_steps)):
            nonterminal = 1 - data["done"][t]
            delta = data["reward"][t] + gamma * next_value * nonterminal - data["value"][t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
            next_value = data["value"][t]
        data["adv"] = adv
        data["ret"] = adv + data["value"]
        flat = {k: v.flatten(0, 1) for k, v in data.items()}  # (T·B, ...)

        # 3) 업데이트
        n = len(flat["adv"])
        for _ in range(epochs):
            for idx in torch.randperm(n).split(minibatch_size):
                b = {k: v[idx].to(agent.device) for k, v in flat.items()}
                dist, value = agent.net(b["img"], b["vec"])
                ratio = (dist.log_prob(b["action"]).sum(1) - b["logp"]).exp()
                adv_b = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
                policy_loss = -torch.min(ratio * adv_b, ratio.clamp(1 - clip, 1 + clip) * adv_b).mean()
                value_loss = 0.5 * (b["ret"] - value).pow(2).mean()
                entropy = dist.entropy().sum(1).mean()
                loss = policy_loss + 0.5 * value_loss - ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.net.parameters(), 0.5)
                optimizer.step()

        if samples >= next_log:
            next_log = samples + log_every
            run.log({
                "train/reward_step": float(data["reward"].mean()),
                "train/episode_return": float(np.mean(recent_returns)) if recent_returns else float("nan"),
                "train/size": float(obs.self_state[:, 0].mean()),
                "train/samples_per_sec": samples / (time.time() - start_time),
                "train/policy_loss": float(policy_loss),
                "train/value_loss": float(value_loss),
                "train/entropy": float(entropy),
            }, step=samples)

    agent.save()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    train(load_config(parser.parse_args().config))
