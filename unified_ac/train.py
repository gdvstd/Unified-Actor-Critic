"""The environment loop: collect per data regime, update, evaluate.

Actions are policy-space [-1, 1]; RescaleAction maps them to the env's
bounds. Single-environment v1 — vectorized collection is backlog.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import gymnasium as gym
import torch

from unified_ac.agent import UnifiedActorCritic
from unified_ac.buffers import ReplayBuffer, RolloutBuffer
from unified_ac.config import UnifiedConfig


class TrainResult(NamedTuple):
    agent: UnifiedActorCritic
    final_return: float
    history: list[tuple[int, float]]  # (env_step, eval_return)


def make_env(env_id: str, seed: int | None = None) -> gym.Env:
    env = gym.wrappers.RescaleAction(gym.make(env_id), -1.0, 1.0)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def _to_tensor(obs) -> torch.Tensor:
    return torch.as_tensor(obs, dtype=torch.float32)


@torch.no_grad()
def evaluate(env: gym.Env, agent: UnifiedActorCritic, episodes: int = 5) -> float:
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.act(_to_tensor(obs).unsqueeze(0), deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).numpy())
            total += float(reward)
            done = terminated or truncated
    return total / episodes


def train_replay(
    env: gym.Env,
    agent: UnifiedActorCritic,
    total_steps: int,
    learning_starts: int = 1000,
    batch_size: int = 256,
    buffer_capacity: int = 100_000,
    step_hook: Callable[[int], None] | None = None,
) -> ReplayBuffer:
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buffer = ReplayBuffer(buffer_capacity, obs_dim, act_dim)

    obs, _ = env.reset()
    for step in range(total_steps):
        obs_t = _to_tensor(obs)
        if step < learning_starts:
            action = _to_tensor(env.action_space.sample())
        else:
            action = agent.act(obs_t.unsqueeze(0)).squeeze(0)
        next_obs, reward, terminated, truncated, _ = env.step(action.numpy())
        # D3: store `terminated` only; next_obs is the real successor
        buffer.add(obs_t, action, float(reward), _to_tensor(next_obs), terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
        if step >= learning_starts:
            agent.update_replay(buffer.sample(batch_size))
        if step_hook is not None:
            step_hook(step + 1)
    return buffer


def train_rollout(
    env: gym.Env,
    agent: UnifiedActorCritic,
    iterations: int,
    rollout_length: int = 2048,
    epochs: int = 10,
    minibatch_size: int = 64,
    normalize_adv: bool = False,
    step_hook: Callable[[int], None] | None = None,
) -> RolloutBuffer:
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buffer = RolloutBuffer(rollout_length, obs_dim, act_dim)

    obs, _ = env.reset()
    step = 0
    for _ in range(iterations):
        buffer.clear()
        while not buffer.full:
            obs_t = _to_tensor(obs)
            action, log_prob = agent.act_with_log_prob(obs_t.unsqueeze(0))
            action = action.squeeze(0)
            next_obs, reward, terminated, truncated, _ = env.step(action.numpy())
            # the rollout end is a truncation unless terminated (D3);
            # lambda_return treats the boundary that way on its own
            buffer.add(
                obs_t, action, float(reward), _to_tensor(next_obs),
                terminated, truncated, float(log_prob.squeeze(0)),
            )
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
            step += 1
            if step_hook is not None:
                step_hook(step)
        agent.update_rollout(
            buffer, epochs=epochs, minibatch_size=minibatch_size,
            normalize_adv=normalize_adv,
        )
    return buffer


def train(
    env_id: str,
    cfg: UnifiedConfig,
    total_steps: int,
    seed: int = 0,
    hidden: tuple[int, ...] = (256, 256),
    eval_every: int | None = None,
    eval_episodes: int = 5,
    **kwargs,
) -> TrainResult:
    """Convenience entry: build env + agent, train per regime, evaluate.

    eval_every records (step, eval_return) into the history — the learning
    curve the benchmark harness consumes.
    """
    torch.manual_seed(seed)
    env = make_env(env_id, seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = UnifiedActorCritic(cfg, obs_dim, act_dim, hidden=hidden)

    history: list[tuple[int, float]] = []
    step_hook = None
    if eval_every is not None:
        eval_env = make_env(env_id, seed + 999)

        def step_hook(step: int) -> None:
            if step % eval_every == 0:
                history.append((step, evaluate(eval_env, agent, eval_episodes)))

    if cfg.data == "replay":
        train_replay(env, agent, total_steps, step_hook=step_hook, **kwargs)
    else:
        rollout_length = kwargs.pop("rollout_length", 2048)
        iterations = max(1, total_steps // rollout_length)
        train_rollout(
            env, agent, iterations, rollout_length=rollout_length,
            step_hook=step_hook, **kwargs,
        )

    final = evaluate(env, agent, eval_episodes)
    history.append((total_steps, final))
    return TrainResult(agent, final, history)
