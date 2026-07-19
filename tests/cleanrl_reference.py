"""Faithful transcriptions of CleanRL's update math — the independent
reference for the exact-reduction tests (D10).

Provenance: fetched 2026-07-19 from vwxyzjn/cleanrl @ master:
- sac_continuous_action.py: get_action L139-151, critic target L261-268,
  actor loss L283
- td3_continuous_action.py: critic target L232-247, actor loss L256
- ppo_continuous_action.py: GAE L232-246, pg_loss L279-282, v_loss L296-297

Conventions: action_scale = 1, action_bias = 0 (RescaleAction to [-1, 1]);
(batch,) shapes, so CleanRL's .view(-1)/.flatten() calls are dropped. The
loss *assembly* below is theirs line by line; only the network forwards are
shared with our implementation (the equivalence claim targets the objective
math, not weight-file compatibility across different sigma parametrizations).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


# ---- SAC ----

def sac_get_action(mean, std):
    normal = torch.distributions.Normal(mean, std)
    x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    y_t = torch.tanh(x_t)
    action = y_t
    log_prob = normal.log_prob(x_t)
    # Enforcing Action Bound
    log_prob = log_prob - torch.log(1 * (1 - y_t.pow(2)) + 1e-6)
    return action, log_prob.sum(-1)


def sac_critic_target(mean_next, std_next, qf1_target, qf2_target,
                      next_obs, rewards, dones, gamma, alpha):
    next_state_actions, next_state_log_pi = sac_get_action(mean_next, std_next)
    qf1_next_target = qf1_target(next_obs, next_state_actions)
    qf2_next_target = qf2_target(next_obs, next_state_actions)
    min_qf_next_target = (
        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
    )
    return rewards + (1 - dones) * gamma * min_qf_next_target


def sac_critic_loss(qf1, qf2, obs, actions, next_q_value):
    qf1_a_values = qf1(obs, actions)
    qf2_a_values = qf2(obs, actions)
    return F.mse_loss(qf1_a_values, next_q_value) + F.mse_loss(qf2_a_values, next_q_value)


def sac_actor_loss(mean, std, qf1, qf2, obs, alpha):
    pi, log_pi = sac_get_action(mean, std)
    qf1_pi = qf1(obs, pi)
    qf2_pi = qf2(obs, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    return ((alpha * log_pi) - min_qf_pi).mean()


# ---- TD3 ----

def td3_critic_target(target_mu_next, qf1_target, qf2_target, next_obs,
                      actions_like, rewards, dones, gamma, policy_noise, noise_clip):
    clipped_noise = (torch.randn_like(actions_like) * policy_noise).clamp(
        -noise_clip, noise_clip
    ) * 1.0
    next_state_actions = (target_mu_next + clipped_noise).clamp(-1.0, 1.0)
    qf1_next_target = qf1_target(next_obs, next_state_actions)
    qf2_next_target = qf2_target(next_obs, next_state_actions)
    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
    return rewards + (1 - dones) * gamma * min_qf_next_target


def td3_actor_loss_qf1(qf1, obs, tanh_mu):
    """CleanRL's literal actor loss: Q1 only (the historical accident)."""
    return -qf1(obs, tanh_mu).mean()


def td3_actor_loss_min(qf1, qf2, obs, tanh_mu):
    """The D2 substitution: min in the actor, our unified form."""
    return -torch.min(qf1(obs, tanh_mu), qf2(obs, tanh_mu)).mean()


# ---- PPO ----

def ppo_gae(rewards, values, next_value, gamma, gae_lambda):
    """No-done fixture (the fixed batches contain no truncation events, D3)."""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0
    for t in reversed(range(num_steps)):
        nextvalues = next_value if t == num_steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextvalues - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * lastgaelam
    returns = advantages + values
    return advantages, returns


def ppo_pg_loss(newlogprob, oldlogprob, advantages, clip_coef):
    logratio = newlogprob - oldlogprob
    ratio = logratio.exp()
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    return torch.max(pg_loss1, pg_loss2).mean()


def ppo_v_loss_unclipped(newvalue, returns):
    """CleanRL's clip_vloss=False branch; the default clipped variant is a
    CleanRL-specific detail outside the unified objective (D8)."""
    return 0.5 * ((newvalue - returns) ** 2).mean()
