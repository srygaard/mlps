# Source: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py
# Extracted: Agent architecture + PPO update kernel, driven with synthetic observations.
# Gymnasium, tyro, tensorboard, and cleanrl_utils are not needed for this benchmark.
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mlps_shared import affinity
from result import Metric, WorkloadResult
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Conv2d, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def ppo_update(
    agent: Agent,
    optimizer: optim.Optimizer,
    b_obs: torch.Tensor,
    b_actions: torch.Tensor,
    b_logprobs: torch.Tensor,
    b_advantages: torch.Tensor,
    b_returns: torch.Tensor,
    b_values: torch.Tensor,
    clip_coef: float = 0.1,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> None:
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
        b_obs, b_actions.long()
    )
    logratio = newlogprob - b_logprobs
    ratio = logratio.exp()

    mb_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
    pg_loss = torch.max(
        -mb_advantages * ratio,
        -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef),
    ).mean()

    newvalue = newvalue.view(-1)
    v_clipped = b_values + torch.clamp(newvalue - b_values, -clip_coef, clip_coef)
    v_loss = (
        0.5
        * torch.max((newvalue - b_returns) ** 2, (v_clipped - b_returns) ** 2).mean()
    )

    loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    optimizer.step()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--result-file", type=str, default=None)
    parser.add_argument(
        "--compile", action="store_true", help="Use torch.compile for PPO update"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    torch.manual_seed(42)

    n_actions = 18  # Breakout action space
    batch_size = 256

    agent = Agent(n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    rng = torch.Generator(device=device).manual_seed(42)
    b_obs = torch.randint(
        0,
        256,
        (batch_size, 4, 84, 84),
        generator=rng,
        dtype=torch.float32,
        device=device,
    )
    b_actions = torch.randint(
        0,
        n_actions,
        (batch_size,),
        generator=rng,
        device=device,
    )
    b_logprobs = torch.randn(batch_size, generator=rng, device=device)
    b_advantages = torch.randn(batch_size, generator=rng, device=device)
    b_returns = torch.randn(batch_size, generator=rng, device=device)
    b_values = torch.randn(batch_size, generator=rng, device=device)
    if args.compile:
        _ppo_update = torch.compile(ppo_update)  # type: ignore
    else:
        _ppo_update = ppo_update

    for _ in range(args.warmup):
        _ppo_update(
            agent,
            optimizer,
            b_obs,
            b_actions,
            b_logprobs,
            b_advantages,
            b_returns,
            b_values,
        )

    start = time.perf_counter()
    deadline = start + args.duration
    steps = 0
    while time.perf_counter() < deadline:
        _ppo_update(
            agent,
            optimizer,
            b_obs,
            b_actions,
            b_logprobs,
            b_advantages,
            b_returns,
            b_values,
        )
        steps += 1

    elapsed = time.perf_counter() - start
    sps = steps / elapsed if elapsed > 0 else 0.0
    print(f"Average steps per second: {sps:.4f}", flush=True)

    if args.result_file:
        result_path = Path(args.result_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        res = WorkloadResult(
            workload=Path(__file__).stem,
            duration=args.duration,
            metrics=[Metric(name="Steps per second", value=sps, unit="steps/s")],
        )
        result_path.write_text(res.model_dump_json())


if __name__ == "__main__":
    affinity.pin_to_allocated_cpus()
    affinity.set_localalloc()
    torch.backends.cudnn.benchmark = True
    main(parse_args())
