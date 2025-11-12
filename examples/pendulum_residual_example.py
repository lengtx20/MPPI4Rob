"""MPPI Pendulum Residual Control Example

This demo shows a simple rule-based baseline (PD controller) used as a
baseline policy and MPPI operating on residuals around that baseline.
"""
import torch
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.mppi_controller import MPPIController
from models.transition_model import TransitionModel
from models.cost_function import CostFunction
from models.baseline_policy import BaselinePolicy
from models.observation import ObservationFunction


class PendulumTransitionModel(TransitionModel):
    """Pendulum dynamics (same compact model as other example)."""
    def __init__(self, dt=0.05, g=10.0, m=1.0, l=1.0, max_speed=8.0):
        self.dt, self.g, self.m, self.l, self.max_speed = dt, g, m, l, max_speed

    def forward(self, states, actions, **kwargs):
        cos_theta, sin_theta, theta_dot = states[:, 0], states[:, 1], states[:, 2]
        torque = torch.clamp(actions[:, 0], -2.0, 2.0)
        theta_ddot = 3 * self.g / (2 * self.l) * sin_theta + 3 / (self.m * self.l ** 2) * torque
        new_theta_dot = torch.clamp(theta_dot + theta_ddot * self.dt, -self.max_speed, self.max_speed)
        theta = torch.atan2(sin_theta, cos_theta)
        new_theta = torch.atan2(torch.sin(theta + new_theta_dot * self.dt), torch.cos(theta + new_theta_dot * self.dt))
        return torch.stack([torch.cos(new_theta), torch.sin(new_theta), new_theta_dot], dim=1)


class PendulumCostFunction(CostFunction):
    """Quadratic cost to keep pendulum upright with small control penalty."""
    def __init__(self, angle_w=1.0, vel_w=0.1, ctrl_w=0.001):
        self.angle_w, self.vel_w, self.ctrl_w = angle_w, vel_w, ctrl_w

    def compute_cost(self, states, actions, **kwargs):
        theta = torch.atan2(states[:, 1], states[:, 0])
        torque = actions[:, 0] if actions.shape[1] > 0 else torch.zeros_like(states[:, 0])
        return self.angle_w * (theta ** 2) + self.vel_w * (states[:, 2] ** 2) + self.ctrl_w * (torque ** 2)


class PendulumObservationFunction(ObservationFunction):
    """Return observations directly from state."""
    def state_to_observation(self, state):
        return state


def pd_baseline_policy_fn(observation: torch.Tensor, kp: float = 10.0, kd: float = 2.0) -> torch.Tensor:
    """A simple PD controller implemented as a function.

    Args:
        observation: tensor shape (batch, 3) containing [cos(theta), sin(theta), theta_dot]
    Returns:
        actions: tensor shape (batch, 1)
    """
    # observation may be (batch, 3)
    if observation.dim() == 1:
        observation = observation.unsqueeze(0)
    theta = torch.atan2(observation[:, 1], observation[:, 0])
    theta_dot = observation[:, 2]
    torque = -(kp * theta + kd * theta_dot)
    return torque.unsqueeze(-1).clamp(-2.0, 2.0)


class FunctionBaselinePolicy(BaselinePolicy):
    """Wrap a function to match BaselinePolicy interface."""
    def __init__(self, func):
        self.func = func

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.func(observation)


def main(render: bool = False, mode: str = None):
    """Run MPPI controlling residuals around a PD baseline.

    mode controls what action the controller returns:
      - 'combined' -> baseline + residual (default when residuals enabled)
      - 'residual' -> residual only
      - 'baseline' -> baseline only
    """
    env = gym.make('Pendulum-v1', render_mode='human' if render else None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    transition = PendulumTransitionModel()
    cost = PendulumCostFunction()
    obs_fn = PendulumObservationFunction()

    baseline = FunctionBaselinePolicy(pd_baseline_policy_fn)

    # Map CLI mode alias to controller control_mode
    ctrl_mode = None
    if mode is not None:
        # allow user to pass 'combined' as an alias
        ctrl_mode = 'residual+baseline' if mode == 'combined' else mode

    mppi = MPPIController(
        transition_model=transition,
        cost_function=cost,
        action_dim=1,
        horizon=15,
        num_samples=800,
        action_bounds=(torch.tensor([-2.0], device=device), torch.tensor([2.0], device=device)),
        noise_std=0.6,
        temperature=0.7,
        device=device,
        use_residual_dynamics=True,
        baseline_policy=baseline,
        observation_function=obs_fn,
        control_mode=ctrl_mode,
        adaptive_temperature=True,
        noise_annealing=True,
        use_log_space_weights=True,
    )

    obs, _ = env.reset()
    mppi.reset()
    returns, states, actions = 0, [], []

    for step in range(200):
        state_t = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_t, info = mppi.compute_control_with_info(state_t)

        # action_t may be residual, baseline or full action depending on mode
        # extract numeric scalar to pass to env
        if isinstance(action_t, torch.Tensor):
            if action_t.dim() == 0:
                action_val = float(action_t.item())
            else:
                action_val = float(action_t[0].item())
        else:
            action_val = float(action_t)

        obs, reward, done, _, _ = env.step([action_val])
        returns += reward
        states.append(obs.copy())
        actions.append(action_val)

        if step % 50 == 0:
            print(f"Step {step}: θ={np.arctan2(obs[1], obs[0]):.3f}, min_cost={info['min_cost']:.3f}")

        if done:
            break

    env.close()

    # Plot
    states = np.array(states)
    actions = np.array(actions)
    angles = np.arctan2(states[:, 1], states[:, 0])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(angles); axes[0].axhline(0, color='r', ls='--'); axes[0].set_title('Angle')
    axes[1].plot(states[:, 2]); axes[1].axhline(0, color='r', ls='--'); axes[1].set_title('Velocity')
    axes[2].plot(actions); axes[2].axhline(0, color='k', ls='--'); axes[2].set_title('Residual Control')
    plt.tight_layout()
    plt.show()

    print(f"Total return: {returns:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pendulum residual MPPI example')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--mode', choices=['baseline', 'residual', 'combined'], default=None,
                        help="control mode: 'baseline'|'residual'|'combined' (combined = baseline+residual)")
    args = parser.parse_args()
    main(render=args.render, mode=args.mode)
