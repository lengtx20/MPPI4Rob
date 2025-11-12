"""MPPI Pendulum Control Example"""
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.mppi_controller import MPPIController
from models.transition_model import TransitionModel
from models.cost_function import CostFunction
from models.observation import ObservationFunction


class PendulumTransitionModel(TransitionModel):
    """Dynamics model of the pendulum env"""
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
    """Cost function for the pendulum env"""
    def __init__(self, angle_w=1.0, vel_w=0.1, ctrl_w=0.001):
        self.angle_w, self.vel_w, self.ctrl_w = angle_w, vel_w, ctrl_w

    def compute_cost(self, states, actions, **kwargs):
        theta = torch.atan2(states[:, 1], states[:, 0])
        torque = actions[:, 0] if actions.shape[1] > 0 else torch.zeros_like(states[:, 0])
        return self.angle_w * (theta ** 2) + self.vel_w * (states[:, 2] ** 2) + self.ctrl_w * (torque ** 2)


class PendulumObservationFunction(ObservationFunction):
    """Observation function (directly returns state)"""
    def state_to_observation(self, state):
        return state


def main(render=False):
    """Main function"""
    env = gym.make('Pendulum-v1', render_mode='human' if render else None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # MPPI parameters
    mppi = MPPIController(
        transition_model=PendulumTransitionModel(),
        cost_function=PendulumCostFunction(),
        action_dim=1, horizon=15, num_samples=2000,
        action_bounds=(torch.tensor([-2.0], device=device), torch.tensor([2.0], device=device)),
        noise_std=0.8, temperature=0.5, device=device,
        adaptive_temperature=True, noise_annealing=True, use_log_space_weights=True,
    )

    # Run control
    obs, _ = env.reset()
    mppi.reset()
    returns, states, actions = 0, [], []
    
    for step in range(200):
        state_t = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_t, info = mppi.compute_control_with_info(state_t)
        
        obs, reward, done, _, _ = env.step([action_t[0].item()])
        returns += reward
        states.append(obs.copy())
        actions.append(action_t[0].item())
        
        if step % 50 == 0:
            print(f"Step {step}: θ={np.arctan2(obs[1], obs[0]):.3f}, cost={info['min_cost']:.3f}")
        
        if done:
            break
    
    env.close()

    # Plotting
    states = np.array(states)
    actions = np.array(actions)
    angles = np.arctan2(states[:, 1], states[:, 0])
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(angles); axes[0].axhline(0, color='r', ls='--'); axes[0].set_title('Angle')
    axes[1].plot(states[:, 2]); axes[1].axhline(0, color='r', ls='--'); axes[1].set_title('Velocity')
    axes[2].plot(actions); axes[2].axhline(0, color='k', ls='--'); axes[2].set_title('Control')
    plt.tight_layout()
    plt.show()
    
    print(f"Total return: {returns:.2f}")


if __name__ == "__main__":
    render_flag = '--render' in sys.argv
    main(render=render_flag)
