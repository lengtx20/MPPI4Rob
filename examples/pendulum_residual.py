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
from models.baseline_policy import BaselinePolicy
from models.observation import ObservationFunction

class PendulumTransitionModel(TransitionModel):
    def __init__(self, dt=0.05, g=10.0, m=1.0, l=1.0, max_speed=8.0):
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l
        self.max_speed = max_speed

    def forward(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        cos_theta = states[:, 0]
        sin_theta = states[:, 1]
        theta_dot = states[:, 2]
        torque = torch.clamp(actions[:, 0], -2.0, 2.0)
        theta_ddot = 3 * self.g / (2 * self.l) * sin_theta + 3 / (self.m * self.l**2) * torque
        new_theta_dot = torch.clamp(theta_dot + theta_ddot * self.dt, -self.max_speed, self.max_speed)
        theta = torch.atan2(sin_theta, cos_theta)
        new_theta = torch.atan2(torch.sin(theta + new_theta_dot * self.dt), torch.cos(theta + new_theta_dot * self.dt))
        new_cos_theta = torch.cos(new_theta)
        new_sin_theta = torch.sin(new_theta)
        return torch.stack([new_cos_theta, new_sin_theta, new_theta_dot], dim=1)


class PendulumCostFunction(CostFunction):
    # cost
    def __init__(self, angle_weight=1.0, velocity_weight=0.1, control_weight=0.001):
        self.angle_weight = angle_weight
        self.velocity_weight = velocity_weight
        self.control_weight = control_weight

    def compute_cost(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        cos_theta = states[:, 0]
        sin_theta = states[:, 1]
        theta_dot = states[:, 2]
        torque = actions[:, 0] if actions.shape[1] > 0 else torch.zeros_like(states[:, 0])
        theta = torch.atan2(sin_theta, cos_theta)
        angle_cost = self.angle_weight * (theta ** 2)
        velocity_cost = self.velocity_weight * (theta_dot ** 2)
        control_cost = self.control_weight * (torque ** 2)
        return angle_cost + velocity_cost + control_cost


class PendulumBaselinePolicy(BaselinePolicy):
    # use PD as baseline policy
    def __init__(self, kp=10.0, kd=1.0):
        self.kp = kp
        self.kd = kd

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        cos_theta = observation[:, 0]
        sin_theta = observation[:, 1]
        theta_dot = observation[:, 2]
        theta = torch.atan2(sin_theta, cos_theta)
        torque = self.kp * (-theta) + self.kd * (-theta_dot)
        torque = torch.clamp(torque, -2.0, 2.0)
        return torque.unsqueeze(1)


class PendulumObservationFunction(ObservationFunction):
    def state_to_observation(self, state: torch.Tensor) -> torch.Tensor:
        return state

def run_residual_mppi_example(render=True, render_mode='human', use_residual: bool = True):
    env = gym.make('Pendulum-v1', render_mode=render_mode if render else None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Rendering: {'Enabled' if render else 'Disabled'}")
    
    # MPPI
    horizon = 15
    num_samples = 2000
    action_dim = 1
    noise_std = 0.8
    temperature = 0.5
    action_lower_bound = torch.tensor([-2.0], device=device)
    action_upper_bound = torch.tensor([2.0], device=device)
    
    transition_model = PendulumTransitionModel()
    cost_function = PendulumCostFunction(angle_weight=1.0, velocity_weight=0.1, control_weight=0.001)
    baseline_policy = PendulumBaselinePolicy(kp=40.0, kd=1.0)
    observation_function = PendulumObservationFunction()
    
    mppi_controller = MPPIController(
        transition_model=transition_model,
        cost_function=cost_function,
        action_dim=action_dim,
        horizon=horizon,
        num_samples=num_samples,
        action_bounds=(action_lower_bound, action_upper_bound),
        noise_std=noise_std,
        temperature=temperature,
        device=device,
        use_residual_dynamics=use_residual,
        baseline_policy=baseline_policy,
        observation_function=observation_function,
        elite_ratio=0.1,
        smoothing_factor=0.9
    )
    

    def test_controller(controller, controller_name, num_episodes=3, render_episode=True):
        episode_returns, states_history, actions_history, costs_history = [], [], [], []
        for episode in range(num_episodes):
            obs, _ = env.reset()
            current_state = torch.tensor(obs, dtype=torch.float32, device=device)
            controller.reset()
            episode_return = 0
            episode_length = 200
            print(f"\n{controller_name} - Episode {episode + 1}")
            should_render = render and render_episode and episode == 0
            
            for step in range(episode_length):
                with torch.no_grad():
                    if use_residual:
                        action_tensor, info = controller.compute_control_with_info(current_state)
                        action = action_tensor[0].item()
                    else:
                        obs_tensor = current_state.unsqueeze(0)
                        action_tensor = baseline_policy.get_action(obs_tensor)
                        action = action_tensor[0, 0].item()
                        info = {'min_cost': 0.0, 'mean_cost': 0.0}

                obs, reward, terminated, truncated, _ = env.step([action])
                episode_return += reward
                if should_render:
                    env.render()
                    import time
                    time.sleep(0.05)
                states_history.append(current_state.cpu().numpy())
                actions_history.append(action)
                costs_history.append(info['min_cost'])
                current_state = torch.tensor(obs, dtype=torch.float32, device=device)
                if terminated or truncated:
                    break
            
            episode_returns.append(episode_return)
            final_angle = np.arctan2(obs[1], obs[0])
            print(f"Final state: cos={obs[0]:.3f}, sin={obs[1]:.3f}, theta_dot={obs[2]:.3f}")
            print(f"Final angle: {final_angle:.3f} rad ({np.degrees(final_angle):.1f} deg)")
            print(f"Episode Return: {episode_return:.2f}")
        
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        print(f"\n{controller_name} - Mean Return: {mean_return:.2f} ± {std_return:.2f}")
        return episode_returns, states_history, actions_history, costs_history
    
    controller_name = "Residual MPPI" if use_residual else "Baseline Only"
    returns, states, actions, costs = test_controller(mppi_controller, controller_name)
    
    env.close()
    
    # plot
    def plot_results(states, actions, costs, title):
        if len(states) == 0:
            return
        states = np.array(states)
        actions = np.array(actions)
        costs = np.array(costs)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title)
        time_steps = range(len(states))
        angles = np.arctan2(states[:, 1], states[:, 0])
        axes[0, 0].plot(time_steps, angles, 'b-', linewidth=2)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Angle (rad)')
        axes[0, 1].plot(time_steps, states[:, 2], 'g-', linewidth=2)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Angular Velocity (rad/s)')
        axes[0, 2].plot(time_steps, actions, 'orange', linewidth=2)
        axes[0, 2].set_title('Control Input (Nm)')
        axes[1, 0].plot(angles, states[:, 2], 'purple', alpha=0.7)
        axes[1, 0].scatter(angles[0], states[0, 2], color='green', s=100, marker='o', label='Start')
        axes[1, 0].scatter(angles[-1], states[-1, 2], color='red', s=100, marker='x', label='End')
        axes[1, 0].scatter(0, 0, color='gold', s=150, marker='*', label='Target')
        axes[1, 0].set_title('Phase Portrait')
        axes[1, 1].plot(time_steps, costs, 'red', linewidth=2)
        axes[1, 1].set_title('Cost Over Time')
        axes[1, 2].plot(time_steps, states[:, 0], 'b-', label='cos(theta)', linewidth=2)
        axes[1, 2].plot(time_steps, states[:, 1], 'r-', label='sin(theta)', linewidth=2)
        axes[1, 2].set_title('State Components')
        axes[1, 2].legend()
        plt.tight_layout()
        plt.show()
    
    plot_results(states, actions, costs, f"{controller_name} Control Results")


if __name__ == "__main__":
    render_enabled = True
    use_residual = True
    
    # args
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == '--headless':
                render_enabled = False
            elif arg == '--baseline-only':
                use_residual = False
    
    print("MPPI Pendulum Residual Dynamics Control Demo")
    run_residual_mppi_example(render=render_enabled, use_residual=use_residual)
