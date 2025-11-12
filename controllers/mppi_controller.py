"""MPPI Controller - Model Predictive Path Integral Control"""

import torch
import math
from typing import Optional, Tuple, Dict, Any
from models.transition_model import TransitionModel
from models.cost_function import CostFunction
from models.baseline_policy import BaselinePolicy
from models.observation import ObservationFunction


class MPPIController:
    """MPPI controller with stability and annealing features."""
    
    def __init__(self, transition_model, cost_function, action_dim, horizon, num_samples,
                 action_bounds, noise_std=1.0, temperature=1.0, device='cuda' if torch.cuda.is_available() else 'cpu',
                 use_residual_dynamics=False, baseline_policy=None, observation_function=None,
                 elite_ratio=0.1, smoothing_factor=0.9,
                 adaptive_temperature=True, noise_annealing=True, use_log_space_weights=True,
                 control_mode: Optional[str] = None):
        self.transition_model = transition_model
        self.cost_function = cost_function
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.device = device
        self.noise_std_init = noise_std
        self.noise_std = noise_std
        self.temperature_init = temperature
        self.temperature = temperature
        self.elite_ratio = elite_ratio
        self.smoothing_factor = smoothing_factor
        self.adaptive_temperature_enabled = adaptive_temperature
        self.noise_annealing_enabled = noise_annealing
        self.use_log_space_weights = use_log_space_weights
        self.action_lower_bound = action_bounds[0].to(device)
        self.action_upper_bound = action_bounds[1].to(device)
        self.use_residual_dynamics = use_residual_dynamics
        self.baseline_policy = baseline_policy
        self.observation_function = observation_function
        if use_residual_dynamics and (baseline_policy is None or observation_function is None):
            raise ValueError("baseline_policy and observation_function required for residual dynamics")
        # control_mode: None -> default behavior
        # if use_residual_dynamics: default -> 'residual+baseline' (returns full action)
        # else: default -> 'mppi' (standard MPPI output)
        if control_mode is None:
            self.control_mode = 'residual+baseline' if use_residual_dynamics else 'mppi'
        else:
            assert control_mode in ('mppi', 'residual', 'baseline', 'residual+baseline', 'combined')
            # accept 'combined' as alias for 'residual+baseline'
            self.control_mode = 'residual+baseline' if control_mode == 'combined' else control_mode
        self.previous_action_sequence = None
        self.step_count = 0
        self.num_elite = max(1, int(self.num_samples * self.elite_ratio))
    
    def get_current_noise_std(self) -> float:
        """Return current noise std (cosine annealing)."""
        if not self.noise_annealing_enabled:
            return self.noise_std
        t = min(self.step_count, 100)
        progress = t / 100.0
        noise_final = self.noise_std_init * 0.5
        return noise_final + 0.5 * (self.noise_std_init - noise_final) * (1 + math.cos(math.pi * progress))
    
    def compute_weight_entropy(self, weights: torch.Tensor) -> float:
        """Return Shannon entropy of weights."""
        weights_safe = torch.clamp(weights, min=1e-10)
        return -torch.sum(weights * torch.log(weights_safe)).item()
    
    def adapt_temperature(self, weights: torch.Tensor) -> None:
        """Adjust temperature λ based on weight entropy."""
        if not self.adaptive_temperature_enabled:
            return
        entropy = self.compute_weight_entropy(weights)
        ideal_entropy = math.log(self.num_samples / 5.0)
        if entropy > ideal_entropy * 1.2:
            self.temperature = max(0.1, self.temperature * 0.95)
        elif entropy < ideal_entropy * 0.8:
            self.temperature = min(2.0, self.temperature * 1.05)
    
    def compute_weights(self, costs: torch.Tensor) -> torch.Tensor:
        """Compute importance weights (log-space for stability)."""
        if self.use_log_space_weights:
            log_weights = -(costs - torch.min(costs)) / self.temperature
            weights = torch.softmax(log_weights, dim=0)
        else:
            min_cost = torch.min(costs)
            exp_costs = torch.exp(-(costs - min_cost) / self.temperature)
            weights = exp_costs / torch.sum(exp_costs)
        return weights
    
    def set_adaptive_temperature(self, enabled: bool) -> None:
        """Toggle adaptive temperature."""
        self.adaptive_temperature_enabled = enabled
    
    def set_noise_annealing(self, enabled: bool) -> None:
        """Toggle noise annealing."""
        self.noise_annealing_enabled = enabled
    
    def sample_actions(self) -> torch.Tensor:
        """Sample action sequences around previous plan."""
        if self.previous_action_sequence is not None:
            mean_actions = torch.cat([self.previous_action_sequence[1:],
                                     torch.zeros(1, self.action_dim, device=self.device)], dim=0)
        else:
            mean_actions = torch.zeros(self.horizon, self.action_dim, device=self.device)
        
        current_noise_std = self.get_current_noise_std()
        noise = torch.randn(self.num_samples, self.horizon, self.action_dim, device=self.device)
        action_sequences = mean_actions.unsqueeze(0) + noise * current_noise_std
        return torch.clamp(action_sequences, self.action_lower_bound, self.action_upper_bound)
    
    def rollout(self, initial_state: torch.Tensor, action_sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate trajectories and accumulate costs."""
        batch_size = action_sequences.shape[0]
        trajectories = torch.zeros(batch_size, self.horizon + 1, initial_state.shape[0], device=self.device)
        trajectories[:, 0] = initial_state.repeat(batch_size, 1)
        total_costs = torch.zeros(batch_size, device=self.device)
        current_states = initial_state.repeat(batch_size, 1)
        
        for t in range(self.horizon):
            current_actions = action_sequences[:, t]
            if self.use_residual_dynamics:
                observations = self.observation_function.state_to_observation(current_states)
                baseline_actions = self.baseline_policy.get_action(observations)
                final_actions = baseline_actions + current_actions
                final_actions = torch.clamp(final_actions, self.action_lower_bound, self.action_upper_bound)
            else:
                final_actions = current_actions
            
            step_costs = self.cost_function.compute_cost(current_states, final_actions)
            total_costs += step_costs
            next_states = self.transition_model.forward(current_states, final_actions)
            current_states = next_states
            trajectories[:, t + 1] = next_states
        
        terminal_costs = self.cost_function.compute_cost(current_states, torch.zeros_like(current_actions))
        total_costs += terminal_costs
        return total_costs, trajectories
    
    def compute_control(self, current_state: torch.Tensor) -> torch.Tensor:
        """Compute and return the next control action.

        Depending on `self.control_mode` this may return:
        - 'mppi': the MPPI action (default non-residual)
        - 'residual': the residual action only
        - 'baseline': baseline action only
        - 'residual+baseline': baseline + residual (full action)
        """
        # Baseline-only mode
        if self.control_mode == 'baseline':
            if self.baseline_policy is None or self.observation_function is None:
                raise ValueError('baseline_policy and observation_function required for baseline mode')
            obs = self.observation_function.state_to_observation(current_state.unsqueeze(0))
            baseline_action = self.baseline_policy.get_action(obs)
            return baseline_action[0]

        # Sample and evaluate (used by mppi/residual modes)
        action_sequences = self.sample_actions()
        costs, _ = self.rollout(current_state, action_sequences)
        weights = self.compute_weights(costs)
        self.adapt_temperature(weights)
        weights = self.compute_weights(costs)
        optimal_action_sequence = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * action_sequences, dim=0)
        self.previous_action_sequence = optimal_action_sequence.detach()
        self.step_count += 1

        residual = optimal_action_sequence[0]

        if self.control_mode == 'residual':
            return residual
        elif self.control_mode in ('residual+baseline',):
            # compute baseline for current state and add residual
            if self.baseline_policy is None or self.observation_function is None:
                raise ValueError('baseline_policy and observation_function required for residual+baseline mode')
            obs = self.observation_function.state_to_observation(current_state.unsqueeze(0))
            baseline_action = self.baseline_policy.get_action(obs)[0]
            full_action = torch.clamp(baseline_action + residual, self.action_lower_bound, self.action_upper_bound)
            return full_action
        else:
            # 'mppi' default return (treats MPPI action as full action)
            return residual
    
    def compute_control_with_info(self, current_state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute control and return diagnostic info."""
        action_sequences = self.sample_actions()
        costs, trajectories = self.rollout(current_state, action_sequences)
        weights = self.compute_weights(costs)
        self.adapt_temperature(weights)
        weights = self.compute_weights(costs)
        optimal_action_sequence = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * action_sequences, dim=0)
        self.previous_action_sequence = optimal_action_sequence.detach()
        _, elite_indices = torch.topk(-costs, self.num_elite)

        residual = optimal_action_sequence[0]

        baseline_action = None
        full_action = None
        if self.baseline_policy is not None and self.observation_function is not None:
            obs = self.observation_function.state_to_observation(current_state.unsqueeze(0))
            baseline_action = self.baseline_policy.get_action(obs)[0]
            full_action = torch.clamp(baseline_action + residual, self.action_lower_bound, self.action_upper_bound)

        info = {
            'optimal_action_sequence': optimal_action_sequence,
            'all_costs': costs,
            'elite_costs': costs[elite_indices],
            'elite_trajectories': trajectories[elite_indices],
            'weights': weights,
            'min_cost': torch.min(costs),
            'mean_cost': torch.mean(costs),
            'temperature': self.temperature,
            'noise_std': self.get_current_noise_std(),
            'weight_entropy': self.compute_weight_entropy(weights),
            'residual': residual,
            'baseline_action': baseline_action,
            'full_action': full_action,
            'control_mode': self.control_mode,
        }
        self.step_count += 1
        # Return an action consistent with control_mode
        if self.control_mode == 'baseline':
            return baseline_action, info
        if self.control_mode == 'residual':
            return residual, info
        if self.control_mode in ('residual+baseline',):
            return full_action, info
        return residual, info
    
    def reset(self):
        """Reset internal controller state."""
        self.previous_action_sequence = None
        self.step_count = 0
        self.temperature = self.temperature_init
        self.noise_std = self.noise_std_init
