import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from models.transition_model import TransitionModel
from models.cost_function import CostFunction
from models.baseline_policy import BaselinePolicy
from models.observation import ObservationFunction


class MPPIController:
    def __init__(
        self,
        transition_model: TransitionModel,
        cost_function: CostFunction,
        action_dim: int,
        horizon: int,
        num_samples: int,
        action_bounds: Tuple[torch.Tensor, torch.Tensor],
        noise_std: float = 1.0,
        temperature: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',

        use_residual_dynamics: bool = False,
        baseline_policy: Optional[BaselinePolicy] = None,
        observation_function: Optional[ObservationFunction] = None,

        elite_ratio: float = 0.1,
        smoothing_factor: float = 0.9
    ):
        self.transition_model = transition_model
        self.cost_function = cost_function
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.device = device
        self.noise_std = noise_std
        self.temperature = temperature
        self.elite_ratio = elite_ratio
        self.smoothing_factor = smoothing_factor
        
        # clip
        self.action_lower_bound = action_bounds[0].to(device)
        self.action_upper_bound = action_bounds[1].to(device)
        
        # Residual dynamics
        self.use_residual_dynamics = use_residual_dynamics
        self.baseline_policy = baseline_policy
        self.observation_function = observation_function
        if use_residual_dynamics:
            if baseline_policy is None or observation_function is None:
                raise ValueError("[ERROR] given NO baseline_policy and observation_function")
        
        self.previous_action_sequence = None
        
        self.num_elite = max(1, int(self.num_samples * self.elite_ratio))
    
    def sample_actions(self) -> torch.Tensor:
        if self.previous_action_sequence is not None:
            mean_actions = torch.cat([
                self.previous_action_sequence[1:],
                torch.zeros(1, self.action_dim, device=self.device)
            ], dim=0)
        else:
            mean_actions = torch.zeros(self.horizon, self.action_dim, device=self.device)
        
        noise = torch.randn(self.num_samples, self.horizon, self.action_dim, device=self.device)
        noise *= self.noise_std
        action_sequences = mean_actions.unsqueeze(0) + noise
        action_sequences = torch.clamp(action_sequences, self.action_lower_bound, self.action_upper_bound)
        return action_sequences
    
    def rollout(self, initial_state: torch.Tensor, action_sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        action_sequences = self.sample_actions()
        costs, _ = self.rollout(current_state, action_sequences)
        min_cost = torch.min(costs)
        exp_costs = torch.exp(-(costs - min_cost) / self.temperature)
        weights = exp_costs / torch.sum(exp_costs)
        optimal_action_sequence = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * action_sequences, dim=0)
        self.previous_action_sequence = optimal_action_sequence.detach()
        return optimal_action_sequence[0]
    
    def compute_control_with_info(self, current_state: torch.Tensor) -> Dict[str, Any]:
        action_sequences = self.sample_actions()
        costs, trajectories = self.rollout(current_state, action_sequences)
        min_cost = torch.min(costs)
        exp_costs = torch.exp(-(costs - min_cost) / self.temperature)
        weights = exp_costs / torch.sum(exp_costs)
        optimal_action_sequence = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * action_sequences, dim=0)
        self.previous_action_sequence = optimal_action_sequence.detach()
        _, elite_indices = torch.topk(-costs, self.num_elite)
        elite_costs = costs[elite_indices]
        elite_trajectories = trajectories[elite_indices]
        info = {
            'optimal_action_sequence': optimal_action_sequence,
            'all_costs': costs,
            'elite_costs': elite_costs,
            'elite_trajectories': elite_trajectories,
            'weights': weights,
            'min_cost': min_cost,
            'mean_cost': torch.mean(costs)
        }
        return optimal_action_sequence[0], info
    
    def reset(self):
        self.previous_action_sequence = None
