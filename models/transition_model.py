import torch
from abc import ABC, abstractmethod

class TransitionModel(ABC):
    """To define the transition model, for example Koopman model"""
    
    @abstractmethod
    def forward(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        (x_t, a_t) -> (x_t+1)
        Args:
            states: (batch_size, state_dim) 
            actions: (batch_size, action_dim) 
        Returns:
            next_states: (batch_size, state_dim) 
        """
        pass
