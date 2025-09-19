import torch
from abc import ABC, abstractmethod

class CostFunction(ABC):
    """define Cost Function"""
    
    @abstractmethod
    def compute_cost(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute cost
        Args:
            states: (batch_size, state_dim) 
            actions: (batch_size, action_dim) 
        Returns:
            costs: (batch_size,)
        """
        pass
