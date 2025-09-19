import torch
from abc import ABC, abstractmethod

class BaselinePolicy(ABC):
    """define Baseline Policy"""
    
    @abstractmethod
    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        """
        get a_baseline
        Args:
            observation: (batch_size, obs_dim)
        Returns:
            action: (batch_size, action_dim)
        """
        pass
