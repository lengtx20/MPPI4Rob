import torch
from abc import ABC, abstractmethod

class ObservationFunction(ABC):
    """define Observation, if needed"""
    
    @abstractmethod
    def state_to_observation(self, state: torch.Tensor) -> torch.Tensor:
        """
        From state to obs, in case state != obs
        Args:
            state: (batch_size, state_dim) state
        Returns:
            observation: (batch_size, obs_dim) obs
        """
        pass
