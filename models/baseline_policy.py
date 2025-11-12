import torch
from abc import ABC, abstractmethod

class BaselinePolicy(ABC):
    """Baseline policy interface."""

    @abstractmethod
    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Return baseline action for an observation."""
        pass
