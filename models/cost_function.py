"""Cost function base class"""
import torch
from abc import ABC, abstractmethod

class CostFunction(ABC):
    """Cost function interface."""

    @abstractmethod
    def compute_cost(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute per-trajectory costs."""
        pass
