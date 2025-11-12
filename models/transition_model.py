"""Transition model base class"""
import torch
from abc import ABC, abstractmethod

class TransitionModel(ABC):
    """Dynamics model interface."""

    @abstractmethod
    def forward(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        """Propagate states with actions to next states."""
        pass
