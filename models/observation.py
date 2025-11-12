import torch
from abc import ABC, abstractmethod

class ObservationFunction(ABC):
    """Observation mapping interface."""

    @abstractmethod
    def state_to_observation(self, state: torch.Tensor) -> torch.Tensor:
        """Map state to observation tensor."""
        pass
