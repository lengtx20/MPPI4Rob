import math
import torch

from controllers.mppi_controller import MPPIController
from models.transition_model import TransitionModel
from models.cost_function import CostFunction


class DummyTransitionModel(TransitionModel):
    """Simple dynamics: x_{t+1} = x_t + action tiled to state dim."""

    def forward(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        b, sdim = states.shape
        adim = actions.shape[1]
        # tile actions to match state dim
        repeats = math.ceil(sdim / adim)
        tiled = actions.repeat(1, repeats)[:, :sdim]
        return states + tiled


class DummyCostFunction(CostFunction):
    """Quadratic cost on state and small penalty on action magnitude."""

    def compute_cost(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        state_cost = torch.sum(states ** 2, dim=1)
        action_cost = 0.1 * torch.sum(actions ** 2, dim=1)
        return state_cost + action_cost


def test_mppi_compute_control():
    device = 'cpu'
    state_dim = 3
    action_dim = 1
    horizon = 5
    num_samples = 50

    transition = DummyTransitionModel()
    cost = DummyCostFunction()

    lower = torch.full((action_dim,), -1.0, device=device)
    upper = torch.full((action_dim,), 1.0, device=device)

    controller = MPPIController(
        transition_model=transition,
        cost_function=cost,
        action_dim=action_dim,
        horizon=horizon,
        num_samples=num_samples,
        action_bounds=(lower, upper),
        noise_std=0.5,
        temperature=1.0,
        device=device,
    )

    initial_state = torch.zeros(state_dim, device=device)

    action = controller.compute_control(initial_state)
    assert isinstance(action, torch.Tensor)
    assert action.shape[0] == action_dim

    # info call
    action2, info = controller.compute_control_with_info(initial_state)
    assert 'all_costs' in info and 'weights' in info
