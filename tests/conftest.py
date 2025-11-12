"""Conftest for pytest configuration"""
import pytest
import torch


@pytest.fixture
def device():
    """Fixture to get the appropriate device"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def dtype():
    """Fixture for tensor data type"""
    return torch.float32


@pytest.fixture
def batch_size():
    """Standard batch size for tests"""
    return 4


@pytest.fixture
def state_dim():
    """Standard state dimension"""
    return 3


@pytest.fixture
def action_dim():
    """Standard action dimension"""
    return 1


@pytest.fixture
def horizon():
    """Standard time horizon"""
    return 15


@pytest.fixture
def num_samples():
    """Standard number of samples"""
    return 100


@pytest.fixture
def sample_states(device, dtype, batch_size, state_dim):
    """Create sample states tensor"""
    return torch.randn(batch_size, state_dim, dtype=dtype, device=device)


@pytest.fixture
def sample_actions(device, dtype, batch_size, action_dim):
    """Create sample actions tensor"""
    return torch.randn(batch_size, action_dim, dtype=dtype, device=device)


@pytest.fixture
def sample_action_sequences(device, dtype, num_samples, horizon, action_dim):
    """Create sample action sequences tensor"""
    return torch.randn(num_samples, horizon, action_dim, dtype=dtype, device=device)
