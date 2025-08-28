"""Test configuration and shared fixtures."""
import pytest
import numpy as np


@pytest.fixture
def unit_square():
    """Standard unit square vertices."""
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


@pytest.fixture  
def optimization_tolerances():
    """Standard tolerances for different contexts."""
    return {
        "reference": {"rtol": 1e-9, "atol": 1e-12},
        "efficient": {"rtol": 1e-6, "atol": 1e-8},
        "gradient": {"rtol": 1e-5, "atol": 1e-7}
    }