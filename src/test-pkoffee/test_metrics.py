import pytest
import numpy as np

def test_size_mismatch_valid():
    from pkoffee.metrics import check_size_match
    a = np.array([1, 2, 3])
    b = np.array([2, 85, 4])
    check_size_match(a, b)

a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([1.1, 1.9, 3.1, 3.9])

def test_r2_result():
    from pkoffee.metrics import compute_r2
    assert compute_r2(a, b) == 0.9920

def test_rmse_result():
    from pkoffee.metrics import compute_rmse
    assert compute_rmse(a, b) == pytest.approx(0.1000)

def test_mae_result():
    from pkoffee.metrics import compute_mae
    assert compute_mae(a, b) == pytest.approx(0.1000)

