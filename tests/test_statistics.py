"""Tests for statistical functions."""
import pytest
import ta_py
from tests import parameters as p
from tests.helpers import assert_close

CASES = [
    "median", "kmeans", "normalize", "denormalize", "mad", "aad", "ssd",
    "er", "variance", "std", "normsinv", "percentile", "cor", "dif",
    "drawdown", "cum", "mse", "permutations", "winratio", "avgwin",
    "avgloss", "kelly", "se", "zscore", "ar", "normalize_pair",
    "normalize_from", "log", "exp", "covariance", "ncdf", "expected_trails",
    "pvalue", "martingale", "antimartingale", "return_positive",
    "return_negative", "recent_high", "recent_low", "standardize",
    "lr_slope",
]


@pytest.mark.parametrize("name", CASES)
def test_statistic(name: str) -> None:
    fixture = getattr(p, name)
    fn = getattr(ta_py, name)
    actual = fn(*fixture["in"])
    assert_close(actual, fixture["out"], name=name)


def test_sum() -> None:
    fixture = p.sum_
    actual = ta_py.sum(*fixture["in"])
    assert_close(actual, fixture["out"], name="sum")
