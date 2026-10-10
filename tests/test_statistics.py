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
    "lr_slope", "lr_intercept", "lr_angle", "tsf", "std_series",
]


@pytest.mark.parametrize("name", CASES)
def test_statistic(name: str) -> None:
    fixture = getattr(p, name)
    fn = getattr(ta_py, name)
    actual = fn(*fixture["in"])
    assert_close(actual, fixture["out"], name=name)


def test_er_one_sided_series() -> None:
    """No losses (or no wins) is not degenerate: that side's weight is zero, so only
    the other side's geometric mean contributes."""
    assert_close(ta_py.er(*p.er__no_losses["in"]), p.er__no_losses["out"], name="er (no losses)")
    assert_close(ta_py.er(*p.er__no_wins["in"]), p.er__no_wins["out"], name="er (no wins)")


def test_kelly_all_winning_series() -> None:
    """Every observation a win means the Kelly fraction is 1, not NaN."""
    assert_close(ta_py.kelly(*p.kelly__no_losses["in"]), p.kelly__no_losses["out"], name="kelly (no losses)")


def test_sum() -> None:
    fixture = p.sum_
    actual = ta_py.sum(*fixture["in"])
    assert_close(actual, fixture["out"], name="sum")
