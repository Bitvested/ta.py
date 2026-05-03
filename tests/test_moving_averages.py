"""Tests for moving average functions."""
import pytest
import ta_py
from tests import parameters as p
from tests.helpers import assert_close

CASES = [
    "sma", "smma", "wma", "wsma", "pwma", "hwma", "ema",
    "vwma", "vwwma", "hull", "kama", "cwma", "lsma",
    "dema", "tema", "trima", "t3", "zlema", "vidya",
]


@pytest.mark.parametrize("name", CASES)
def test_moving_average(name: str) -> None:
    fixture = getattr(p, name)
    fn = getattr(ta_py, name)
    actual = fn(*fixture["in"])
    assert_close(actual, fixture["out"], name=name)
