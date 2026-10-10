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


HULL_ROUNDING_CASES = ["hull__odd_length", "hull__sqrt_rounds_up"]


@pytest.mark.parametrize("name", HULL_ROUNDING_CASES)
def test_hull_rounding_matches_the_reference(name: str) -> None:
    """Hull's half-length and final window must round like ta.js's Math.round.

    Python's round() is half-to-even and Go's int() truncates, so both ports
    disagreed with the reference: ta.py for every length that is 1 mod 4 (the
    half-length is a half-integer there), and ta.go at every odd length plus
    wherever sqrt rounds up. Two fixture lengths: 5 (half-integer half-length) and
    8 (sqrt 8 = 2.83, which rounds up).
    """
    fixture = getattr(p, name)
    assert_close(ta_py.hull(*fixture["in"]), fixture["out"], name=name)


@pytest.mark.parametrize("name", CASES)
def test_moving_average(name: str) -> None:
    fixture = getattr(p, name)
    fn = getattr(ta_py, name)
    actual = fn(*fixture["in"])
    assert_close(actual, fixture["out"], name=name)
