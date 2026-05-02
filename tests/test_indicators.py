"""Tests for indicator functions."""
import pytest
import ta_py
from tests import parameters as p
from tests.helpers import assert_close

CASES = [
    "rsi", "wrsi", "tsi", "macd", "macd_signal", "macd_bars", "atr",
    "mfi", "roc", "mom", "kst", "cop", "obv", "vwap", "ichimoku",
    "fractals", "cross", "halftrend", "psar", "supertrend", "elderray",
    "hv", "rvi", "rvi_signal", "rsi_divergence", "divergence", "bop",
    "fi", "asi", "alligator", "pr", "stoch", "fib", "aroon_up",
    "aroon_down", "aroon_osc",
]


@pytest.mark.parametrize("name", CASES)
def test_indicator(name: str) -> None:
    fixture = getattr(p, name)
    fn = getattr(ta_py, name)
    actual = fn(*fixture["in"])
    assert_close(actual, fixture["out"], name=name)


def test_zigzag_pairs() -> None:
    fixture = p.zigzag__pairs
    actual = ta_py.zigzag(*fixture["in"])
    assert_close(actual, fixture["out"], name="zigzag (pairs)")


def test_zigzag_series() -> None:
    fixture = p.zigzag__series
    actual = ta_py.zigzag(*fixture["in"])
    assert_close(actual, fixture["out"], name="zigzag (series)")
