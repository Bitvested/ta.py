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
    "fi", "asi", "alligator", "occ", "pr", "stoch", "fib", "aroon_up",
    "aroon_down", "aroon_osc", "adl", "nvi", "pvi", "emv", "dpo", "ulcer",
    "pdm", "mdm", "pdi", "mdi", "dx", "adx", "adxr", "natr",
    "trix", "ppo", "apo", "mass",
    "cci", "stoch_rsi", "kdj",
    "cmf", "vortex",
]


@pytest.mark.parametrize("name", CASES)
def test_indicator(name: str) -> None:
    fixture = getattr(p, name)
    fn = getattr(ta_py, name)
    actual = fn(*fixture["in"])
    assert_close(actual, fixture["out"], name=name)


def test_aroon_namespace() -> None:
    """ta_py.aroon.{up,down,osc} are silent aliases of the flat names."""
    data = [5, 4, 5, 2]
    assert ta_py.aroon.up(data, 3) == ta_py.aroon_up(data, 3)
    data2 = [2, 5, 4, 5]
    assert ta_py.aroon.down(data2, 3) == ta_py.aroon_down(data2, 3)
    assert ta_py.aroon.osc(data2, 3) == ta_py.aroon_osc(data2, 3)


AROON_EXTREME_CASES = [
    "aroon_up__newest_high", "aroon_up__oldest_high",
    "aroon_down__newest_low", "aroon_down__oldest_low",
]


@pytest.mark.parametrize("name", AROON_EXTREME_CASES)
def test_aroon_extreme_position(name: str) -> None:
    """Aroon is 100 when the window's extreme is its newest bar, 0 when it is the oldest.

    The plain aroon fixtures place the extreme at a tie and at the window's middle,
    the two positions where an inverted Aroon returns the same number — these are the
    cases that tell the conventions apart.
    """
    fixture = getattr(p, name)
    fn = getattr(ta_py, name.split("__")[0])
    assert_close(fn(*fixture["in"]), fixture["out"], name=name)


def test_roc_zero_reference() -> None:
    """A zero reference yields NaN rather than dividing by zero."""
    fixture = p.roc__zero_reference
    assert_close(ta_py.roc(*fixture["in"]), fixture["out"], name="roc (zero reference)")


def test_zigzag_pairs() -> None:
    fixture = p.zigzag__pairs
    actual = ta_py.zigzag(*fixture["in"])
    assert_close(actual, fixture["out"], name="zigzag (pairs)")


def test_zigzag_series() -> None:
    fixture = p.zigzag__series
    actual = ta_py.zigzag(*fixture["in"])
    assert_close(actual, fixture["out"], name="zigzag (series)")
