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


def test_zigzag_pairs() -> None:
    fixture = p.zigzag__pairs
    actual = ta_py.zigzag(*fixture["in"])
    assert_close(actual, fixture["out"], name="zigzag (pairs)")


def test_zigzag_series() -> None:
    fixture = p.zigzag__series
    actual = ta_py.zigzag(*fixture["in"])
    assert_close(actual, fixture["out"], name="zigzag (series)")
