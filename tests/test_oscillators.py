"""Tests for oscillator functions."""
import pytest
import ta_py
from tests import parameters as p
from tests.helpers import assert_close

CASES = ["gator", "mom_osc", "chaikin_osc", "ao", "ac", "fisher", "kvo"]


@pytest.mark.parametrize("name", CASES)
def test_oscillator(name: str) -> None:
    fixture = getattr(p, name)
    fn = getattr(ta_py, name)
    actual = fn(*fixture["in"])
    assert_close(actual, fixture["out"], name=name)
