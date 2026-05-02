"""Tests for experimental functions."""
import ta_py
from tests import parameters as p
from tests.helpers import assert_close


def test_divergence_state() -> None:
    fixture = p.divergence_state
    actual = ta_py.divergence_state(*fixture["in"])
    assert_close(actual, fixture["out"], name="divergence_state")


def test_support() -> None:
    sup = ta_py.support([4, 3, 2, 5, 7, 6, 5, 4, 7, 8, 5, 4, 6, 7, 5])
    assert_close(sup["calculate"](9), 4.0, name="support.calculate(9)")


def test_resistance() -> None:
    res = ta_py.resistance([5, 7, 5, 5, 4, 6, 5, 4, 6, 5, 4, 3, 2, 4, 3, 2, 1])
    assert_close(res["calculate"](4), 6.428571428571429, name="resistance.calculate(4)")
