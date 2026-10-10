"""Contract tests for alligator/gator against the Pine source in issue #10.

TradingView's ``plot(..., offset=shift)`` shifts the plot *right*, so bar ``i``
draws the value computed at bar ``i - shift``. The assertions below encode that
reading directly, plus the SMMA seed (``ta.sma(src, length)`` over the first
``length`` values) that the Pine shows verbatim. They pin the contract against
regression; they do not by themselves prove TradingView parity — that is what
``test_tradingview_parity.py`` is for.
"""
import math

import pytest

import ta_py

DATA = [8, 7, 8, 9, 7, 8, 9, 6, 7, 8, 6, 8, 10, 8, 7, 9, 8, 7, 9, 6, 7, 9]


def test_rows_align_to_input_bars():
    for n in (22, 60, 200):
        data = (DATA * 10)[:n]
        assert len(ta_py.alligator(data)) == n
        assert len(ta_py.gator(data)) == n


def test_first_defined_bar_is_length_minus_one_plus_shift():
    out = ta_py.alligator(DATA)
    for col, first in enumerate((13 - 1 + 8, 8 - 1 + 5, 5 - 1 + 3)):
        defined = [i for i, row in enumerate(out) if not math.isnan(row[col])]
        assert defined[0] == first
        assert defined[-1] == len(DATA) - 1


def test_zero_shifts_drop_the_displacement():
    """js=ts=ls=0 is TradingView's Gator Oscillator, which does not displace."""
    out = ta_py.alligator(DATA, 13, 8, 5, 0, 0, 0)
    for col, first in enumerate((13 - 1, 8 - 1, 5 - 1)):
        defined = [i for i, row in enumerate(out) if not math.isnan(row[col])]
        assert defined[0] == first


def test_smma_seed_lands_shift_bars_later():
    """Hand arithmetic, no implementation involved.

    The fixture's first 13 values sum to 101, its first 8 to 62 and its first 5
    to 39, so the Pine seeds are 101/13, 62/8 and 39/5 at bars 12, 7 and 4 --
    each drawn ``shift`` bars later.
    """
    out = ta_py.alligator(DATA)
    assert out[20][0] == pytest.approx(101 / 13)
    assert out[12][1] == pytest.approx(62 / 8)
    assert out[7][2] == pytest.approx(39 / 5)


@pytest.mark.parametrize("jl,js", [(13, 8), (13, 0), (5, 3), (21, 1)])
def test_jaw_is_the_raw_smma_displaced_by_shift(jl, js):
    data = (DATA * 10)[:120]
    out = ta_py.alligator(data, jl, 8, 5, js, 5, 3)
    raw = ta_py.smma(data, jl)
    first = jl - 1 + js
    assert all(math.isnan(row[0]) for row in out[:first])
    for i in range(first, len(data)):
        assert out[i][0] == pytest.approx(raw[i - first])


def test_gator_is_derived_from_the_displaced_lines():
    for shifts in ((), (8, 5, 3), (0, 0, 0), (3, 2, 1)):
        lines = ta_py.alligator(DATA, 13, 8, 5, *shifts)
        histogram = ta_py.gator(DATA, 13, 8, 5, *shifts)
        for (jaw, teeth, lips), (upper, lower) in zip(lines, histogram):
            assert math.isnan(upper) == (math.isnan(jaw) or math.isnan(teeth))
            if not math.isnan(upper):
                assert upper == pytest.approx(abs(jaw - teeth))
            assert math.isnan(lower) == (math.isnan(teeth) or math.isnan(lips))
            if not math.isnan(lower):
                assert lower == pytest.approx(-abs(teeth - lips))


def test_gator_upper_histogram_is_never_negative():
    assert all(math.isnan(u) or u >= 0 for u, _ in ta_py.gator(DATA))


def test_data_shorter_than_the_jaw_length_is_all_nan():
    for fn in (ta_py.alligator, ta_py.gator):
        out = fn([1.0] * 5)
        assert len(out) == 5
        assert all(math.isnan(v) for row in out for v in row)
