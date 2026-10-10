"""Contract tests for halftrend, encoding everget's Pine source.

Ported from tradingview.com/script/U1SJ8ubc-HalfTrend/. Three properties of that
Pine are what a values fixture alone cannot pin, and what the previous
implementation got wrong:

* one row per input bar, with the ATR channel na until the ATR window is full
  while the trend line is defined from bar 0;
* the price extremes come from `amplitude` bars, not from the ATR window;
* the trend flips at all. The old code read `sma(...)[0]`, the SMA at the start
  of its window rather than the current bar, so its flip conditions compared
  values around 100 bars old and never fired — on 4901 bars of 5-minute data it
  produced zero flips.
"""
import math

import ta_py
from tests.helpers import assert_close


def sawtooth():
    """A deterministic zig-zag: twelve legs of four bars, 48 bars in total."""
    bars, price = [], 100.0
    for leg in (3, -3, 4, -4, 5, -5, 6, -6, 4, -4, 2, -2):
        for _ in range(4):
            price += leg
            bars.append([price + 1.0, price, price - 1.0])
    return bars


def test_one_row_per_bar():
    for n in (6, 12, 48):
        data = sawtooth()[:n]
        assert len(ta_py.halftrend(data, 6, 3, 2)) == len(data)


def test_channel_is_nan_until_the_atr_window_is_full():
    data = sawtooth()
    out = ta_py.halftrend(data, 6, 3, 2)
    for i, row in enumerate(out):
        for col in (0, 2):
            assert math.isnan(row[col]) == (i < 5), f"atrHigh/atrLow at row {i}"
        assert not math.isnan(row[1]), f"trend line undefined at row {i}"


def test_trend_flips_and_pinned_values():
    out = ta_py.halftrend(sawtooth(), 6, 3, 2)
    flips = [i for i in range(1, len(out)) if out[i][3] != out[i - 1][3]]
    assert flips == [6, 10, 14, 18, 22, 26, 30, 35, 38]
    assert_close(
        [out[i][1] for i in (5, 6, 20, 24, 47)],
        [108.0, 108.0, 114.0, 107.0, 105.0],
        name="halftrend trend line",
    )
    assert_close(
        [out[i][0] for i in (5, 6, 24)],
        [111.66666666666667, 111.72222222222223, 112.91735310176668],
        name="halftrend atrHigh",
    )
