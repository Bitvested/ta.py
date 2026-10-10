"""Parity test against a TradingView chart-data export.

Everything else in the suite is derived from the Pine source pasted in issue
#10; this is the only check that compares against numbers TradingView itself
plotted, and therefore the only one that can actually establish parity.

To capture the fixture: put the Williams Alligator on a chart (defaults), scroll
fully left so every loaded bar is in memory, then Chart menu -> "Download chart
data" -> CSV. The export carries a column per study in the Data Window, so
Jaw/Teeth/Lips come along with the OHLC. Save it as
``tests/fixtures/tradingview_alligator.csv``.

Downloading chart data needs a paid plan. On a free plan, paste this into the
Pine Editor instead, open the Logs panel, and copy the whole log out — it is
already a CSV with a header:

    //@version=5
    indicator("Alligator export", overlay=true)

    smma(src, length) =>
        s = 0.0
        s := na(s[1]) ? ta.sma(src, length) : (s[1] * (length - 1) + src) / length
        s

    jaw   = smma(hl2, 13)
    teeth = smma(hl2, 8)
    lips  = smma(hl2, 5)

    if bar_index == 0
        log.info("bar_index,high,low,Jaw,Teeth,Lips")

    // jaw[8] is exactly what plot(jaw, offset = 8) draws on this bar.
    log.info("{0},{1},{2},{3},{4},{5}", bar_index, high, low, jaw[8], teeth[5], lips[3])

Either source works, and an undefined plot is read as undefined whether it is
written as an empty cell or as ``NaN``.

The window matters. The SMMA is recursive, so its value at any bar depends on
every bar before it: if the export starts partway through the loaded history,
the leading rows will not match and no amount of tolerance helps. Scroll left
first, and this test will tell you if you did not.
"""
import csv
import math
from pathlib import Path

import pytest

import ta_py

FIXTURE = Path(__file__).parent / "fixtures" / "tradingview_alligator.csv"

# TradingView writes a fixed number of decimals, so this is the export's print
# precision, not algorithmic slack (the library's own tolerance is 1e-12).
TV_TOL = 1e-6

DEFAULTS = (13, 8, 5, 8, 5, 3)
JAW_FIRST_BAR = 13 - 1 + 8  # jaw length - 1 + jaw shift


def _column(rows, *names):
    """Return the export's column whose header matches one of ``names``."""
    for key in rows[0]:
        if key.strip().lower() in names:
            return [row[key].strip() for row in rows]
    return None


def _value(text):
    """The plotted value, or None where there is none.

    TradingView writes an undefined plot as an empty cell in a CSV export and as
    ``NaN`` in the Pine Logs panel, so both spellings mean "not defined here".
    """
    text = text.strip()
    if not text or text.lower() == "nan":
        return None
    return float(text)


def _fixture_rows():
    if not FIXTURE.exists():
        pytest.skip(f"no TradingView export at {FIXTURE} — see this module's docstring")
    with FIXTURE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, f"{FIXTURE} is empty"

    high = _column(rows, "high")
    low = _column(rows, "low")
    plots = [_column(rows, "jaw"), _column(rows, "teeth"), _column(rows, "lips")]
    if high is None or low is None or any(p is None for p in plots):
        pytest.fail(
            f"{FIXTURE} has columns {list(rows[0])}, so it is missing either the "
            "OHLC or the Jaw/Teeth/Lips study columns. Re-export with the "
            "Alligator on the chart; the export only carries studies that are "
            "in the Data Window."
        )
    return [float(h) for h in high], [float(l) for l in low], plots


def test_export_covers_the_full_history():
    """The export must begin where TradingView's own computation began.

    With the defaults the jaw is defined from bar 20 (13 - 1 + 8). A non-empty
    jaw in the first row means TradingView had earlier bars loaded that the
    export dropped, which makes every subsequent SMMA value incomparable.
    """
    _, _, (jaw, _, _) = _fixture_rows()
    filled = [i for i, v in enumerate(jaw) if _value(v) is not None]
    assert filled, "the export has no Jaw column values at all"
    assert filled[0] == JAW_FIRST_BAR, (
        f"jaw first appears at row {filled[0]}, expected {JAW_FIRST_BAR}: the "
        "export window does not match the loaded history, so the SMMA seeds differ"
    )


def test_jaw_teeth_lips_match_tradingview():
    high, low, plots = _fixture_rows()
    hl2 = [(h + l) / 2 for h, l in zip(high, low)]
    ours = ta_py.alligator(hl2, *DEFAULTS)
    assert len(ours) == len(hl2)

    compared = 0
    for col, (name, tv) in enumerate(zip(("jaw", "teeth", "lips"), plots)):
        for i, text in enumerate(tv):
            expected = _value(text)
            if expected is None:
                continue
            actual = ours[i][col]
            assert not math.isnan(actual), (
                f"{name}[{i}] is NaN here but TradingView plotted {expected}"
            )
            assert actual == pytest.approx(expected, abs=TV_TOL), (
                f"{name}[{i}]: ta.py {actual!r}, TradingView {expected!r}"
            )
            compared += 1
    assert compared > 0, "the export had no data points to compare"
