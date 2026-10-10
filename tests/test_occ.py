"""Contract tests for the Open Close Cross.

Encodes the Pine this was ported from rather than the implementation:

    closeSeries = variant(basisType, close[delayOffset], basisLen, ...)
    openSeries  = variant(basisType, open[delayOffset],  basisLen, ...)
    xlong  = ta.crossover(closeSeries, openSeries)
    xshort = ta.crossunder(closeSeries, openSeries)

The point of most of these is the alignment contract: both bases come from the same
primitive over same-length inputs, so they must be defined on exactly the same bars,
and a delay must shift both by the same amount. A wrong warm-up assumption would
produce plausible numbers at the wrong bars, which is the failure this guards.
"""
import math
import random

import pytest

import ta_py
from tests.helpers import assert_close

DATA = [[10, 12], [11, 13], [12, 14], [13, 15], [14, 16], [15, 17]]

BASIS_TYPES = ("SMA", "EMA", "DEMA", "TEMA", "WMA", "SMMA", "HullMA", "TMA")


def series(n=200, seed=5):
    random.seed(seed)
    out = []
    price = 100.0
    for _ in range(n):
        o = price
        price = o + (random.random() - 0.5) * 2
        out.append([o, price])
    return out


def test_row_shape_and_alignment():
    for n in (6, 60, 200):
        data = series(n)
        out = ta_py.occ(data, basis_len=3, basis_type="SMA")
        assert len(out) == n
        assert all(len(row) == 3 for row in out)


def test_sma_basis_is_the_rolling_mean_by_hand():
    """basis_len=3 SMA, arithmetic done here rather than by the library."""
    out = ta_py.occ(DATA, basis_len=3, basis_type="SMA")
    opens = [r[0] for r in DATA]
    closes = [r[1] for r in DATA]
    for i in range(len(DATA)):
        if i < 2:
            assert all(math.isnan(v) for v in out[i])
            continue
        expected_close = sum(closes[i - 2:i + 1]) / 3
        expected_open = sum(opens[i - 2:i + 1]) / 3
        assert out[i][0] == pytest.approx(expected_close)
        assert out[i][1] == pytest.approx(expected_open)


def test_direction_is_whichever_basis_is_on_top():
    for basis in BASIS_TYPES:
        for row in ta_py.occ(series(120), basis_len=8, basis_type=basis):
            close_basis, open_basis, direction = row
            if math.isnan(close_basis):
                assert math.isnan(open_basis) and isinstance(direction, float)
                assert math.isnan(direction)
            else:
                assert direction == ("long" if close_basis > open_basis else "short")


def test_both_bases_are_defined_on_the_same_bars():
    """No basis type may leave one line lagging the other.

    The warm-up differs per type (DEMA needs 2*len-1 bars, TEMA 3*len-2, Hull its
    own), and the row is only usable where both lines exist, so the first defined
    row must have both.
    """
    for basis in BASIS_TYPES:
        for basis_len in (3, 8, 13):
            out = ta_py.occ(series(200), basis_len=basis_len, basis_type=basis)
            defined = [i for i, row in enumerate(out) if not math.isnan(row[0])]
            assert defined, f"{basis} len={basis_len}: nothing defined"
            first = defined[0]
            assert defined == list(range(first, len(out))), f"{basis}: gap in the output"
            for row in out[first:]:
                assert not math.isnan(row[1]), f"{basis}: close defined but open is not"


@pytest.mark.parametrize("basis", BASIS_TYPES)
def test_basis_matches_the_underlying_primitive(basis):
    """occ's basis must be that primitive, right-aligned to the last bar.

    This is what pins `_occ_line`'s assumption that a primitive's output ends at the
    final input bar, which is where the first-defined-bar arithmetic comes from.
    """
    data = series(200)
    out = ta_py.occ(data, basis_len=8, basis_type=basis)
    closes = [r[1] for r in data]
    opens = [r[0] for r in data]

    if basis == "TMA":
        expected_close = ta_py.sma(ta_py.sma(closes, 8), 8)
        expected_open = ta_py.sma(ta_py.sma(opens, 8), 8)
    else:
        fn = getattr(ta_py, {"HullMA": "hull"}.get(basis, basis.lower()))
        expected_close = fn(closes, 8)
        expected_open = fn(opens, 8)

    n = len(data)
    first = n - len(expected_close)
    assert first > 0, "fixture too short to be a meaningful check"
    for i in range(first, n):
        assert out[i][0] == pytest.approx(expected_close[i - first]), f"{basis} close at {i}"
        assert out[i][1] == pytest.approx(expected_open[i - first]), f"{basis} open at {i}"


def test_delay_shifts_both_bases_right_by_exactly_delay():
    """Pine's delayOffset, which is how R6 forces non-repainting."""
    data = series(120)
    for basis in BASIS_TYPES:
        base = ta_py.occ(data, basis_len=8, basis_type=basis, delay=0)
        delayed = ta_py.occ(data, basis_len=8, basis_type=basis, delay=3)
        for i, row in enumerate(delayed):
            if math.isnan(row[0]):
                continue
            assert row[0] == pytest.approx(base[i - 3][0]), f"{basis} at {i}"
            assert row[1] == pytest.approx(base[i - 3][1]), f"{basis} at {i}"
        # and the first 3 rows gain nothing
        assert all(math.isnan(v) for row in delayed[:3] for v in row[:2])


@pytest.mark.parametrize("basis", BASIS_TYPES)
def test_truncating_the_tail_does_not_change_earlier_rows(basis):
    """An independent check on alignment, and on causality.

    Every basis is a causal filter, so dropping the last bar cannot change any earlier
    row. This holds only if the output is right-aligned — the previous test derives
    its offset the same way the implementation does, so it cannot catch a systematic
    off-by-one on its own.
    """
    data = series(150)
    full = ta_py.occ(data, basis_len=8, basis_type=basis)
    cut = ta_py.occ(data[:-1], basis_len=8, basis_type=basis)
    assert len(cut) == len(full) - 1
    for i, row in enumerate(cut):
        for col in range(2):
            a, b = row[col], full[i][col]
            if math.isnan(a) or math.isnan(b):
                assert math.isnan(a) and math.isnan(b), f"{basis} at {i} col {col}"
            else:
                assert a == pytest.approx(b), f"{basis} at {i} col {col}"
        mine, theirs = row[2], full[i][2]
        if isinstance(mine, float) and math.isnan(mine):
            assert isinstance(theirs, float) and math.isnan(theirs), f"{basis} at {i} direction"
        else:
            assert mine == theirs, f"{basis} at {i} direction"


def test_delay_zero_is_the_default():
    data = series(80)
    assert_close(ta_py.occ(data, 8, "SMMA"), ta_py.occ(data, 8, "SMMA", 0), name="occ default delay")


def test_default_basis_is_smma_8():
    """The Pine's declared defaults: basisType SMMA, basisLen 8."""
    data = series(80)
    assert_close(ta_py.occ(data), ta_py.occ(data, 8, "SMMA"), name="occ defaults")


def test_bad_input_is_rejected():
    data = series(40)
    with pytest.raises(ValueError):
        ta_py.occ(data, basis_type="ALMA")
    with pytest.raises(ValueError):
        ta_py.occ(data, basis_type="LSMA")
    with pytest.raises(ValueError):
        ta_py.occ(data, basis_len=0)
    with pytest.raises(ValueError):
        ta_py.occ(data, delay=-1)


def test_data_shorter_than_the_basis_is_all_nan():
    out = ta_py.occ(series(4), basis_len=13)
    assert len(out) == 4
    assert all(math.isnan(v) for row in out for v in row)
