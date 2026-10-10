# Migration Guide: ta.py 1.17.0 → 2.0.0

## TL;DR

- **39 new public names** added to `ta_py.__all__` (was 112, now 151) — full functional parity with ta.js 2.0.
- **0 public APIs removed.**
- **6 behavioural breaks** — `atr`, `smma`, `mfi`, `rsi`, `wrsi` (now an alias of `rsi`), `stoch`. Each is documented below with a before/after example.
- **2 new namespaces** — `ta_py.random` and `ta_py.aroon` — alongside the existing flat exports.
- **Internal restructure** — the monolithic `ta_py/ta.py` was split into category modules in Phase 1. The shim that allowed `from ta_py.ta import <name>` is **removed in 2.0.0**.

If you only used `from ta_py import <name>` and didn't depend on the exact output values of `atr` / `smma` / `mfi` / `rsi` / `stoch`, your code keeps working.

## Public surface diff (auto-generated)

Captured by `scripts/snapshot_signatures.py`; baseline `tests/signatures_v1.json` (112 entries), post-Phase-3 `tests/signatures_v2.json` (151 entries). No signatures changed; all changes are additive at the signature level.

**Added (39 names):**

- *Moving averages (Batch A, 6):* `dema`, `tema`, `trima`, `t3`, `zlema`, `vidya`
- *Indicators (Batches B + C + D + E + F, 25):* `adl`, `nvi`, `pvi`, `emv`, `dpo`, `ulcer`, `pdm`, `mdm`, `pdi`, `mdi`, `dx`, `adx`, `adxr`, `natr`, `trix`, `ppo`, `apo`, `mass`, `cci`, `stoch_rsi`, `kdj`, `cmf`, `vortex`
- *Oscillators (Batches F + G, 2):* `kvo`, `ult`
- *Statistics (Batch H, 5):* `lr_slope`, `lr_intercept`, `lr_angle`, `tsf`, `std_series`
- *Constants (Batch B, 1):* `fibnumbers`
- *Namespaces (Batches I + J, 2):* `random` (`prng` / `pick` / `range` / `float` / `order`), `aroon` (`up` / `down` / `osc`)

**Removed:** none.

**Signatures changed:** none.

## Behavioural breaks (algorithm or output-shape changes)

All six landed in Phase 3 to match ta.js 2.0 algorithms exactly. Each has a regenerated test fixture against ta.js 2.0 reference values.

#### `atr(data, length=14)` — output shape and true-range fix

The true-range terms `H − Cprev` and `L − Cprev` now use absolute values (matching Wilder's original definition), and the output drops the first `length − 1` transient values so the result aligns with `sma(data, length)` (length `n − length + 1`).

```python
# 1.17.0
ta_py.atr([[3, 2, 1], [2, 2, 1], [4, 3, 1], [2, 2, 1]], 3)
# → [2.0, 1.6666666666666667, 2.111111111111111, 1.7407407407407407]   # 4 rows, no abs

# 2.0.0
ta_py.atr([[3, 2, 1], [2, 2, 1], [4, 3, 1], [2, 2, 1]], 3)
# → [2.111111111111111, 2.074074074074074]                              # 2 rows, abs applied
```

Cascading effects: `keltner` (now uses correct ATR), `supertrend` (output values shift on the second row onward), `halftrend` (loop range tightened to match ta.js, atr-derived deviation values shift). All callers updated.

#### `smma(data, length=14)` — Wilder smoothing

Replaced an idiosyncratic `(sliding_sum − prev) / length` recurrence with the standard Wilder smoothing `prev = prev + (data[i] − prev) / length`, seeded from `sma(data[:length], length)`. Output now includes the SMA seed, so the result is `length n − length + 1` rows (was `n − length`).

```python
# 1.17.0
ta_py.smma([1, 2, 3, 4, 5, 6, 10], 5)
# → [3.4, 4.92]              # 2 rows; non-Wilder recurrence

# 2.0.0
ta_py.smma([1, 2, 3, 4, 5, 6, 10], 5)
# → [3, 3.6, 4.88]            # 3 rows; Wilder recurrence seeded from SMA
```

Cascading effects: `alligator` and `gator` outputs change. Both indicators continue to align via existing slice-trim logic; no caller code change required.

#### `mfi(data, length=14)` — input tuple shape

Inputs are now OHLCV rows `[high, low, close, volume]` (matching ta.js 2.0). Previously each row was a precomputed `[positive_flow, negative_flow]` pair. Internally, typical price `(H + L + C) / 3` and money flow are now derived per bar; positive/negative flows are tracked separately and the `length`-window ratio uses the standard `100 × pos / (pos + neg)` formula.

```python
# 1.17.0 — caller had to precompute pos/neg flows
ta_py.mfi([[19, 13], [14, 38], [21, 25], [32, 17]], 3)
# → [41.53846153846154, 45.578231292517]

# 2.0.0 — pass raw OHLCV
ta_py.mfi([[10, 8, 9, 100], [12, 10, 11, 150], [11, 9, 10, 200], [13, 11, 12, 120], [12, 10, 11, 80]], 3)
# → [60.70726915520628, 33.333333333333336]
```

#### `rsi(data, length=14)` — Wilder smoothing

`rsi` previously used Cutler RSI (a strict sliding window of the last `length - 1` deltas). It now uses Wilder smoothing — gains and losses are seeded as the mean of the first `length` raw values, then smoothed via the recurrence `avg = (avg * (length - 1) + cur) / length`. This matches ta.js 2.0's `rsi` exactly and aligns with TA-Lib's RSI / Wilder's original definition.

`wrsi` is now a thin alias for `rsi` (same as ta.js 2.0). Output length drops by one (was `n - length + 1`, now `n - length`).

```python
# 1.17.0 (Cutler / sliding window)
ta_py.rsi([1, 2, 3, 4, 5, 6, 7, 5], 6)
# → [100.0, 100.0, 66.66666666666666]    # 3 rows

# 2.0.0 (Wilder smoothing)
ta_py.rsi([1, 2, 3, 4, 5, 6, 7, 5, 6, 4], 6)
# → [100, 71.42857142857143, 75.60975609756098, 55.95667870036102]   # n - length rows
```

If you want the old Cutler-style RSI, the simplest substitute is to copy the 1.x rsi source out of git history into your own module — there's no equivalent in ta.py 2.0.

#### `stoch(data, length=14, sd=3, sk=3)` — rolling window matches ta.js 2.0

`stoch` previously used an extended rolling-max/min window of `length + sk + sd - 1` bars, producing fewer output rows than ta.js. It now uses just `length` bars for the high/low window (matching ta.js 2.0 and TA-Lib), and stacks the `sk`-bar SMA over raw %K plus the `sd`-bar SMA over smoothed %K independently. The function also auto-swaps `length`/`sd` and clamps `sk ≤ sd` (mirroring ta.js's lenient parameter handling).

Output length increases by `sk + sd - 2` rows compared to 1.x.

```python
# 1.17.0 (extended window)
ta_py.stoch([[3,2,1],[2,2,1],[4,3,1],[2,2,1]], 2, 1, 1)
# → [[66.667, 66.667], [33.333, 33.333]]   # 2 rows

# 2.0.0 (length-only window, ta.js parity)
ta_py.stoch([[3,2,1],[2,2,1],[4,3,1],[2,2,1]], 2, 1, 1)
# → [[50, 50], [66.667, 66.667], [33.333, 33.333]]   # 3 rows
```

## Breaks landing after 2.0.0

### `alligator(data, ...)` / `gator(data, ...)` — rows now align to the input bars

Both returned their rows newest-first and truncated to `n - jl - js + 2` rows, so the
array could not be indexed by bar and could not be lined up with the series it was
computed from. Each line was also displaced by `shift - 1` bars instead of `shift`,
`gator`'s upper histogram had the wrong sign (it plotted `jaw - teeth`, not
`|jaw - teeth|`), and a `shift` of `0` raised `IndexError` because `jaw[i - (js - 1)]`
with `js = 0` indexes past the end of the SMMA output.

This shape is old — the 1.17.0 body is character-for-character the same — so the
break is against `1.x` and `2.0.0` alike. Both now return one row per input bar, with
each line displaced right by its own shift, matching TradingView's
`plot(..., offset = shift)`; a line is NaN until bar `length - 1 + shift`.

```python
data = [8, 7, 8, 9, 7, 8, 9, 6, 7, 8, 6, 8, 10, 8, 7, 9, 8, 7, 9, 6, 7, 9]

# 1.x / 2.0.0 — 3 rows, newest first, not aligned to `data`
ta_py.alligator(data)
# → [[7.726445152480656, 7.813665029359981, 7.6378735707815935],
#    [7.78698224852071, 7.929902890697122, 8.047341963476992],
#    [7.769230769230769, 7.919889017939568, 7.809177454346241]]

# now — 22 rows, one per bar of `data`
ta_py.alligator(data)
# → [[nan, nan, nan], ... 19 more ...,
#    [7.769230769230769, 7.919889017939568, 7.809177454346241],
#    [7.78698224852071, 7.929902890697122, 8.047341963476992]]
```

Every row the old array held except its first is still present, reversed:
`old[r] == new[len(data) - r]` for `r >= 1`. The first row is the one value the
aligned output cannot represent — it held the jaw computed at bar `n - js`, one bar
newer than the newest jaw an offset-aligned series can show. If you need the old
shape back:

```python
def alligator_20(data, jl=13, tl=8, ll=5, js=8, ts=5, ls=3):
    """The 1.x / 2.0.0 return shape, recovered from the aligned output.

    Drops the old array's first row, which has no counterpart here.
    """
    out = ta_py.alligator(data, jl, tl, ll, js, ts, ls)
    n = len(out)
    return [out[n - r] for r in range(1, n - jl - js + 2)]
```

Most callers do not need that. `out[0]` used to be the newest row and is now bar 0;
the jaw/teeth/lips of bar `i` finally sit together in `out[i]`, so the array indexes
by bar directly instead of needing manual realignment. Note also that the jaw is
deliberately 8 bars stale by the time it is drawn — that is what the offset means,
and `out[-1][0]` is the jaw from bar `n - 9`, not the latest SMMA value.

`gator` is now derived from `alligator`'s displaced lines, so it takes the same
alignment, and `js=ts=ls=0` — previously a crash — reproduces TradingView's Gator
Oscillator, which does not displace its lines.

## New namespaces (additive — no break)

### `ta_py.random`

Five seeded random helpers grouped under one namespace, mirroring ta.js 2.0's `random` module:

```python
from ta_py import random
r = random.prng("seed")           # deterministic [0, 1) generator
random.pick([1, 2, 3], r)         # random element
random.range(0, 10, r)            # random int in [ceil(min), floor(max))
random.float(1.5, 2.5, r)         # random float in [min, max)
random.order([1, 2, 3, 4, 5], r)  # shuffled list via repeated picks
```

`prng` remains flat-exported as `ta_py.prng` (unchanged from 1.x). The other four are *not* flat-exported because `range` / `float` / `pick` / `order` would shadow Python builtins under `from ta_py import *`.

### `ta_py.aroon`

```python
from ta_py import aroon
aroon.up(data, length)
aroon.down(data, length)
aroon.osc(data, length)
```

Flat `aroon_up` / `aroon_down` / `aroon_osc` continue to work in 2.x as silent aliases (no `DeprecationWarning`).

## Removed in 2.0.0

- `ta_py/ta.py` shim. `from ta_py.ta import <name>` no longer works. Use `from ta_py import <name>` instead — the canonical import has been the same since 1.x.
