# Migration Guide: ta.py 1.17.0 → 2.0.0

> This document is a working draft maintained across the v2.0 development cycle.
> The final canonical migration list ships with the 2.0.0 release.

## TL;DR

- **No public API breakage in Phase 1 of v2.0 development.** Every `from ta_py import <name>` from 1.17.0 still works, returns the same values, and accepts the same arguments.
- The internal file structure changed: `ta_py/ta.py` is now split into category modules. If you imported privately via `from ta_py.ta import <name>`, that still works in 1.x for one cycle but will emit a `DeprecationWarning` in a future release and be removed for 2.0.0 final.

## Breaking changes scheduled for 2.0.0 (from later phases)

The following will be implemented and documented in subsequent phases of the v2.0 overhaul. **None of these have landed yet.**

### Signature & output-shape normalizations (Phase 3 — match ta.js 2.0)

The following landed in Phase 3 pre-flight to match ta.js 2.0 algorithms exactly:

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

### Aroon namespace (Phase 3)

`aroon_up` / `aroon_down` / `aroon_osc` (still importable for one cycle) become the canonical `aroon.up` / `aroon.down` / `aroon.osc` namespace, mirroring ta.js. Both forms work during the deprecation window.

### Removed (Phase 5)

- `from ta_py.ta import <name>` shim — to be removed at 2.0.0 release. Use `from ta_py import <name>` instead.

## What you can do today

If you want to future-proof your imports against 2.0.0:

```python
# Preferred (works in 1.17.0 and 2.0.0)
from ta_py import sma, ema, rsi, macd

# Avoid (works in 1.17.0; removed in 2.0.0)
from ta_py.ta import sma, ema, rsi, macd
```
