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
