# Migration Guide: ta.py 1.17.0 → 2.0.0

> This document is a working draft maintained across the v2.0 development cycle.
> The final canonical migration list ships with the 2.0.0 release.

## TL;DR

- **No public API breakage in Phase 1 of v2.0 development.** Every `from ta_py import <name>` from 1.17.0 still works, returns the same values, and accepts the same arguments.
- The internal file structure changed: `ta_py/ta.py` is now split into category modules. If you imported privately via `from ta_py.ta import <name>`, that still works in 1.x for one cycle but will emit a `DeprecationWarning` in a future release and be removed for 2.0.0 final.

## Breaking changes scheduled for 2.0.0 (from later phases)

The following will be implemented and documented in subsequent phases of the v2.0 overhaul. **None of these have landed yet.**

### Signature normalizations (Phase 3 — match ta.js 2.0)

To be filled in by Phase 3 by diffing ta.js v1.17.0 against ta.js v2.0.0 signatures and porting each change. Known candidates:
- `mfi` input tuple shape may change.
- `atr`, `smma` parameter naming may normalize.
- A few default values may be adjusted to match ta.js.

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
