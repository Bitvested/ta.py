# Contributing to ta.py

## Running the tests

```
pip install -e ".[dev]"
pytest
```

The full suite runs in well under a second. PRs should keep it green.

## Project layout

Source lives under `ta_py/`, split into category modules
(`moving_averages.py`, `statistics.py`, `indicators.py`, `oscillators.py`,
`bands.py`, `chart_types.py`, `misc.py`, `experimental.py`,
`random_.py`). The package's public surface is curated in
`ta_py/__init__.py`'s `__all__`.

Tests under `tests/` are fixture-driven — every (input, expected output)
pair lives in `tests/parameters.py`, and per-category test files iterate
over those fixtures using `tests/helpers.py::assert_close` for
float-tolerant comparisons.

## Cross-indicator calls go through `_registry`

When an indicator in one category file calls an indicator from a
**different** category file, dispatch through `ta_py._registry.registry`
rather than importing the function directly:

```python
# ta_py/indicators.py
from ._registry import registry

def macd(data, l1=12, l2=26):
    em1 = registry["ema"](data[:], l1)   # not: from .moving_averages import ema
    ...
```

Direct imports work, but they bypass user monkey-patches. Users can
swap any indicator implementation at runtime via:

```python
import ta_py
ta_py._registry.registry["ema"] = my_custom_ema
```

This is a documented feature (see `ta_py/_registry.py`). Cross-module
direct imports break it silently. Within a single file (e.g.,
`macd_signal` calling `macd` in `indicators.py`) a direct call is
fine — same-file functions aren't separately registered.

## Adding new indicators

ta.py mirrors [ta.js](https://github.com/Bartmr/ta.js) (kept locally at
`~/ta.js-1` in development setups). Reference the JS source at
`src/<category>/<name>.js` for algorithms, and copy the fixture from
`test/parameters.js` into `tests/parameters.py` so behavior stays in
parity. New functions also need to be added to:

1. Their category module (with type hints and a one-line docstring).
2. `ta_py/__init__.py`'s import block and `__all__` list.
3. The relevant `tests/test_<category>.py` `CASES` list.
