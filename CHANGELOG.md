# Changelog

All notable changes to ta.py are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 2.0.0

### Added
- `pyproject.toml` (PEP 621) replaces `setup.py` as primary packaging metadata.
- Type hints on every public function.
- One-line docstrings on every public function.
- `ta_py._registry` cross-indicator dispatch dict, enabling runtime monkey-patching of internal calls (e.g., `ta_py._registry.registry["smma"] = my_fn` is picked up by `alligator`, `gator`, etc.).
- pytest-based test suite (`tests/test_*.py`) parameterized from `tests/parameters.py`.

### Changed
- `ta_py/ta.py` (1,309-line monolith) split into category modules: `moving_averages.py`, `statistics.py`, `indicators.py`, `oscillators.py`, `bands.py`, `chart_types.py`, `misc.py`, `experimental.py`, `random_.py`. All public symbols remain importable as `from ta_py import <name>`.
- `from ta_py.ta import <name>` continues to work in 1.x via a thin re-export shim (will be removed in 2.0.0 release).

### Fixed
- Removed duplicate `fisher` and `avgwin` definitions. Behavior unchanged: the surviving definitions are the ones that previously won the late-binding race in Python.

## [1.17.0] — 2026-04 (last 1.x release)

- Added: `times_down` function.
- Added: `times_up` function.

## [1.16.0]

- Added: universal divergence.
- Added: RSI Divergence.

## [1.15.0]

- Added: Relative Vigor Index.
- Added: Historical Volatility.

## [Prior versions]

See `git log` for the full history.
