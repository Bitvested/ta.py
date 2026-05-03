"""Benchmark case definitions. Each case = one function × one input shape."""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Any

from adapters import (
    ta_py_adapter,
    stdlib_baseline,
    talib_adapter,
    pandas_ta_adapter,
    finta_adapter,
)


def _close_series(size: int, seed: int = 42) -> list[float]:
    """Geometric-Brownian-ish close prices, deterministic."""
    rng = random.Random(seed)
    p = 100.0
    out = [p]
    for _ in range(size - 1):
        p *= math.exp(rng.gauss(0, 0.01))
        out.append(p)
    return out


def _ohlc_series(size: int, seed: int = 42) -> list[list[float]]:
    """[high, close, low] rows (matches ta_py atr/stoch convention)."""
    rng = random.Random(seed)
    rows: list[list[float]] = []
    p = 100.0
    for _ in range(size):
        c = p * math.exp(rng.gauss(0, 0.01))
        h = max(p, c) * (1 + abs(rng.gauss(0, 0.005)))
        lo = min(p, c) * (1 - abs(rng.gauss(0, 0.005)))
        rows.append([h, c, lo])
        p = c
    return rows


def _price_volume_series(size: int, seed: int = 42) -> list[list[float]]:
    """[price, volume] rows for vwma."""
    closes = _close_series(size, seed)
    rng = random.Random(seed + 1)
    return [[c, abs(rng.gauss(1_000, 200))] for c in closes]


def _hlcv_series(size: int, seed: int = 42) -> list[list[float]]:
    """[high, low, close, volume] rows for mfi (ta_py 2.0 signature)."""
    rng = random.Random(seed)
    rows: list[list[float]] = []
    p = 100.0
    for _ in range(size):
        c = p * math.exp(rng.gauss(0, 0.01))
        h = max(p, c) * (1 + abs(rng.gauss(0, 0.005)))
        lo = min(p, c) * (1 - abs(rng.gauss(0, 0.005)))
        v = abs(rng.gauss(1_000, 200))
        rows.append([h, lo, c, v])
        p = c
    return rows


@dataclass
class Case:
    name: str
    description: str
    generate: Callable[[int], Any]
    length: int
    adapters: dict = field(default_factory=dict)


def _adapters_for(name: str) -> dict:
    return {
        "ta_py":      getattr(ta_py_adapter,      name, lambda *_a: None),
        "stdlib":     getattr(stdlib_baseline,    name, lambda *_a: None),
        "talib":      getattr(talib_adapter,      name, lambda *_a: None),
        "pandas_ta":  getattr(pandas_ta_adapter,  name, lambda *_a: None),
        "finta":      getattr(finta_adapter,      name, lambda *_a: None),
    }


CASES: list[Case] = [
    Case("sma",       "Simple Moving Average",            _close_series,        14, _adapters_for("sma")),
    Case("smma",      "Smoothed Moving Average",          _close_series,        14, _adapters_for("smma")),
    Case("wma",       "Weighted Moving Average",          _close_series,        14, _adapters_for("wma")),
    Case("vwma",      "Volume-Weighted MA",               _price_volume_series, 20, _adapters_for("vwma")),
    Case("ema",       "Exponential MA",                   _close_series,        14, _adapters_for("ema")),
    Case("variance",  "Rolling variance",                 _close_series,        14, _adapters_for("variance")),
    Case("std",       "Standard deviation (window)",      _close_series,        14, _adapters_for("std")),
    Case("bands",     "Bollinger Bands (length=14, k=2)", _close_series,        14, _adapters_for("bands")),
    Case("bandwidth", "Bollinger Bandwidth",              _close_series,        14, _adapters_for("bandwidth")),
    Case("keltner",   "Keltner Channels",                 _ohlc_series,         14, _adapters_for("keltner")),
    Case("don",       "Donchian Channels",                _ohlc_series,         20, _adapters_for("don")),
    Case("pr",        "Williams %R",                      _close_series,        14, _adapters_for("pr")),
    Case("stoch",     "Stochastic Oscillator (14,3,3)",   _ohlc_series,         14, _adapters_for("stoch")),
    Case("rsi",       "RSI (length=14)",                  _close_series,        14, _adapters_for("rsi")),
    Case("mfi",       "Money Flow Index (HLCV)",          _hlcv_series,         14, _adapters_for("mfi")),
    Case("atr",       "Average True Range",               _ohlc_series,         14, _adapters_for("atr")),
    Case("macd",      "MACD (12,26)",                     _close_series,        12, _adapters_for("macd")),
]
