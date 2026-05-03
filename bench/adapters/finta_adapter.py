"""finta adapter (optional). finta takes a single OHLCV DataFrame for everything."""
from __future__ import annotations

try:
    import numpy as np
    import pandas as pd
    from finta import TA as F
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def _ohlc_df(data):
    a = np.asarray(data, dtype=np.float64)
    if a.ndim == 1:
        return pd.DataFrame({"open": a, "high": a, "low": a, "close": a, "volume": np.ones_like(a)})
    # data shape is [high, close, low] in ta_py atr/stoch convention
    return pd.DataFrame({"open": a[:, 1], "high": a[:, 0], "low": a[:, 2], "close": a[:, 1], "volume": np.ones(len(a))})


if AVAILABLE:
    def sma(data, length):       df = _ohlc_df(data); return lambda: F.SMA(df, period=length)
    def smma(data, length):      df = _ohlc_df(data); return lambda: F.SMMA(df, period=length)
    def wma(data, length):       df = _ohlc_df(data); return lambda: F.WMA(df, period=length)
    def ema(data, length):       df = _ohlc_df(data); return lambda: F.EMA(df, period=length)
    def std(data, length):       df = _ohlc_df(data); return lambda: F.STDDEV(df, period=length)
    def bands(data, length):     df = _ohlc_df(data); return lambda: F.BBANDS(df, period=length, std_multiplier=2)
    def bandwidth(data, length): df = _ohlc_df(data); return lambda: F.BBWIDTH(df, period=length)
    def rsi(data, length):       df = _ohlc_df(data); return lambda: F.RSI(df, period=length)
    def atr(data, length):       df = _ohlc_df(data); return lambda: F.ATR(df, period=length)
    def stoch(data, length):     df = _ohlc_df(data); return lambda: F.STOCH(df, period=length)
    def don(data, length):       df = _ohlc_df(data); return lambda: F.DO(df, upper_period=length, lower_period=length)
    def pr(data, length):        df = _ohlc_df(data); return lambda: F.WILLIAMS(df, period=length)
    def macd(data, length):      df = _ohlc_df(data); return lambda: F.MACD(df, period_fast=12, period_slow=26, signal=9)
    def keltner(data, length):   df = _ohlc_df(data); return lambda: F.KC(df, period=length)
    def variance(data, length):  return None  # not in finta
    def vwma(data, length):
        a = np.asarray(data, dtype=np.float64)
        df = pd.DataFrame({"open": a[:, 0], "high": a[:, 0], "low": a[:, 0], "close": a[:, 0], "volume": a[:, 1]})
        return lambda: F.VWAP(df)  # closest analog; flag in BENCHMARKS.md that this isn't a true VWMA
    def mfi(data, length):
        a = np.asarray(data, dtype=np.float64)
        df = pd.DataFrame({
            "open": a[:, 2], "high": a[:, 0], "low": a[:, 1],
            "close": a[:, 2], "volume": a[:, 3],
        })
        return lambda: F.MFI(df, period=length)
else:
    def __getattr__(name):
        return lambda *_a, **_kw: None
