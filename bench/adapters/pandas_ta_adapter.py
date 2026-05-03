"""pandas-ta adapter (optional)."""
from __future__ import annotations

try:
    import numpy as np
    import pandas as pd
    import pandas_ta as pta
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def _series(data):  return pd.Series(np.asarray(data, dtype=np.float64))
def _ohlc_df(data):
    a = np.asarray(data, dtype=np.float64)
    return pd.DataFrame({"high": a[:, 0], "close": a[:, 1], "low": a[:, 2]})


if AVAILABLE:
    def sma(data, length):       s = _series(data); return lambda: pta.sma(s, length=length)
    def smma(data, length):      s = _series(data); return lambda: pta.rma(s, length=length)
    def wma(data, length):       s = _series(data); return lambda: pta.wma(s, length=length)
    def ema(data, length):       s = _series(data); return lambda: pta.ema(s, length=length)
    def variance(data, length):  s = _series(data); return lambda: pta.variance(s, length=length)
    def std(data, length):       s = _series(data); return lambda: pta.stdev(s, length=length)
    def bands(data, length):     s = _series(data); return lambda: pta.bbands(s, length=length, std=2)
    def rsi(data, length):       s = _series(data); return lambda: pta.rsi(s, length=length)
    def atr(data, length):       df = _ohlc_df(data); return lambda: pta.atr(df["high"], df["low"], df["close"], length=length)
    def stoch(data, length):     df = _ohlc_df(data); return lambda: pta.stoch(df["high"], df["low"], df["close"], k=length, d=3, smooth_k=3)
    def don(data, length):       df = _ohlc_df(data); return lambda: pta.donchian(df["high"], df["low"], lower_length=length, upper_length=length)
    def pr(data, length):
        s = _series(data)
        df = pd.DataFrame({"high": s, "low": s, "close": s})
        return lambda: pta.willr(df["high"], df["low"], df["close"], length=length)
    def macd(data, length):      s = _series(data); return lambda: pta.macd(s, fast=12, slow=26, signal=9)
    def vwma(data, length):
        a = np.asarray(data, dtype=np.float64)
        price = pd.Series(a[:, 0]); vol = pd.Series(a[:, 1])
        return lambda: pta.vwma(price, vol, length=length)
    def bandwidth(data, length): return None  # pandas-ta returns it inside bbands, no isolated call
    def keltner(data, length):   df = _ohlc_df(data); return lambda: pta.kc(df["high"], df["low"], df["close"], length=length)
    def mfi(data, length):
        a = np.asarray(data, dtype=np.float64)
        h = pd.Series(a[:, 0]); lo = pd.Series(a[:, 1])
        c = pd.Series(a[:, 2]); v = pd.Series(a[:, 3])
        return lambda: pta.mfi(h, lo, c, v, length=length)
else:
    def __getattr__(name):
        return lambda *_a, **_kw: None
