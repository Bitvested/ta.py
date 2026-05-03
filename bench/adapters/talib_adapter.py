"""TA-Lib adapter (optional). Skips entire library if import fails."""
from __future__ import annotations

try:
    import numpy as np
    import talib
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def _arr(data):     return np.asarray(data, dtype=np.float64)
def _ohlc(data):
    a = np.asarray(data, dtype=np.float64)
    return a[:, 0], a[:, 1], a[:, 2]  # high, close, low (matches ta_py atr/stoch input shape)


if AVAILABLE:
    def sma(data, length):       a = _arr(data); return lambda: talib.SMA(a, timeperiod=length)
    def ema(data, length):       a = _arr(data); return lambda: talib.EMA(a, timeperiod=length)
    def wma(data, length):       a = _arr(data); return lambda: talib.WMA(a, timeperiod=length)
    def std(data, length):       a = _arr(data); return lambda: talib.STDDEV(a, timeperiod=length)
    def variance(data, length):  a = _arr(data); return lambda: talib.VAR(a, timeperiod=length)
    def bands(data, length):     a = _arr(data); return lambda: talib.BBANDS(a, timeperiod=length, nbdevup=2, nbdevdn=2)
    def rsi(data, length):       a = _arr(data); return lambda: talib.RSI(a, timeperiod=length)
    def atr(data, length):       h, c, l = _ohlc(data); return lambda: talib.ATR(h, l, c, timeperiod=length)
    def stoch(data, length):     h, c, l = _ohlc(data); return lambda: talib.STOCH(h, l, c, fastk_period=length, slowk_period=3, slowd_period=3)
    def macd(data, length):      a = _arr(data); return lambda: talib.MACD(a, fastperiod=12, slowperiod=26, signalperiod=9)
    # vwma, smma, bandwidth, keltner, don, pr, mfi: TA-Lib doesn't ship these → skip
    def smma(data, length):      return None
    def vwma(data, length):      return None
    def bandwidth(data, length): return None
    def keltner(data, length):   return None
    def don(data, length):       return None
    def pr(data, length):        return None
    def mfi(data, length):
        a = np.asarray(data, dtype=np.float64)
        h, lo, c, v = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        return lambda: talib.MFI(h, lo, c, v, timeperiod=length)
else:
    def __getattr__(name):
        return lambda *_a, **_kw: None
