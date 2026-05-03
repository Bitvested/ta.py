"""ta_py adapter — the library under test."""
from __future__ import annotations
import ta_py


def sma(data, length):       return lambda: ta_py.sma(data, length)
def smma(data, length):      return lambda: ta_py.smma(data, length)
def wma(data, length):       return lambda: ta_py.wma(data, length)
def vwma(data, length):      return lambda: ta_py.vwma(data, length)
def ema(data, length):       return lambda: ta_py.ema(data, length)
def variance(data, length):  return lambda: ta_py.variance(data, length)
def std(data, length):       return lambda: ta_py.std(data, length)
def bands(data, length):     return lambda: ta_py.bands(data, length, 2)
def bandwidth(data, length): return lambda: ta_py.bandwidth(data, length, 2)
def keltner(data, length):   return lambda: ta_py.keltner(data, length, 1)
def don(data, length):       return lambda: ta_py.don(data, length)
def pr(data, length):        return lambda: ta_py.pr(data, length)
def stoch(data, length):     return lambda: ta_py.stoch(data, length, 3, 3)
def rsi(data, length):       return lambda: ta_py.rsi(data, length)
def mfi(data, length):       return lambda: ta_py.mfi(data, length)
def atr(data, length):       return lambda: ta_py.atr(data, length)
def macd(data, length):      return lambda: ta_py.macd(data, 12, 26)
