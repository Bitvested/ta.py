"""Naïve pure-Python implementations — the floor every library must clear."""
from __future__ import annotations
import math


def _sma_naive(data, length):
    out = []
    for i in range(length, len(data) + 1):
        out.append(sum(data[i - length:i]) / length)
    return out


def _std_naive(data, length):
    win = data[-length:]
    m = sum(win) / length
    return math.sqrt(sum((x - m) ** 2 for x in win) / length)


def _bands_naive(data, length):
    out = []
    for i in range(length, len(data) + 1):
        win = data[i - length:i]
        m = sum(win) / length
        s = math.sqrt(sum((x - m) ** 2 for x in win) / length)
        out.append([m + 2 * s, m, m - 2 * s])
    return out


def sma(data, length):   return lambda: _sma_naive(data, length)
def std(data, length):   return lambda: _std_naive(data, length)
def bands(data, length): return lambda: _bands_naive(data, length)
# All other case names: not implemented → return None to mark "skip".
def smma(data, length):      return None
def wma(data, length):       return None
def vwma(data, length):      return None
def ema(data, length):       return None
def variance(data, length):  return None
def bandwidth(data, length): return None
def keltner(data, length):   return None
def don(data, length):       return None
def pr(data, length):        return None
def stoch(data, length):     return None
def rsi(data, length):       return None
def mfi(data, length):       return None
def atr(data, length):       return None
def macd(data, length):      return None
