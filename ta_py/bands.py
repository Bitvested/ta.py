"""Bands and channels."""
from .ta import std, atr
from .moving_averages import sma, vwma


def bands(data: list[float], l1: int = 14, l2: float = 1) -> list[list[float]]:
    """Bollinger Bands. Each row of output is [upper, middle, lower]."""
    pl = []; deviation = []; boll = [];
    sm = sma(data[:], l1);
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l1:
            devi = std(pl[:], l1);
            deviation.append(devi);
            pl = pl[1:];
    for i in range(len(sm)):
        boll.append([sm[i] + deviation[i] * l2, sm[i], sm[i] - deviation[i] * l2]);
    return boll;


def bandwidth(data: list[float], l1: int = 14, l2: float = 1) -> list[float]:
    """Bollinger Bandwidth (upper-lower). Calls bands() internally."""
    band = bands(data[:], l1, l2); boll = [];
    for i in range(len(band)): boll.append((band[i][0] - band[i][2]) / band[i][1]);
    return boll;


def keltner(data: list[list[float]], l1: int = 14, l2: float = 1) -> list[list[float]]:
    """Keltner Channels. Each row of data is [high, low, close]."""
    closing = []; tr = atr(data[:], l1); kelt = [];
    for i in range(len(data)): closing.append(sum(data[i]) / 3.0);
    kma = sma(closing, l1);
    tr = tr[(len(tr) - len(kma)):];
    for i in range(len(kma)): kelt.append([kma[i] + tr[i] * l2, kma[i], kma[i] - tr[i] * l2]);
    return kelt


def don(data: list[list[float]], l1: int = 20) -> list[list[float]]:
    """Donchian Channels. Each row of data is [high, low]."""
    pl = []; channel = [];
    for i in range(len(data)):
        pl.append(data[i]);
        if(len(pl) >= l1):
            highs = []; lows = [];
            for h in range(len(pl)):
                highs.append(float(pl[h][0]));
                lows.append(float(pl[h][1]));
            m = max(highs); l = min(lows);
            channel.append([m, (m+l) / 2.0, l]);
            pl = pl[1:];
    return channel;


def envelope(data: list[float], l1: int = 10, p: float = 0.005) -> list[list[float]]:
    """Percent envelope around an SMA."""
    enve = [];
    for i in range(l1, len(data)):
        sm = sma(data[i-l1:i], l1);
        enve.append([sm[0]*(1+p),sm[0],sm[0]*(1-p)]);
    return enve;


def fibbands(data: list[list[float]], length: int = 20, deviations: float = 3) -> list[list[float]]:
    """Fibonacci-spaced Bollinger-style bands. Each row of data is [price, volume]."""
    pl = []; deviation = []; ma = vwma(data, length); boll = [];
    for i in range(0, len(data)):
        pl.append(data[i][0]);
        if len(pl) >= length:
            devi = std(pl, length);
            deviation.append(devi*deviations);
            pl = pl[1:];
    for i in range(0, len(ma)):
        upper1 = ma[i] + (0.236 * deviation[i]);
        upper2 = ma[i] + (0.382 * deviation[i]);
        upper3 = ma[i] + (0.5 * deviation[i]);
        upper4 = ma[i] + (0.618 * deviation[i]);
        upper5 = ma[i] + (0.764 * deviation[i]);
        upper6 = ma[i] + deviation[i];
        lower1 = ma[i] - 0.236 * deviation[i];
        lower2 = ma[i] - 0.382 * deviation[i];
        lower3 = ma[i] - 0.5 * deviation[i];
        lower4 = ma[i] - 0.618 * deviation[i];
        lower5 = ma[i] - 0.764 * deviation[i];
        lower6 = ma[i] - deviation[i];
        boll.append([upper6, upper5, upper4, upper3, upper2, upper1, ma[i], lower1, lower2, lower3, lower4, lower5, lower6])
    return boll;
