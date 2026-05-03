"""Bands and channels."""
from ._registry import registry


def bands(data: list[float], l1: int = 14, l2: float = 1) -> list[list[float]]:
    """Bollinger Bands. Each row of output is [upper, middle, lower]."""
    import math
    n = len(data)
    if n < l1:
        return []
    out: list[list[float]] = []
    sum_x = 0.0
    sum_x2 = 0.0
    for i in range(l1):
        v = float(data[i])
        sum_x += v
        sum_x2 += v * v

    def push(sum_x: float, sum_x2: float) -> None:
        mean = sum_x / l1
        var = sum_x2 / l1 - mean * mean
        sd = math.sqrt(var) if var > 0 else 0.0
        out.append([mean + sd * l2, mean, mean - sd * l2])

    push(sum_x, sum_x2)
    for i in range(l1, n):
        v_in = float(data[i])
        v_out = float(data[i - l1])
        sum_x += v_in - v_out
        sum_x2 += v_in * v_in - v_out * v_out
        push(sum_x, sum_x2)
    return out


def bandwidth(data: list[float], l1: int = 14, l2: float = 1) -> list[float]:
    """Bollinger Bandwidth (upper-lower). Calls bands() internally."""
    band = bands(data[:], l1, l2); boll = [];
    for i in range(len(band)): boll.append((band[i][0] - band[i][2]) / band[i][1]);
    return boll;


def keltner(data: list[list[float]], l1: int = 14, l2: float = 1) -> list[list[float]]:
    """Keltner Channels. Each row of data is [high, low, close]."""
    closing = []; tr = registry["atr"](data[:], l1); kelt = [];
    for i in range(len(data)): closing.append(sum(data[i]) / 3.0);
    kma = registry["sma"](closing, l1);
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
        sm = registry["sma"](data[i-l1:i], l1);
        enve.append([sm[0]*(1+p),sm[0],sm[0]*(1-p)]);
    return enve;


def fibbands(data: list[list[float]], length: int = 20, deviations: float = 3) -> list[list[float]]:
    """Fibonacci-spaced Bollinger-style bands. Each row of data is [price, volume]."""
    pl = []; deviation = []; ma = registry["vwma"](data, length); boll = [];
    for i in range(0, len(data)):
        pl.append(data[i][0]);
        if len(pl) >= length:
            devi = registry["std"](pl, length);
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
