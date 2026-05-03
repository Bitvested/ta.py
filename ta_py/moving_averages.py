"""Moving averages and smoothers."""
from ._registry import registry


def sma(data: list[float], l: int = 14) -> list[float]:
    """Simple Moving Average."""
    n = len(data)
    if n < l:
        return []
    s = 0.0
    for i in range(l):
        s += float(data[i])
    out = [s / l]
    for i in range(l, n):
        s += float(data[i]) - float(data[i - l])
        out.append(s / l)
    return out

def smma(data: list[float], l: int = 14) -> list[float]:
    """Smoothed Moving Average (Wilder smoothing seeded from SMA).

    Output length is ``n - l + 1`` (matching ta.js 2.0).
    """
    n = len(data)
    if n < l:
        return []
    seed = 0.0
    for i in range(l):
        seed += float(data[i])
    prev = seed / l
    out = [prev]
    for i in range(l, n):
        prev = prev + (float(data[i]) - prev) / l
        out.append(prev)
    return out

def wma(data: list[float], l: int = 14) -> list[float]:
    """Weighted Moving Average (linear weights 1..l)."""
    n = len(data)
    if n < l:
        return []
    total_weight = l * (l + 1) // 2
    wsum = 0.0
    pwsum = 0.0
    for i in range(l):
        v = float(data[i])
        wsum += v
        pwsum += v * (i + 1)
    out = [pwsum / total_weight]
    for i in range(l, n):
        v = float(data[i])
        outgoing = float(data[i - l])
        pwsum += l * v - wsum
        wsum += v - outgoing
        out.append(pwsum / total_weight)
    return out

def wsma(data: list[float], l: int = 14) -> list[float]:
    """Wilder's Smoothing Moving Average."""
    n = len(data)
    if n < l:
        return []
    seed = 0.0
    for i in range(l):
        seed += float(data[i])
    out = [seed / l]
    weight = 1.0 / l
    for j in range(l, n):
        prev = out[-1]
        out.append((float(data[j]) - prev) * weight + prev)
    return out

def pwma(data: list[float], l: int = 14) -> list[float]:
    """Parabolic Weighted Moving Average."""
    weights: list[int] = []
    weight = 0
    i = l / 2
    b = l
    while i >= 1:
        if i % 1 != 0:
            i = round(i)
            weight += i * b
        else:
            weights.append(int(i * b))
            weight += int(i * b * 2)
        weights.insert(0, int(i * b))
        i -= 1
        b -= 1
    out: list[float] = []
    n = len(data)
    for j in range(l, n + 1):
        average = 0.0
        base = j - l
        for x in range(len(weights)):
            average += float(data[base + x]) * weights[x] / weight
        out.append(average)
    return out

def hwma(data: list[float], l: int = 14) -> list[float]:
    """Hyperbolic Weighted Moving Average."""
    weights: list[int] = []
    weight = 0
    i = 1
    b = l
    while i <= l / 2:
        if i % 1 != 0:
            i = round(i)
            weight += i * b
        else:
            weights.append(int(i * b))
            weight += int(i * b * 2)
        weights.insert(0, int(i * b))
        i += 1
        b -= 1
    out: list[float] = []
    n = len(data)
    for j in range(l, n + 1):
        average = 0.0
        base = j - l
        for x in range(len(weights)):
            average += float(data[base + x]) * weights[x] / weight
        out.append(average)
    return out

def ema(data: list[float], l: int = 12) -> list[float]:
    """Exponential Moving Average."""
    pl = []; em = []; weight = 2 / (float(l) + 1);
    for i in range(len(data)):
        pl.append(float(data[i]));
        if (len(pl)) >= l:
            f = sum(pl) / l if len(em) <= 0 else (data[i] - em[len(em) - 1]) * weight + em[len(em) - 1]
            em.append(f);
            pl = pl[1:];
    return em;

def vwma(data: list[list[float]], l: int = 20) -> list[float]:
    """Volume Weighted Moving Average. Each row of data is [price, volume]."""
    n = len(data)
    if n < l:
        return []
    sum_pv = 0.0
    sum_v = 0.0
    for i in range(l):
        p = float(data[i][0]); v = float(data[i][1])
        sum_pv += p * v
        sum_v  += v
    out = [sum_pv / sum_v]
    for i in range(l, n):
        p_in  = float(data[i][0]);     v_in  = float(data[i][1])
        p_out = float(data[i - l][0]); v_out = float(data[i - l][1])
        sum_pv += p_in * v_in - p_out * v_out
        sum_v  += v_in - v_out
        out.append(sum_pv / sum_v)
    return out

def vwwma(d: list[list[float]], length: int = 20) -> list[float]:
    """Volume Weighted Weighted Moving Average. Each row of data is [price, volume]."""
    n = len(d)
    weight = length * (length + 1) // 2
    out: list[float] = []
    for i in range(length, n + 1):
        totalv = 0.0
        totalp = 0.0
        base = i - length
        for q in range(length):
            row = d[base + q]
            v = float(row[1])
            totalv += v * (q + 1) / weight
            totalp += float(row[0]) * v * (q + 1) / weight
        out.append(totalp / totalv)
    return out

def hull(data: list[float], l: int = 14) -> list[float]:
    """Hull Moving Average. Calls wma() internally (same module)."""
    pl = []; hma = []; ewma = wma(data[:], l); sqn = round(l**(1.0/2.0));
    first = wma(data[:], int(round(l / 2)));
    first = first[-(len(ewma)-len(first)):];
    for i in range(len(ewma)):
        pl.append((first[i] * 2) - ewma[i]);
        if (len(pl)) >= sqn:
            h = wma(pl[:], int(sqn));
            hma.append(h[len(h) - 1]);
            pl = pl[1:];
    return hma;

def kama(data: list[float], l1: int = 10, l2: int = 2, l3: int = 30) -> list[float]:
    """Kaufman Adaptive Moving Average."""
    ka = sma(data[:], l1); ka = [ka[len(ka)-1]];
    for i in range(l1 + 1, len(data)):
        vola = 0; change = abs(data[i] - data[i - l1]);
        for a in range(1, l1):
            vola += abs(float(data[i-a]) - float(data[(i-a)-1]));
        sc = (change/vola * (2.0/(l2+1.0) - 2.0/(l3+1.0) + 2.0/(l3+1.0))) ** 2.0;
        ka.append(ka[len(ka)-1] + sc * (float(data[i]) - ka[len(ka)-1]));
    return ka;

def cwma(data: list[float], weights: list[float]) -> list[float]:
    """Custom Weighted Moving Average. Window length = len(weights)."""
    L = len(weights)
    n = len(data)
    weight_sum = 0.0
    for w in weights:
        weight_sum += float(w)
    out: list[float] = []
    for i in range(L, n + 1):
        s = 0.0
        base = i - L
        for q in range(L):
            s += float(data[base + q]) * float(weights[q])
        out.append(s / weight_sum)
    return out

def dema(data: list[float], length: int = 30) -> list[float]:
    """Double Exponential Moving Average.

    Streaming dual-EMA: ``2*EMA1 − EMA2`` where ``EMA2`` is seeded from the
    average of the first ``length`` ``EMA1`` values. Output length is
    ``n − 2*length + 2``.
    """
    n = len(data)
    if n < 2 * length - 1:
        return []
    alpha = 2.0 / (length + 1.0)
    s = 0.0
    for i in range(length):
        s += float(data[i])
    e1 = s / length
    sum_e1 = e1
    for i in range(length, 2 * length - 1):
        e1 = (float(data[i]) - e1) * alpha + e1
        sum_e1 += e1
    e2 = sum_e1 / length
    out = [2.0 * e1 - e2]
    for i in range(2 * length - 1, n):
        e1 = (float(data[i]) - e1) * alpha + e1
        e2 = (e1 - e2) * alpha + e2
        out.append(2.0 * e1 - e2)
    return out


def tema(data: list[float], length: int = 30) -> list[float]:
    """Triple Exponential Moving Average (``3*EMA1 − 3*EMA2 + EMA3``)."""
    e1 = ema(data, length)
    if len(e1) < length:
        return []
    e2 = ema(e1, length)
    if len(e2) < length:
        return []
    e3 = ema(e2, length)
    off1 = len(e1) - len(e3)
    off2 = len(e2) - len(e3)
    return [3.0 * e1[i + off1] - 3.0 * e2[i + off2] + e3[i] for i in range(len(e3))]


def trima(data: list[float], length: int = 30) -> list[float]:
    """Triangular Moving Average (SMA of SMA)."""
    if length % 2 == 0:
        n_inner = length // 2 + 1
        n_outer = length // 2
    else:
        n_inner = n_outer = (length + 1) // 2
    return sma(sma(data, n_inner), n_outer)


def t3(data: list[float], length: int = 5, vfactor: float = 0.7) -> list[float]:
    """T3 Moving Average (Tillson). Six chained EMAs with polynomial blend."""
    e1 = ema(data, length)
    if len(e1) < length:
        return []
    e2 = ema(e1, length)
    if len(e2) < length:
        return []
    e3 = ema(e2, length)
    if len(e3) < length:
        return []
    e4 = ema(e3, length)
    if len(e4) < length:
        return []
    e5 = ema(e4, length)
    if len(e5) < length:
        return []
    e6 = ema(e5, length)
    v2 = vfactor * vfactor
    v3 = v2 * vfactor
    c1 = -v3
    c2 = 3.0 * v2 + 3.0 * v3
    c3 = -6.0 * v2 - 3.0 * vfactor - 3.0 * v3
    c4 = 1.0 + 3.0 * vfactor + v3 + 3.0 * v2
    o3 = len(e3) - len(e6)
    o4 = len(e4) - len(e6)
    o5 = len(e5) - len(e6)
    return [
        c1 * e6[i] + c2 * e5[i + o5] + c3 * e4[i + o4] + c4 * e3[i + o3]
        for i in range(len(e6))
    ]


def zlema(data: list[float], length: int = 14) -> list[float]:
    """Zero-Lag Exponential Moving Average. Output length matches input."""
    n = len(data)
    if n == 0:
        return []
    lag = (length - 1) // 2
    alpha = 2.0 / (length + 1.0)
    prev = float(data[0])
    out = [prev]
    for i in range(1, n):
        adj = 2.0 * float(data[i]) - float(data[i - lag]) if i >= lag else float(data[i])
        prev = alpha * adj + (1.0 - alpha) * prev
        out.append(prev)
    return out


def vidya(data: list[float], short_length: int = 2, long_length: int = 5, alpha: float = 0.2) -> list[float]:
    """Variable Index Dynamic Average (volatility-adaptive EMA)."""
    n = len(data)
    if n < long_length:
        return []
    _std = registry["std"]
    prev = float(data[long_length - 2])
    out = [prev]
    for i in range(long_length - 1, n):
        s_short = _std(data[i - short_length + 1:i + 1], short_length)
        s_long = _std(data[i - long_length + 1:i + 1], long_length)
        k = 0.0 if s_long == 0 else s_short / s_long
        prev = alpha * k * float(data[i]) + (1.0 - alpha * k) * prev
        out.append(prev)
    return out


def lsma(data: list[float], l1: int = 25) -> list[float]:
    """Least Squares (linear regression) Moving Average."""
    pl = []; lr = [];
    for i in range(len(data)):
        pl.append(float(data[i]));
        if(len(pl) >= l1):
            sum_x = 0.0; sum_y = 0.0; sum_xy = 0.0; sum_xx = 0.0; sum_yy = 0.0;
            for a in range(1, len(pl)+1):
                sum_x += a;
                sum_y += pl[a-1];
                sum_xy += (pl[a-1] * a);
                sum_xx += (a*a);
                sum_yy += (pl[a-1] * pl[a-1]);
            m = ((sum_xy - sum_x * sum_y / l1) / (sum_xx - sum_x * sum_x / l1));
            b = sum_y / l1 - m * sum_x / l1;
            lr.append(m * l1 + b);
            pl = pl[1:];
    return lr;
