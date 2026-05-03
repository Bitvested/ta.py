"""Moving averages and smoothers."""


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
    """Smoothed Moving Average."""
    n = len(data)
    if n < l + 1:
        return []
    s = 0.0
    for i in range(l):
        s += float(data[i])
    first = s / l
    s += float(data[l]) - float(data[0])
    prev = (s - first) / l
    out = [prev]
    for i in range(l + 1, n):
        s += float(data[i]) - float(data[i - l])
        cur = (s - prev) / l
        out.append(cur)
        prev = cur
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
    pl = []; wm = [];
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            weight = 0; sum = 0;
            for a in range(len(pl)):
                sum += (pl[a][0] * pl[a][1]);
                weight += pl[a][1];
            wm.append(sum/weight);
            pl = pl[1:];
    return wm;

def vwwma(d: list[list[float]], length: int = 20) -> list[float]:
    """Volume Weighted Weighted Moving Average. Each row of data is [price, volume]."""
    data = []; weight = 0; vwm = [];
    for i in range(len(d)):
        data.append([d[i][0]*d[i][1],d[i][1]]);
    for i in range(1, length+1):
        weight += i;
    for i in range(length, len(data)+1):
        pl = data[i-length:i];
        totalv = 0; totalp = 0;
        for q in range(len(pl)):
            totalv += pl[q][1] * (q+1) / weight;
            totalp += pl[q][0] * (q+1) / weight;
        vwm.append(totalp / totalv);
    return vwm;

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
    ma = [];
    for i in range(len(weights),len(data)+1):
        pl = data[i-len(weights):i]; sum = 0; weight = 0;
        for q in range(len(weights)):
            sum += pl[q] * weights[q];
            weight += weights[q];
        ma.append(sum / weight);
    return ma;

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
