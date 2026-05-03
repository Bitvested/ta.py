"""Statistical and utility functions."""
from .moving_averages import ema, sma


def median(data: list[float], l: int = 0) -> list[float] | float:
    """Compute the rolling median of data with window l."""
    l = (l) if l > 0 else len(data); pl = []; med = [];
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            tmp = pl[:];
            tmp.sort(reverse=True);
            med.append(tmp[int(round((len(tmp)) / 2))]);
            pl = pl[1:];
    return med;

def kmeans(data: list[float], clusters: int) -> list[list[float]]:
    """Cluster data into k groups using k-means algorithm."""
    means = []; centers = []; old = []; changed = True; init = round(len(data)/(clusters+1));
    for i in range(clusters): centers.append([]);
    for i in range(clusters): centers[i] = data[init*(i+1)];
    while(changed):
        for i in range(clusters): means.append([]);
        changed = False;
        for x in range(len(data)):
            rang = -1; oldrange = -1;
            for y in range(clusters):
                r = abs(centers[y]-data[x]);
                if(oldrange == -1):
                    oldrange = r;
                    n = y;
                elif(r <= oldrange):
                    oldrange = r;
                    n = y;
            means[n].append(data[x]);
        old = centers;
        for x in range(clusters):
            sum = 0;
            for y in range(len(means[x])): sum += means[x][y];
            m = sum / len(means);
            centers[x] = m;
        for x in range(clusters):
            if centers[x] != old[x]: changed = True;
    return means;

def normalize(data: list[float], marg: float = 0) -> list[float]:
    """Normalize data to [0, 1] range with optional margin."""
    ma = max(data)*(1+marg); mi = min(data)*(1-marg); norm = [];
    for i in range(0, len(data)):
        norm.append(1-(ma-data[i])/(ma-mi));
    return norm;

def denormalize(data: list[float], norm: list[float], marg: float = 0) -> list[float]:
    """Denormalize previously normalized data back to original scale."""
    ma = max(data)*(1+marg); mi = min(data)*(1-marg); dnorm = [];
    for i in range(0, len(norm)):
        dnorm.append(mi+norm[i]*(ma-mi));
    return dnorm;

def standardize(data: list[float]) -> list[float]:
    """Standardize data to zero mean and unit standard deviation."""
    mean = sma(data, len(data));
    st = std(data); res = [];
    for i in range(len(data)):
        res.append((data[i]-mean[0])/st);
    return res;

def mad(data: list[float], l: int = 0) -> list[float] | float:
    """Compute the rolling median absolute deviation."""
    l = l if l > 0 else len(data); med = [];
    for i in range(l, len(data)+1):
        tmp = data[i-l:i];
        m = median(data[i-l:i]); adev = [];
        for q in range(len(tmp)): adev.append(abs(float(tmp[q]) - float(m[len(m)-1])));
        ad = median(adev);
        med.append(ad[len(ad)-1]);
    return med;

def aad(data: list[float], l: int = 0) -> list[float] | float:
    """Compute the rolling average absolute deviation."""
    l = l if l > 0 else len(data); med = [];
    for i in range(l, len(data)+1):
        tmp = data[i-l:i];
        m = sma(tmp, l); sum = 0;
        for q in range(len(tmp)): sum += abs(tmp[q] - m[len(m)-1]);
        med.append(sum/l);
    return med;

def ssd(data: list[float], l: int = 0) -> list[float] | float:
    """Compute the rolling sum of squared deviations (sample std dev variant)."""
    l = l if l > 0 else len(data); sd = [];
    for i in range(l, len(data)+1):
        tmp = data[i-l:i]; mean = sma(tmp, l); sum = 0;
        for q in range(len(tmp)): sum += (tmp[q] - mean[0])**2;
        sd.append(sum**(1/2));
    return sd;

def er(data: list[float]) -> float:
    """Compute the expected return of a series of returns."""
    wins = []; losses = []; wp = 1; lp = 1;
    for i in range(len(data)):
        if data[i] >= 0:
            wins.append(data[i]+1);
        else:
            losses.append(data[i]+1);
    win = (len(wins) / len(data)); loss = (len(losses) / len(data));
    for i in range(len(losses)):
        lp *= losses[i];
    for i in range(len(wins)):
        wp *= wins[i];
    return (((wp**(1/len(wins))-1)*100) * win + ((lp**(1/len(losses))-1)*100)*loss) / 100

def variance(data: list[float], l1: int = 0) -> list[float] | float:
    """Compute the rolling variance of data."""
    n = len(data)
    L = l1 if l1 > 0 else n
    if n < L:
        return []
    out: list[float] = []
    sum_x = 0.0
    sum_x2 = 0.0
    for i in range(L):
        v = float(data[i])
        sum_x += v
        sum_x2 += v * v
    mean = sum_x / L
    var = sum_x2 / L - mean * mean
    out.append(var if var > 0 else 0.0)
    for i in range(L, n):
        v_in = float(data[i])
        v_out = float(data[i - L])
        sum_x += v_in - v_out
        sum_x2 += v_in * v_in - v_out * v_out
        mean = sum_x / L
        var = sum_x2 / L - mean * mean
        out.append(var if var > 0 else 0.0)
    return out

def std(data: list[float], l1: int = 0) -> list[float] | float:
    """Compute the standard deviation of data."""
    l1 = (l1) if l1 > 0 else len(data); v = variance(data[:], l1);
    std = v[len(v)-1] ** (1.0/2.0)
    return std;

def normsinv(p: float) -> float:
    """Compute the inverse of the normal cumulative distribution function."""
    import math
    a1 = -39.6968302866538; a2 = 220.946098424521; a3 = -275.928510446969; a4 = 138.357751867269; a5 = -30.6647980661472; a6 = 2.50662827745924;
    b1 = -54.4760987982241; b2 = 161.585836858041; b3 = -155.698979859887; b4 = 66.8013118877197; b5 = -13.2806815528857; c1 = -7.78489400243029E-03;
    c2 = -0.322396458041136; c3 = -2.40075827716184; c4 = -2.54973253934373; c5 = 4.37466414146497; c6 = 2.93816398269878; d1 = 7.78469570904146E-03;
    d2 = 0.32246712907004; d3 = 2.445134137143; d4 = 3.75440866190742; p_low = 0.02425; p_high = 1 - p_low;
    if p < 0 or p > 1:
        return 0;
    if p < p_low:
        q = math.sqrt(-2*math.log(1-p));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
    if p <= p_high:
        q = p - 0.5;
        r = q * q;
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
    q = math.sqrt(-2*math.log(1-p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);

def sim(data: list[float], l: int = 50, sims: int = 1000, perc: float = -1) -> list[list[float]]:
    """Run Monte Carlo simulations of price paths using geometric Brownian motion."""
    import math, random
    sd = [];
    for i in range(sims):
        projected = data[:];
        for x in range(l):
            change = [];
            for y in range(1, len(projected)):
                change.append((projected[y]-projected[y-1])/projected[y-1]);
            mean = sma(change, len(change));
            st = std(change); rando = normsinv(random.random());
            projected.append(projected[len(projected)-1]*math.exp(mean[0]-(st*st)/2+st*rando));
        sd.append(projected);
    if perc <= -1: return sd;
    finalprojection = data[:];
    for i in range(len(sd[0])):
        so = [];
        for x in range(len(sd)):
            so.append(sd[x][i])
        so.sort();
        finalprojection.append(so[round((len(so)-1)*perc)]);
    return finalprojection;

def percentile(data: list[list[float]], perc: float = 0.5) -> list[float]:
    """Compute the percentile value across simulation paths at each time step."""
    final = [];
    for i in range(len(data[0])):
        tmp = [];
        for x in range(len(data)):
            tmp.append(data[x][i]);
        tmp.sort();
        final.append(tmp[round((len(tmp)-1)*perc)])
    return final;

def cor(data1: list[float], data2: list[float]) -> float:
    """Compute the Pearson correlation coefficient between two series."""
    d1avg = sma(data1[:], len(data1)); d1avg = d1avg[len(d1avg)-1];
    d2avg = sma(data2[:], len(data2)); d2avg = d2avg[len(d2avg)-1]
    sumavg = 0; sx = 0; sy = 0;
    for i in range(len(data1)):
        x = data1[i]-d1avg;
        y = data2[i]-d2avg;
        sumavg += (x * y); sx += x**2; sy += y**2;
    n = len(data1)-1;
    sx/=n; sy/=n; sx = sx ** (1/2); sy = sy ** (1/2);
    return (sumavg / (n*sx*sy));

def dif(n: float, o: float) -> float:
    """Compute the percentage difference between two values."""
    return (n-o)/o;

def drawdown(d: list[float]) -> float:
    """Compute the maximum drawdown of a price series."""
    max = d[0]; min = d[0]; big = 0;
    for y in range(1,len(d)):
        if(d[y] > max):
            if(min != 0):
                diff = dif(min, max);
                if(diff < big): big = diff;
                min = d[y];
            max = d[y];
        if(d[y] < min): min = d[y];
    diff = dif(min, max);
    if(diff < big): big = diff;
    return big;

def cum(data: list[float], length: int) -> list[float]:
    """Compute the rolling cumulative sum over a fixed window."""
    res = [];
    for i in range(length, len(data)+1):
        res.append(sum(data[i-length:i]));
    return res;

def mse(data1: list[float], data2: list[float]) -> float:
    """Compute the mean squared error between two series."""
    import math
    err = 0;
    for i in range(len(data1)):
        err += math.pow((data2[i]-data1[i]), 2);
    return err / len(data1);

def permutations(data: list[float]) -> float:
    """Compute the product of all elements in data."""
    total = 1;
    for i in range(len(data)):
        total *= data[i];
    return total;

def winratio(data: list[float]) -> float:
    """Compute the ratio of winning (non-negative) trades."""
    wins = 0; losses = 0;
    for i in range(len(data)):
        if data[i] >= 0:
            wins += 1;
        else:
            losses += 1;
    return wins / (losses + wins)

def avgwin(data: list[float]) -> float:
    """Compute the average of winning (non-negative) returns."""
    wins = [];
    for i in range(len(data)):
        if data[i] >= 0:
            wins.append(data[i]);
    avg = sma(wins, len(wins));
    return avg[0];

def avgloss(data: list[float]) -> float:
    """Compute the average of losing (negative) returns."""
    loss = [];
    for i in range(len(data)):
        if data[i] < 0: loss.append(data[i]);
    avg = sma(loss, len(loss));
    return avg[0];

def kelly(data: list[float]) -> float:
    """Compute the Kelly criterion fraction for a series of returns."""
    exp = er(data) + 1
    winr = winratio(data);
    return winr - (1-winr) / exp;

def se(data: list[float], size: int = 0) -> float:
    """Compute the standard error of data."""
    if size == 0:
        size = len(data);
    stdv = std(data);
    return stdv / (size ** 0.5)

def zscore(data: list[float], l: int) -> list[float]:
    """Compute the rolling z-score of data."""
    out = []; pl = data[0:l-1];
    for i in range(l-1, len(data)):
        pl.append(data[i]);
        mean = sma(pl, l);
        stdv = std(pl, l);
        out.append((data[i]-mean[0])/stdv);
        pl = pl[1:];
    return out;

def ar(data: list[float], l: int) -> list[float]:
    """Compute the autoregression residuals for each window."""
    out = [];
    for i in range(l, len(data)):
        exp = er(data[i-l:i]);
        out.append(data[i]-exp);
    return out;

def normalize_pair(data1: list[float], data2: list[float]) -> list[list[float]]:
    """Normalize two series to a common starting value."""
    f = (data1[0] + data2[0]) / 2; ret = [[f,f]];
    for i in range(1, len(data1)):
        ret.append([ret[len(ret)-1][0]*((data1[i]-data1[i-1])/data1[i-1]+1),ret[len(ret)-1][1]*((data2[i]-data2[i-1])/data2[i-1]+1)]);
    return ret;

def normalize_from(data: list[float], value: float) -> list[float]:
    """Normalize a series starting from a given base value."""
    ret = [value];
    for i in range(1, len(data)):
        ret.append(ret[len(ret)-1]*((data[i]-data[i-1])/data[i-1]+1));
    return ret;

def log(d: list[float]) -> list[float]:
    """Compute the natural logarithm of each element."""
    import math
    return list(map(lambda x: math.log(x),d));

def exp(d: list[float]) -> list[float]:
    """Compute the exponential of each element."""
    import math
    return list(map(lambda x: math.exp(x),d));

def covariance(data1: list[float], data2: list[float], length: int) -> list[float]:
    """Compute the rolling covariance between two series."""
    out = []; x_mean = sma(data1, len(data1));
    y_mean = sma(data2, len(data2)); res = 0;
    for z in range(length, len(data1)+1):
        x = data1[z-length:z]; y = data2[z-length:z];
        x_mean = sma(x, length); y_mean = sma(y, length);
        for i in range(length):
            res += (x[i]-x_mean[0])*(y[i]-y_mean[0]);
        out.append(res/length);
    return out;

def ncdf(x: float, mean: float = 0, std: float = 0) -> float:
    """Compute the normal cumulative distribution function."""
    import math
    if mean != 0 and std != 0:
        x = (x - mean) / std;
    t = 1 / ( 1+ .2315419 * abs(x));
    d = .3989423*math.exp(-x*x/2);
    p = d*t*(.3193815+t*(-.3565638+t*(1.781478+t*(-1.821256+t*1.330274))));
    if x > 0:
        return 1 - p;
    else:
        return p;

def pvalue(t: float, df: int) -> float:
    """Look up the p-value for a t-statistic given degrees of freedom."""
    t_table = [
        {'value':0.5,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0,22:0,23:0,24:0,25:0},
        {'value':0.25,1:1,2:0.816,3:0.765,4:0.741,5:0.727,6:0.718,7:0.711,8:0.706,9:0.703,10:0.7,11:0.697,12:0.695,13:0.694,14:0.692,15:0.691,16:0.69,17:0.689,18:0.688,19:0.688,20:0.687,21:0.686,22:0.686,23:0.685,24:0.685,25:0.684},
        {'value':0.2,1:1.376,2:1.061,3:0.978,4:0.941,5:0.92,6:0.906,7:0.896,8:0.889,9:0.883,10:0.879,11:0.876,12:0.873,13:0.87,14:0.868,15:0.866,16:0.865,17:0.863,18:0.862,19:0.861,20:0.86,21:0.859,22:0.858,23:0.858,24:0.857,25:0.856},
        {'value':0.15,1:1.963,2:1.386,3:1.25,4:1.19,5:1.156,6:1.134,7:1.119,8:1.108,9:1.1,10:1.093,11:1.088,12:1.088,13:1.079,14:1.076,15:1.074,16:1.071,17:1.069,18:1.067,19:1.066,20:1.064,21:1.063,22:1.061,23:1.06,24:1.059,25:1.058},
        {'value':0.1,1:3.078,2:1.886,3:1.638,4:1.533,5:1.476,6:1.44,7:1.415,8:1.397,9:1.383,10:1.372,11:1.363,12:1.356,13:1.35,14:1.345,15:1.341,16:1.337,17:1.333,18:1.33,19:1.328,20:1.325,21:1.323,22:1.321,23:1.319,24:1.318,25:1.316},
        {'value':0.05,1:6.314,2:2.92,3:2.353,4:2.132,5:2.015,6:1.943,7:1.895,8:1.86,9:1.833,10:1.812,11:1.796,12:1.782,13:1.771,14:1.761,15:1.753,16:1.746,17:1.74,18:1.734,19:1.729,20:1.725,21:1.721,22:1.717,23:1.714,24:1.711,25:1.708},
        {'value':0.025,1:12.71,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,13:2.16,14:2.145,15:2.131,16:2.12,17:2.11,18:2.101,19:2.093,20:2.086,21:2.08,22:2.074,23:2.069,24:2.064,25:2.06},
        {'value':0.01,1:31.82,2:6.965,3:4.541,4:3.747,5:3.365,6:3.143,7:2.998,8:2.896,9:2.821,10:2.764,11:2.718,12:2.681,13:2.65,14:2.624,15:2.602,16:2.583,17:2.567,18:2.552,19:2.539,20:2.528,21:2.518,22:2.508,23:2.5,24:2.492,25:2.485},
        {'value':0.005,1:63.66,2:9.925,3:5.841,4:4.604,5:4.032,6:3.707,7:3.499,8:3.355,9:3.25,10:3.169,11:3.106,12:3.055,13:3.012,14:2.977,15:2.947,16:2.921,17:2.898,18:2.878,19:2.861,20:2.845,21:2.831,22:2.819,23:2.807,24:2.797,25:2.787},
        {'value':0.001,1:318.13,2:22.327,3:10.215,4:7.173,5:5.893,6:5.208,7:4.785,8:4.501,9:4.297,10:4.144,11:4.025,12:3.93,13:3.852,14:3.787,15:3.733,16:3.686,17:3.646,18:3.61,19:3.579,20:3.552,21:3.527,22:3.505,23:3.485,24:3.467,25:3.45},
        {'value':0.0005,1:636.62,2:31.599,3:12.924,4:8.61,5:6.869,6:5.959,7:5.408,8:5.041,9:4.781,10:4.587,11:4.437,12:4.318,13:4.221,14:4.14,15:4.073,16:4.015,17:3.965,18:3.922,19:3.883,20:3.85,21:3.819,22:3.792,23:3.768,24:3.745,25:3.725}
    ];
    if df > 25: raise Exception('ta.py | pvalue | df too high!');
    if df < 1: raise Exception('ta.py | pvalue | df too low!');
    for x in range(len(t_table) - 1):
        if t >= t_table[x][df] and t <= t_table[x+1][df]:
            return t_table[x+1]['value'] + (t_table[x+1]['value'] - t_table[x]['value']) * (t_table[x+1][df] - t) / (t_table[x][df] - t_table[x+1][df]);
    return 0.0001;

def expected_trails(n: int) -> int:
    """Compute the expected number of trials before seeing all n outcomes."""
    import math
    s = 0;
    for i in range(1, n+1): s += 1.0 / i;
    return math.ceil(n * s);

def martingale(data: list[int], bet: float, max: float | None = None, multiplier: float = 2) -> float:
    """Compute the next bet size using a martingale strategy."""
    current = bet;
    for i in range(len(data)):
        if data[i] < 0:
            current *= multiplier;
        elif data[i] > 0:
            current = bet;
    return max if max is not None and current > max else current;

def antimartingale(data: list[int], bet: float, max: float | None = None, multiplier: float = 2) -> float:
    """Compute the next bet size using an anti-martingale strategy."""
    current = bet;
    for i in range(len(data)):
        if data[i] > 0:
            current *= multiplier;
        elif data[i] < 0:
            current = bet;
    return max if max is not None and current > max else current;

def return_positive(data: list[float]) -> list[float]:
    """Filter data to return only positive values."""
    out = [];
    for i in range(len(data)):
        if data[i] > 0: out.append(data[i]);
    return out;

def return_negative(data: list[float]) -> list[float]:
    """Filter data to return only negative values."""
    out = [];
    for i in range(len(data)):
        if data[i] < 0: out.append(data[i]);
    return out;

def recent_high(data: list[float], lb: int = 25) -> dict:
    """Find the most recent significant high within the lookback window."""
    xback = lb; hindex = 0; highest = data[len(data)-1];
    for i in range(len(data)-2, 0, -1):
        if data[i] >= highest and xback > 0:
            highest = data[i];
            hindex = i;
            xback = lb;
        else:
            xback -= 1;
        if xback <= 0: break;
    return {"index": hindex, "value": highest};

def recent_low(data: list[float], lb: int = 25) -> dict:
    """Find the most recent significant low within the lookback window."""
    xback = lb; lindex = 0; lowest = data[len(data)-1];
    for i in range(len(data)-2, 0, -1):
        if data[i] <= lowest and xback > 0:
            lowest = data[i];
            lindex = i;
            xback = lb;
        else:
            xback -= 1;
        if xback <= 0: break;
    return {"index": lindex, "value": lowest};

def _lr_windows(data: list[float], length: int) -> list[list[float]]:
    """Internal helper: rolling least-squares ``[[slope, intercept], ...]`` over
    windows of size ``length``. ``x`` is ``0..length-1`` per window (TA-Lib /
    tulind convention). Output length is ``n - length + 1`` (matches sma).
    """
    n = len(data)
    if n < length:
        return []
    sum_x = (length - 1) * length / 2.0
    sum_x2 = (length - 1) * length * (2 * length - 1) / 6.0
    denom = length * sum_x2 - sum_x * sum_x
    sum_y = 0.0
    sum_xy = 0.0
    for q in range(length):
        y = float(data[q])
        sum_y += y
        sum_xy += y * q
    out: list[list[float]] = [[0.0, 0.0]] * (n - length + 1)
    def _pair(sy: float, sxy: float) -> list[float]:
        m = float("nan") if denom == 0 else (length * sxy - sum_x * sy) / denom
        b = (sy - m * sum_x) / length
        return [m, b]
    out[0] = _pair(sum_y, sum_xy)
    for i in range(length, n):
        incoming = float(data[i])
        outgoing = float(data[i - length])
        sum_y += incoming - outgoing
        sum_xy += length * incoming - sum_y
        out[i - length + 1] = _pair(sum_y, sum_xy)
    return out


def lr_slope(data: list[float], length: int = 14) -> list[float]:
    """Linear regression slope over rolling windows of ``length`` bars."""
    w = _lr_windows(data, length)
    return [pair[0] for pair in w]


# --- sum MUST be last: it shadows the builtin sum() for all code defined above ---
def sum(data: list[float]) -> float:
    """Compute the running total of all elements in data."""
    s = 0;
    for i in range(len(data)): s += data[i];
    return s;
