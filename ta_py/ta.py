import math;
import random;
def median(data, l=0):
    l = (l) if l > 0 else len(data); pl = []; med = [];
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            tmp = pl[:];
            tmp.sort(reverse=True);
            med.append(tmp[int(round((len(tmp)) / 2))]);
            pl = pl[1:];
    return med;
def kmeans(data, clusters):
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
def normalize(data, marg=0):
    ma = max(data)*(1+marg); mi = min(data)*(1-marg); norm = [];
    for i in range(0, len(data)):
        norm.append(1-(ma-data[i])/(ma-mi));
    return norm;
def denormalize(data, norm, marg=0):
    ma = max(data)*(1+marg); mi = min(data)*(1-marg); dnorm = [];
    for i in range(0, len(norm)):
        dnorm.append(mi+norm[i]*(ma-mi));
    return dnorm;
def standardize(data):
    mean = sma(data, len(data));
    st = std(data); res = [];
    for i in range(len(data)):
        res.append((data[i]-mean[0])/st);
    return res;
def mad(data, l=0):
    l = l if l > 0 else len(data); med = [];
    for i in range(l, len(data)+1):
        tmp = data[i-l:i];
        m = median(data[i-l:i]); adev = [];
        for q in range(len(tmp)): adev.append(abs(float(tmp[q]) - float(m[len(m)-1])));
        ad = median(adev);
        med.append(ad[len(ad)-1]);
    return med;
def aad(data, l=0):
    l = l if l > 0 else len(data); med = [];
    for i in range(l, len(data)+1):
        tmp = data[i-l:i];
        m = sma(tmp, l); sum = 0;
        for q in range(len(tmp)): sum += abs(tmp[q] - m[len(m)-1]);
        med.append(sum/l);
    return med;
def ssd(data, l=0):
    l = l if l > 0 else len(data); sd = [];
    for i in range(l, len(data)+1):
        tmp = data[i-l:i]; mean = sma(tmp, l); sum = 0;
        for q in range(len(tmp)): sum += (tmp[q] - mean[0])**2;
        sd.append(sum**(1/2));
    return sd;
def er(data):
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
def rsi(data, l=14):
    pl = []; rs = [];
    for i in range(1, len(data)):
        pl.append(data[i] - data[i - 1]);
        if (len(pl)) >= l:
            gain = 0.0; loss = 0.0;
            for a in range(len(pl)):
                if pl[a] > 0: gain += float(pl[a]);
                if pl[a] < 0: loss += abs(float(pl[a]));
            gain /= l; loss /= l;
            try:
                f = 100.0 - 100.0 / (1.0 + (gain/loss));
            except:
                f = 100.0
            rs.append(f);
            pl = pl[1:];
    return rs;
def wrsi(data, l=14):
    arrsi = []; u = []; d = [];
    for i in range(1,len(data)):
        if(data[i]-data[i-1]<0):
            d.append(abs(data[i]-data[i-1]));
            u.append(0.0);
        else:
            d.append(0.0);
            u.append(data[i]-data[i-1]);
    d = wsma(d, l); u = wsma(u, l);
    for i in range(0, len(d)):
        try:
            f = 100.0-100.0/(1.0+(u[i]/d[i]));
        except:
            f = 100.0;
        arrsi.append(f);
    return arrsi;
def sma(data, l=14):
    pl = []; sm = [];
    for i in range(len(data)):
        pl.append(float(data[i]));
        if (len(pl)) >= l:
            sm.append(sum(pl) / l);
            pl = pl[1:];
    return sm;
def smma(data, l=14):
    pl = []; sm = [];
    for i in range(len(data)):
        pl.append(float(data[i]));
        if (len(pl)) >= l:
            average = 0;
            for a in range(len(pl)): average += pl[a];
            f = average / l if len(sm) <= 0 else (average - sm[len(sm) - 1]) / l;
            sm.append(f);
            pl = pl[1:];
    sm = sm[1:];
    return sm;
def wma(data, l=14):
    pl = []; wm = []; weight = [];
    for i in range(1, l+1):
        weight.append(i);
    for i in range(len(data)):
        pl.append(float(data[i]));
        if (len(pl)) >= l:
            average = 0;
            for q in range(1, len(pl)+1):
                average += pl[q-1] * q / sum(weight);
            wm.append(average);
            pl = pl[1:];
    return wm;
def wsma(data, l=14):
    em = []; weight = 1/l;
    for i in range(l, len(data)+1):
        if len(em) > 0:
            em.append((data[i-1]-em[len(em)-1]) * weight + em[len(em)-1]);
            continue;
        pl = data[i-l:i]; average = 0;
        for q in range(0, len(pl)):
            average += pl[q];
        em.append(average/len(pl));
    return em;
def pwma(data, l=14):
    weight = 0; wmaa = []; weights = []; b = l;
    for i in range(-round(l/2), 0):
        if(abs(i) % 1 != 0):
            i = round(abs(i));
            weight += (i*b);
        else:
            weights.append(abs(i)*b);
            weight += (abs(i)*b*2);
        weights.insert(0,abs(i)*b);
        b -= 1;
    for i in range(l, len(data)+1):
        average = 0; pl = data[i-l:i];
        for x in range(0, len(weights)):
            average += pl[x] * weights[x] / weight;
        wmaa.append(average);
    return wmaa;
def hwma(data, l=14):
    weight = 0; wmaa = []; weights = []; b = l;
    for i in range(1, round(l/2+1)):
        if(i%1 != 0):
            i = round(i);
            weight += (i*b);
        else:
            weights.append(i*b);
            weight += (i*b*2)
        weights.insert(0,i*b);
        b -= 1;
    for i in range(l, len(data)+1):
        average = 0; pl = data[i-l:i];
        for x in range(len(weights)):
            average += pl[x] * weights[x] / weight;
        wmaa.append(average);
    return wmaa;
def ema(data, l=12):
    pl = []; em = []; weight = 2 / (float(l) + 1);
    for i in range(len(data)):
        pl.append(float(data[i]));
        if (len(pl)) >= l:
            f = sum(pl) / l if len(em) <= 0 else (data[i] - em[len(em) - 1]) * weight + em[len(em) - 1]
            em.append(f);
            pl = pl[1:];
    return em;
def tsi(data, long=25, short=13, sig=13):
    mo = []; ab = []; ts = []; tsi = [];
    for i in range(1, len(data)):
        mo.append(data[i] - data[i - 1]);
        ab.append(abs(data[i] - data[i - 1]));
    sma1 = ema(mo, long); sma2 = ema(ab, long);
    ema1 = ema(sma1, short); ema2 = ema(sma2, short);
    for i in range(len(ema1)):
        ts.append(ema1[i] / ema2[i]);
    tma = ema(ts, sig);
    ts = ts[(len(ts)-len(tma)):];
    for i in range(len(tma)):
        tsi.append([tma[i], ts[i]]);
    return tsi;
def vwma(data, l=20):
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
def hull(data, l=14):
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
def kama(data, l1=10, l2=2, l3=30):
    ka = sma(data[:], l1); ka = [ka[len(ka)-1]];
    for i in range(l1 + 1, len(data)):
        vola = 0; change = abs(data[i] - data[i - l1]);
        for a in range(1, l1):
            vola += abs(float(data[i-a]) - float(data[(i-a)-1]));
        sc = (change/vola * (2.0/(l2+1.0) - 2.0/(l3+1.0) + 2.0/(l3+1.0))) ** 2.0;
        ka.append(ka[len(ka)-1] + sc * (float(data[i]) - ka[len(ka)-1]));
    return ka;
def macd(data, l1=12, l2=26):
    if l1 > l2: [l1, l2] = [l2, l1];
    em1 = ema(data[:], l1); em2 = ema(data[:], l2);
    em1 = em1[-(len(em2) - len(em1)):]; emf = [];
    for i in range(len(em1)):
        emf.append(em1[i] - em2[i]);
    return emf;
def variance(data, l1=0):
    l1 = l1 if l1 > 0 else len(data); va = [];
    for i in range(l1, len(data)+1):
        tmp = data[i-l1:i]; mean = sma(tmp, len(tmp)); sum = 0;
        for x in range(len(tmp)): sum += ((tmp[x] - mean[len(mean)-1]) ** 2);
        va.append(sum/l1);
    return va;
def std(data, l1=0):
    l1 = (l1) if l1 > 0 else len(data); v = variance(data[:], l1);
    std = v[len(v)-1] ** (1.0/2.0)
    return std;
def normsinv(p):
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
def sim(data, l=50, sims=1000, perc=-1):
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
def percentile(data, perc=0.5):
    final = [];
    for i in range(len(data[0])):
        tmp = [];
        for x in range(len(data)):
            tmp.append(data[x][i]);
        tmp.sort();
        final.append(tmp[round((len(tmp)-1)*perc)])
    return final;
def bands(data, l1=14, l2=1):
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
def bandwidth(data, l1=14, l2=1):
    band = bands(data[:], l1, l2); boll = [];
    for i in range(len(band)): boll.append((band[i][0] - band[i][2]) / band[i][1]);
    return boll;
def atr(data, l1=14):
    atr = [float(data[0][0]) - float(data[0][2])];
    for i in range(1, len(data)):
        t0 = max((float(data[i][0]) - float(data[i - 1][1])), (float(data[i][2]) - float(data[i - 1][1])), (float(data[i][0]) - float(data[i][2])));
        atr.append((atr[len(atr)-1] * (l1 - 1) + t0) / l1)
    return atr;
def keltner(data, l1=14, l2=1):
    closing = []; tr = atr(data[:], l1); kelt = [];
    for i in range(len(data)): closing.append(sum(data[i]) / 3.0);
    kma = sma(closing, l1);
    tr = tr[(len(tr) - len(kma)):];
    for i in range(len(kma)): kelt.append([kma[i] + tr[i] * l2, kma[i], kma[i] - tr[i] * l2]);
    return kelt
def cor(data1, data2):
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
def dif(n, o): return (n-o)/o;
def drawdown(d):
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
def aroon_up(data, l1=10):
    pl = []; aroon = [];
    for i in range(len(data)):
        pl.append(float(data[i]));
        if(len(pl) >= l1):
            hl = pl[:];
            aroon.append((100.0 * (l1-1-pl.index(max(hl))) / (l1-1)) if l1-1 != 0 else 0);
            pl = pl[1:];
    return aroon;
def aroon_down(data, l1=10):
    pl = []; aroon = [];
    for i in range(len(data)):
        pl.append(float(data[i]));
        if(len(pl) >= l1):
            hl = pl[:];
            hl.reverse();
            aroon.append(100.0 * (l1-1-hl.index(min(hl))) / (l1-1) if l1-1 != 0 else 0);
            pl = pl[1:];
    return aroon;
def aroon_osc(data, l1=10):
    pl = []; aroon = [];
    u = aroon_up(data[:], l1); d = aroon_down(data[:], l1);
    for i in range(len(u)): aroon.append(u[i] - d[i]);
    return aroon;
def mfi(data, l1=14):
    mfi = []; n = []; p = [];
    for i in range(len(data)):
        p.append(float(data[i][0])); n.append(float(data[i][1]));
        if(len(p) >= l1):
            mfi.append((100.0 - 100.0 / (1 + sum(p) / sum(n))));
            p = p[1:]; n = n[1:];
    return mfi;
def roc(data, l1=14):
    pl = []; roc = [];
    for i in range(len(data)):
        pl.append(float(data[i]));
        if(len(pl) >= l1):
            roc.append((pl[len(pl)-1] - pl[0]) / pl[0]);
            pl = pl[1:];
    return roc;
def cop(data, l1=11, l2=14, l3=10):
    m = max(l1, l2); co = [];
    if l1 > l2: [l1, l2] = [l2, l1];
    for i in range(m + l3, len(data)):
        r1 = roc(data[i-(m+l3):i], l1); r2 = roc(data[i-(m+l3):i], l2); tmp = [];
        r1 = r1[(len(r1) - len(r2)):];
        for a in range(len(r1)):
            tmp.append(r1[a] + r2[a]);
        tmp = wma(tmp[:], l3);
        co.append(tmp[len(tmp)-1]);
    return co;
def kst(data, r1=10, s1=10, r2=15, s2=10, r3=20, s3=10, r4=30, s4=15, sig=9):
    ks = []; fs = []; ms = (max(r1, r2, r3, r4) + max(s1, s2, s3, s4));
    for i in range(ms, len(data)+1):
        rcma1 = roc(data[i-ms:i], r1); rcma2 = roc(data[i-ms:i], r2); rcma3 = roc(data[i-ms:i], r3); rcma4 = roc(data[i-ms:i], r4);
        rcma1 = sma(rcma1, s1); rcma2 = sma(rcma2, s2); rcma3 = sma(rcma3, s3); rcma4 = sma(rcma4, s4);
        ks.append(rcma1[len(rcma1)-1] + rcma2[len(rcma2)-1] + rcma3[len(rcma3)-1] + rcma4[len(rcma4)-1]);
    sl = sma(ks[:], sig);
    ks = ks[(len(ks) - len(sl)):];
    for i in range(len(ks)): fs.append([ks[i], sl[i]]);
    return fs;
def obv(data):
    obv = [0];
    for i in range(1, len(data)):
        if(data[i][1] > data[i-1][1]): obv.append(obv[len(obv)-1] + data[i][0]);
        if(data[i][1] < data[i-1][1]): obv.append(obv[len(obv)-1] - data[i][0]);
        if(data[i][1] == data[i-1][1]): obv.append(obv[len(obv)-1]);
    return obv;
def vwap(data, l1=0):
    l1 = l1 if l1 > 0 else len(data); pl = []; vwap = [];
    for i in range(len(data)):
        pl.append([(data[i][0] * data[i][1]), data[i][1]]);
        if(len(pl) >= l1):
            totalv = 0; totalp = 0;
            for a in range(len(pl)):
                totalv += pl[a][1];
                totalp += pl[a][0];
            vwap.append(totalp/totalv);
            pl = pl[1:];
    return vwap;
def mom(data, l1=10, p=False):
    mom = [];
    for i in range(l1-1, len(data)):
        mom.append(data[i] / data[i - (l1-1)] * 100) if p == True else mom.append(data[i] - data[i - (l1 - 1)]);
    return mom
def mom_osc(data, l1=9):
    pl = []; osc = []; l1+=1;
    for i in range(len(data)):
        pl.append(data[i]);
        if(len(pl) >= l1):
            sumh = 0; suml = 0;
            for a in range(1, l1):
                if(pl[a-1] < pl[a]): sumh = sumh+pl[a];
                else: suml = suml+pl[a];
            osc.append((sumh - suml) / (sumh + suml) * 100);
            pl = pl[1:];
    return osc;
def bop(data, l1=14):
    bo = [];
    for i in range(len(data)):
        bo.append((data[i][3] - data[i][0]) / (data[i][1] - data[i][2]) if (data[i][1] - data[i][2]) != 0 else 0);
    bo = sma(bo, l1);
    return bo;
def fi(data, l1=13):
    pl = []; ff = [];
    for i in range(1, len(data)):
        pl.append((data[i][0] - data[i-1][0]) * data[i][1]);
        if(len(pl) >= l1):
            vfi = ema(pl[:], l1);
            ff.append(vfi[len(vfi)-1]);
            pl = pl[1:];
    return ff;
def asi(data):
    pl = []; a = [];
    for i in range(1, len(data)):
        c = float(data[i][1]); y = float(data[i-1][1]); h = float(data[i][0]); hy = float(data[i-1][0]); cy = float(data[i-1][1]);
        l = float(data[i][2]); ly = float(data[i-1][2]); o = float(data[i][0]); oy = float(data[i-1][0]); t = max(float(data[i][0]), float(data[i-1][0])) - min(float(data[i][2]), float(data[i-1][2]));
        if(hy-c > ly-c): k = hy-c;
        else: k = ly-c;
        if((h - cy > l - cy) & (h - cy > h - l)): r = h - cy - (l - cy) / 2.0 + (cy - oy) / 4.0;
        if((l - cy > h - cy) & (l - cy > h - l)): r = l - cy - (h - cy) / 2.0 + (cy - oy) / 4.0;
        if((h - l > h - cy) & (h - l > l - cy)): r = h - l + (cy - oy) / 4.0;
        a.append(50.0 * ((cy - c + (cy - oy) / 2.0 + (c - o) / 2.0) / r) * k / t if r and t != 0 else 0);
    return a;
def ao(data, l1=5, l2=35):
    pl = []; a = [];
    for i in range(len(data)):
        pl.append((float(data[i][0]) + float(data[i][1])) / 2.0);
        if(len(pl) >= l2):
            f = sma(pl[:], l1);
            s = sma(pl[:], l2);
            a.append(f[len(f)-1] - s[len(s)-1]);
            pl = pl[1:];
    return a;
def pr(data, l1=14):
    n = []; pl = [];
    for i in range(len(data)):
        pl.append(float(data[i]));
        if(len(pl) >= l1):
            highd = max(pl[:]); lowd = min(pl[:]);
            n.append((highd - data[i]) / (highd - lowd) * -100.0 if highd-lowd != 0 else 0);
            pl = pl[1:];
    return n;
def lsma(data, l1=25):
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
def don(data, l1=20):
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
def ichimoku(data, l1=9, l2=26, l3=52, l4=26):
    cloud = []; place = []; pl = [];
    for i in range(len(data)):
        pl.append(data[i]);
        if(len(pl) >= l3):
            highs = []; lows = [];
            for a in range(len(pl)):
                highs.append(float(pl[a][0]));
                lows.append(float(pl[a][2]));
            tsen = (max(highs[len(highs)-l1:len(highs)]) + (min(lows[len(lows)-l1:len(lows)]))) / 2.0;
            ksen = (max(highs[len(highs)-l2:len(highs)]) + (min(lows[len(lows)-l2:len(lows)]))) / 2.0;
            senka = float(data[i][1]) + ksen;
            senkb = (max(highs[len(highs)-l3:len(highs)]) + (min(lows[len(lows)-l2:len(lows)]))) / 2.0;
            chik = float(data[i][1]);
            place.append([tsen, ksen, senka, senkb, chik]);
            pl = pl[1:];
    for i in range(l4, len(place)-l4):
        cloud.append([place[i][0], place[i][1], place[i+l4][2], place[i+l4][3], place[i-l4][4]]);
    return cloud;
def stoch(data, l1=14, sd=3, sk=3):
    stoch = []; high = []; low = []; ka = [];
    for i in range(len(data)):
        high.append(float(data[i][0]));
        low.append(float(data[i][2]));
        if(len(high) >= l1):
            highd = max(high); lowd = min(low);
            k = 100.0 * (data[i][1] - lowd) / (highd - lowd) if highd - lowd != 0 else 0;
            ka.append(k);
        if(sk > 0 & len(ka) > sk):
            smoothedk = sma(ka[:], sk);
            ka.push(smoothedk[len(smoothedk)-1]);
        if(len(ka) - sk >= sd):
            d = sma(ka[:], sd);
            stoch.append([k, d[len(d)-1]]);
            high = high[1:];
            low = low[1:];
            ka = ka[1:];
    return stoch;
def ha(data):
    h = [[(float(data[0][0]) + float(data[0][3])) / 2, float(data[0][1]), float(data[0][2]), (float(data[0][0]) + float(data[0][1]) + float(data[0][2]) + float(data[0][3])) / 4.0]];
    for i in range(len(data)):
        h.append([(h[len(h)-1][0] + h[len(h)-1][3]) / 2, max(h[len(h)-1][0], h[len(h)-1][3], float(data[i][1])), min(h[len(h)-1][0], h[len(h)-1][3], float(data[i][2])), (float(data[i][0]) + float(data[i][1]) + float(data[i][2]) + float(data[i][3])) / 4.0]);
    return h;
def ren(data, bs=1):
    re = []; decimals = len(str(bs-int(bs))[1:]);
    bh = round(round(float(data[0][0]) / float(bs) * (10.0 ** decimals)) / (10.0 ** decimals) * float(bs)); bl = bh - bs;
    for i in range(1, len(data)):
        if(data[i][0] > bh + bs):
            while (data[i][0] > bh + bs):
                re.append([bh,bh+bs,bh,bh+bs]);
                bh+=bs;
                bl+=bs;
        elif (data[i][1] < bl - bs):
            while (data[i][1] < bl - bs):
                re.append([bl,bl,bl-bs,bl-bs]);
                bh-=bs;
                bl-=bs;
    return re;
def envelope(data, l1=10, p=0.005):
    enve = [];
    for i in range(l1, len(data)):
        sm = sma(data[i-l1:i], l1);
        enve.append([sm[0]*(1+p),sm[0],sm[0]*(1-p)]);
    return enve;
def chaikin_osc(data, ema1=3, ema2=10):
    cha = []; adl = [];
    for i in range(len(data)):
        mfm = ((data[i][1]-data[i][2])-(data[i][0]-data[i][1]))/(data[i][0]-data[i][2]) if data[i][0]-data[i][2] != 0 else 0;
        adl.append(mfm*data[i][3]);
    ef = ema(adl, ema1); es = ema(adl, ema2);
    if len(ef) > len(es):
        ef = ef[len(ef)-len(es):];
    else:
        es = es[len(es)-len(ef):];
    for i in range(len(ef)):
        cha.append(ef[i]-es[i]);
    return cha;
def fractals(data):
    fractals = [[False, False], [False, False]];
    for i in range(2, len(data)-2):
        up = True if (data[i-2][0] < data[i][0] and data[i-1][0] < data[i][0] and data[i][0] > data[i+1][0] and data[i][0] > data[i+2][0]) else False
        down = True if (data[i-2][1] > data[i][1] and data[i-1][1] > data[i][1] and data[i][1] < data[i+1][1] and data[i][1] < data[i+2][1]) else False
        fractals.append([up, down]);
    fractals.append([False, False]);
    fractals.append([False, False]);
    return fractals;
def alligator(data, jl=13, tl=8, ll=5, js=8, ts=5, ls=3):
    ret = []; jaw = smma(data, jl); teeth = smma(data, tl); lips = smma(data, ll);
    teeth = teeth[len(teeth)-len(jaw):];
    lips = lips[len(lips)-len(jaw):];
    for i in range(len(jaw)-1, (js-2), -1):
        ret.append([jaw[i-(js-1)], teeth[i-(ts-1)], lips[i-(ls-1)]]);
    return ret;
def gator(data, jl=13, tl=8, ll=5, js=8, ts=5, ls=3):
    ret = []; jaw = smma(data, jl); teeth = smma(data, tl); lips = smma(data, ll);
    teeth = teeth[len(teeth)-len(jaw):];
    lips = lips[len(lips)-len(jaw):];
    for i in range(len(jaw)-1, (js-2), -1):
        ret.append([jaw[i-(js-1)]-teeth[i-(ts-1)], -(abs(teeth[i-(ts-1)]-lips[i-(ls-1)]))]);
    return ret;
def recent_high(data, lb=25):
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
def recent_low(data, lb=25):
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
def support(d, hl=0):
    hl = hl if hl else recent_low(d);
    findex = False; lowform = hl['value'];
    while findex == False:
        for i in range(hl['index'], len(d)):
            if hl['index']-i == 0: continue;
            newlow = (hl['value']-d[i])/(hl['index']-i);
            if newlow < lowform:
                lowform = newlow;
                index2 = i;
        if hl['index'] + 1 == index2 and index2 != len(d):
            hl['index'] = index2;
            lowform = min(d[:]);
            hl['value'] = d[hl['index']];
            findex = False;
        else:
            findex = True;
        if hl['index'] == len(d)-1: findex = True;
    if index2 == len(d)-1 or hl['index'] == len(d)-1:
        def calculate(pos):
            return pos*0+hl['value'];
        return {"calculate": calculate, "slope": 0, "lowest": hl['value'], "index": hl['index']};
    else:
        def calculate(pos):
            return pos*lowform+hl['value']
        return {"calculate": calculate, "slope": lowform, "lowest": hl['value'], "index": hl['index']};
def resistance(d, hl=0):
    hl = hl if hl else recent_high(d);
    findex = False; highform = hl['value'];
    while findex == False:
        for i in range(hl['index'], len(d)):
            if hl['index']-i == 0: continue;
            newhigh = (d[i]-hl['value'])/(hl['index']-i);
            if newhigh < highform:
                highform = newhigh
                index2 = i;
        if hl['index']+1 == index2 and index2 != len(d)-1:
            hl['index'] = index2;
            highform = max(d[:]);
            hl['value'] = d[hl['index']];
            findex = False;
        else:
            findex = True;
        if hl['index'] == len(d)-1: findex = True;
    if index2 == len(d)-1 or hl['index'] == len(d)-1:
        highform = 0;
    if index2 == len(d)-1 or hl['index'] == len(d)-1:
        def calculate(pos):
            return pos*0+hl['value'];
        return {"calculate": calculate, "slope": 0, "highest": hl['value'], "index": hl['index']}
    else:
        def calculate(pos):
            return pos*-highform+hl['value'];
        return {"calculate": calculate, "slope": highform, "highest": hl['value'], "index": hl['index']};
def ac(data, l1=5, l2=35):
    pl = [];
    for i in range(len(data)):
        pl.append((data[i][0]+data[i][1])/2);
    a = ao(data, l1, l2); sm =sma(a, l1); acr = [];
    if len(a) > len(sm):
        a = a[len(a)-len(sm):]
    else:
        sm = sm[len(sm)-len(a):];
    for i in range(len(a)): acr.append(a[i]-sm[i]);
    return acr;
def fib(start, end):
    return [start, (end-start)*.236+start, (end-start)*.382+start, (end-start)*.5+start, (end-start)*.618+start, (end-start)*.786+start, end, (end-start)*1.618+start, (end-start)*2.618+start, (end-start)*3.618+start, (end-start)*4.236+start]
def ar(data, l):
    out = [];
    for i in range(l, len(data)):
        exp = er(data[i-l:i]);
        out.append(data[i]-exp);
    return out;
def winratio(data):
    wins = 0; losses = 0;
    for i in range(len(data)):
        if data[i] >= 0:
            wins += 1;
        else:
            losses += 1;
    return wins / (losses + wins);
def avgwin(data):
    wins = [];
    for i in range(len(data)):
        if data[i] >= 0:
            wins.append(data[i]);
    avg = sma(wins, len(wins));
    return avg[0];
def avgloss(data):
    loss = [];
    for i in range(len(data)):
        if data[i] < 0:
            loss.append(data[i]);
    avg = sma(loss, len(loss));
    return avg[0];
def kelly(data):
    exp = er(data) + 1
    winr = winratio(data);
    return winr - (1-winr) / exp;
def se(data, size=0):
    if size == 0:
        size = len(data);
    stdv = std(data);
    return stdv / (size ** 0.5)
def zscore(data, l):
    out = []; pl = data[0:l-1];
    for i in range(l-1, len(data)):
        pl.append(data[i]);
        mean = sma(pl, l);
        stdv = std(pl, l);
        out.append((data[i]-mean[0])/stdv);
        pl = pl[1:];
    return out;
def normalize_pair(data1, data2):
    f = (data1[0] + data2[0]) / 2; ret = [[f,f]];
    for i in range(1, len(data1)):
        ret.append([ret[len(ret)-1][0]*((data1[i]-data1[i-1])/data1[i-1]+1),ret[len(ret)-1][1]*((data2[i]-data2[i-1])/data2[i-1]+1)]);
    return ret;
def normalize_from(data, value):
    ret = [value];
    for i in range(1, len(data)):
        ret.append(ret[len(ret)-1]*((data[i]-data[i-1])/data[i-1]+1));
    return ret;
def fisher(data, l):
    out = []; fish = 0; v1 = 0;
    for i in range(l, len(data)):
        pl = data[i-l:i]; pf = fish;
        mn = min(pl);
        v1 = .33*2*((data[i]-mn)/(max(pl)-mn)-.5)+.67*v1;
        if v1 > .99:
            v1 = .999;
        if v1 < -.99:
            v1 = -.999;
        fish = 0.5 * math.log((1+v1)/(1-v1)) + 0.5 * pf;
        out.append([fish, pf]);
    return out[1:];
def cross(d1, d2):
    d1 = d1[len(d1)-len(d2):];
    cross = (d1[0] > d2[0]);
    indexes = [];
    for i in range(len(d1)):
        if d1[i] < d2[i] and cross:
            indexes.append({"index": i, "cross": False});
            cross = False;
        if d1[i] > d2[i] and cross == False:
            indexes.append({"index": i, "cross": True});
            cross = True;
    return indexes;
def halftrend(data, atrlen, amplitude, deviation):
    out = []; nexttrend = [0]; trend = [0]; up = [0]; down = [0]; direction = None;
    for i in range(atrlen, len(data)):
        maxlow = data[i-1][2];
        minhigh = data[i-1][0];
        atr2 = atr(data[i-atrlen:i], atrlen);
        atr2 = atr2[len(atr2)-1] / 2;
        dev = deviation * atr2;
        highprice = max(data[i-1][0], data[i][0]);
        lowprice = min(data[i-1][2], data[i][2]);
        highs = list(map(lambda x: x[0], data[i-amplitude:i]));
        lows = list(map(lambda x: x[2], data[i-amplitude:i]));
        highma = sma(highs, len(highs));
        lowma = sma(lows, len(lows));
        if nexttrend[len(nexttrend)-1] == 1:
            maxlow = max(lowprice, maxlow);
            if highma[0] < maxlow and data[i][1] < data[i-1][2]:
                trend.append(1);
                nexttrend.append(0);
                minhigh = data[i-1][0]
        else:
            minhigh = min(highprice, minhigh);
            if lowma[0] > minhigh and data[i][1] < data[i-1][0]:
                trend.append(0);
                nexttrend.append(1);
                maxlow = lowprice
        if trend[len(trend)-1] == 0:
            if not math.isnan(trend[len(trend)-2]) and trend[len(trend)-2] != 0:
                if math.isnan(down[len(down)-2]):
                    up.append(down[len(down-1)]);
                else:
                    up.append(down[len(down)-2]);
            else:
                if math.isnan(up[len(up)-2]):
                    up.append(maxlow);
                else:
                    up.append(max(up[len(up)-2], maxlow));
            direction = 'long';
            atrHigh = up[len(up)-1] + dev;
            atrLow = up[len(up)-1] - dev;
        else:
            if not math.isnan(trend[len(trend)-2] and trend[len(trend)-2] != 1):
                if math.isnan(up[len(up)-2]):
                    down.append(up[len(up)-1]);
                else:
                    down.append(up[len(up)-2]);
            else:
                if math.isnan(down[len(down)-2]):
                    down.append(minhigh);
                else:
                    down.append(min(minhigh, down[len(down)-2]));
            direction = 'short';
            atrHigh = down[len(down)-1] + dev;
            atrLow = down[len(down)-1] - dev;
        if trend[len(trend)-1] == 0:
            out.append([atrHigh, up[len(up)-1], atrLow, direction]);
        else:
            out.append([atrHigh, down[len(down)-1], atrLow, direction]);
    return out
def log(d):
    return list(map(lambda x: math.log(x),d));
def exp(d):
    return list(map(lambda x: math.exp(x),d));
