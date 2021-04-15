def median(data, l=0):
    l = (l) if l > 0 else len(data); pl = []; med = [];
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            tmp = pl.copy();
            tmp.sort(reverse=True);
            med.append(tmp[round((len(tmp)-1) / 2)]);
            pl = pl[1:];
    return med;
def rsi(data, l=0):
    l = (l) if l > 0 else len(data); pl = []; rs = [];
    for i in range(1, len(data)):
        pl.append(data[i] - data[i - 1]);
        if (len(pl)) >= l:
            gain = 0; loss = 0;
            for a in range(len(pl)):
                if pl[a] > 0: gain += pl[a];
                if pl[a] < 0: loss += abs(pl[a]);
            gain /= l; loss /= l; f = 100 - 100 / (1 + (gain / loss)) if loss != 0 else 100;
            rs.append(f);
            pl = pl[1:];
    return rs;
def sma(data, l=0):
    l = (l) if l > 0 else len(data) - 1; pl = []; sm = [];
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            sm.append(sum(pl) / l);
            pl = pl[1:];
    return sm;
def smma(data, l=0):
    l = (l) if l > 0 else len(data) - 1; pl = []; sm = [];
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            average = 0;
            for a in range(len(pl)): average += pl[a];
            f = average / l if len(sm) <= 0 else (average - sm[len(sm) - 1]) / l;
            sm.append(f);
            pl = pl[1:];
    sm = sm[1:];
    return sm;
def wma(data, l=0):
    l = (l) if l > 0 else len(data) - 1; pl = []; wm = []; weight = [];
    for i in range(1, l+1):
        weight.append(i);
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            average = 0;
            for q in range(1, len(pl)+1):
                average += pl[q-1] * q / sum(weight);
            wm.append(average);
            pl = pl[1:];
    return wm;
def ema(data, l=0):
    l = (l) if l > 0 else len(data) - 1; pl = []; em = []; weight = 2 / (l + 1);
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            f = sum(pl) / l if len(em) <= 0 else (data[i] - em[len(em) - 1]) * weight + em[len(em) - 1]
            em.append(f);
            pl = pl[1:];
    return em;
def vwma(data, l=0):
    l = (l) if l > 0 else len(data) - 1; pl = []; wm = [];
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
def hull(data, l=0):
    l = (l) if l > 0 else len(data) - 1; pl = []; hma = []; ewma = wma(data.copy(), l); sqn = round(l**(1/2));
    first = wma(data.copy(), round(l / 2));
    first = first[-(len(ewma)-len(first)):];
    for i in range(len(ewma)):
        pl.append((first[i] * 2) - ewma[i]);
        if (len(pl)) >= sqn:
            h = wma(pl.copy(), sqn);
            hma.append(h[len(h) - 1]);
    return hma;
def kama(data, l1=0, l2=0, l3=0):
    l1 = l1 if l1 > 0 else 10; l2 = l2 if l2 > 0 else 2; l3 = l3 if l3 > 0 else 30;
    ka = sma(data.copy(), l1); ka = [ka[len(ka)-1]];
    for i in range(l1 + 1, len(data)):
        vola = 0; change = abs(data[i] - data[i - l1]);
        for a in range(1, l1):
            vola += abs(data[i-a] - data[(i-a)-1]);
        sc = (change/vola * (2/(l2+1) - 2/(l3+1) + 2/(l3+1))) ** 2;
        ka.append(ka[len(ka)-1] + sc * (data[i] - ka[len(ka)-1]));
    return ka;
def macd(data, l1=0, l2=0):
    l1 = l1 if l1 > 0 else 12; l2 = l2 if l2 > 0 else 26;
    if l1 > l2: [l1, l2] = [l2, l1];
    em1 = ema(data.copy(), l1); em2 = ema(data.copy(), l2);
    em1 = em1[-(len(em2) - len(em1)):]; emf = [];
    for i in range(len(em1)):
        emf.append(em1[i] - em2[i]);
    return emf;
def variance(data, l1=0):
    mean = sma(data.copy(), l1);
    return sum((x - mean[len(mean)-1]) ** 2 for x in data) / len(data);
def std(data, l1=0):
    l1 = (l1) if l1 > 0 else len(data);
    std = variance(data.copy(), l1) ** (1/2)
    return std;
def bands(data, l1=0, l2=0):
    l1 = l1 if l1 > 0 else 14; l2 = l2 if l2 > 0 else 1;
    pl = []; deviation = []; boll = [];
    sm = sma(data.copy(), l1);
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l1:
            devi = std(pl.copy(), l1);
            deviation.append(devi);
            pl = pl[1:];
    for i in range(len(sm)):
        boll.append([sm[i] + deviation[i] * l2, sm[i], sm[i] - deviation[i] * l2]);
    return boll;
