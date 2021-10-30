import ta_py as ta
median = ta.median([4, 6, 3, 1, 2, 5], 4);
assert median == [3, 2, 2];
kmeans = ta.kmeans([2, 3, 4, 5, 3, 5, 7, 8, 6, 8, 6, 4, 2, 6], 4);
assert kmeans == [[ 4, 5, 5, 4 ], [ 7, 6, 6, 6 ], [ 8, 8 ], [ 2, 3, 3, 2 ]];
normal = ta.normalize([5,4,9,4], 0.1);
assert normal == [0.2222222222222222, 0.06349206349206349, 0.8571428571428571, 0.06349206349206349];
denormal = ta.denormalize([5,4,9,4], [0.2222222222222222, 0.06349206349206349, 0.8571428571428571, 0.06349206349206349, 0.4444444444444444], 0.1);
assert denormal == [5.0, 4.0, 9.0, 4.0, 6.4];
mad = ta.mad([3, 7, 5, 4, 3, 8, 9], 6);
assert mad == [1.0, 2.0];
aad = ta.aad([4, 6, 8, 6, 8, 9, 10, 11], 7);
assert aad == [1.6734693877551021, 1.469387755102041];
ssd = ta.ssd([7, 6, 5, 7, 9, 8, 3, 5, 4], 7);
assert ssd == [4.869731585445518, 4.9856938190329, 5.3718844791323335];
er = ta.er([0.02, -0.01, 0.03, 0.05, -0.03]);
assert er == 0.011934565489708282;
rsi = ta.rsi([1,2,3,4,5,6,7,5], 6);
assert rsi == [100.0, 71.42857142857143];
wrsi = ta.wrsi([1, 2, 3, 4, 5, 6, 7, 5, 6], 6);
assert wrsi == [100, 71.42857142857143, 75.60975609756098];
sma = ta.sma([1, 2, 3, 4, 5, 6, 10], 6);
assert sma == [3.5, 5];
smma = ta.smma([1, 2, 3, 4, 5, 6, 10], 5);
assert smma == [3.4, 4.92];
wma = ta.wma([69, 68, 66, 70, 68], 4);
assert wma == [68.3, 68.2];
pwma = ta.pwma([17, 26, 23, 29, 20], 4);
assert pwma == [24.09090909090909, 25.18181818181818];
hwma = ta.hwma([54, 51, 86, 42, 47], 4);
assert hwma == [56.199999999999996, 55.0];
ema = ta.ema([1, 2, 3, 4, 5, 6, 10], 6);
assert ema == [3.5, 5.357142857142857];
wsma = ta.wsma([1, 2, 3, 4, 5, 6, 10], 6);
assert wsma == [3.5, 4.583333333333333];
tsi = ta.tsi([1.32, 1.27, 1.42, 1.47, 1.42, 1.45, 1.59], 3, 2, 2);
assert tsi == [[0.3268608414239478, 0.32038834951456274], [0.5795418491021003, 0.7058823529411765]];
vwma = ta.vwma([[1, 59], [1.1, 82], [1.21, 27], [1.42, 73], [1.32, 42]], 4);
assert vwma == [1.184771784232365, 1.258794642857143];
hull = ta.hull([6, 7, 5, 6, 7, 4, 5, 7], 6);
assert hull == [4.761904761904762, 5.476190476190476];
kama = ta.kama([8, 7, 8, 9, 7, 9], 2, 4, 8);
assert kama == [8, 8.64, 8.377600000000001, 8.377600000000001];
macd = ta.macd([1, 2, 3, 4, 5, 6, 14], 3, 6);
assert macd == [1.5, 3];
variance = ta.variance([6, 7, 2, 3, 5, 8, 6, 2], 7);
assert variance == [3.918367346938776, 5.061224489795919];
std = ta.std([1, 2, 3], 3);
assert std <= 0.816496580928;
normsinv = ta.normsinv(0.4732);
assert normsinv == -0.06722824471054376;
bands = ta.bands([1, 2, 3, 4, 5, 6], 5, 2);
assert bands == [[5.82842712474619, 3.0, 0.1715728752538097], [6.82842712474619, 4.0, 1.1715728752538097]];
bandwidth = ta.bandwidth([1, 2, 3, 4, 5, 6], 5, 2);
assert bandwidth == [1.8856180831641265, 1.414213562373095];
atr = ta.atr([[3,2,1], [2,2,1], [4,3,1], [2,2,1]], 3);
assert atr == [2.0, 1.6666666666666667, 2.111111111111111, 1.7407407407407407];
keltner = ta.keltner([[3,2,1], [2,2,1], [4,3,1], [2,2,1], [3,3,1]], 5, 1);
assert keltner == [[3.932266666666667, 2.066666666666667, 0.20106666666666695]];
cor = ta.cor([1, 2, 3, 4, 5, 2], [1, 3, 2, 4, 6, 3]);
assert cor >= 0.8808929232684737
dif = ta.dif(0.75, 0.5);
assert dif == 0.5;
draw = ta.drawdown([1,2,3,4,2,3]);
assert draw == -0.5;
aroon_up = ta.aroon_up([5, 4, 5, 2], 3);
assert aroon_up == [100.0, 50.0];
aroon_down = ta.aroon_down([2, 5, 4, 5], 3);
assert aroon_down == [0.0, 50.0];
aroon_osc = ta.aroon_osc([2, 5, 4, 5], 3);
assert aroon_osc == [50, 50];
mfi = ta.mfi([[19, 13], [14, 38], [21, 25], [32, 17]], 3);
assert mfi == [41.53846153846154, 45.578231292517];
roc = ta.roc([1, 2, 3, 4], 3);
assert roc == [2, 1];
cop = ta.cop([3, 4, 5, 3, 4, 5, 6, 4, 7, 5, 4, 7, 5], 4, 6, 5);
assert cop == [0.3755555555555556, 0.23666666666666666];
kst = ta.kst([8, 6, 7, 6, 8, 9, 7, 5, 6, 7, 6, 8, 6, 7, 6, 8, 9, 9, 8, 6, 4, 6, 5, 6, 7, 8, 9], 5, 5, 7, 5, 10, 5, 15, 7, 4);
assert kst == [[-0.6828231292517006, -0.5174886621315192], [-0.2939342403628118, -0.5786281179138322], [0.3517800453514739, -0.35968820861678]];
obv = ta.obv([[25200, 10], [30000, 10.15], [25600, 10.17], [32000, 10.13]]);
assert obv == [0, 30000, 55600, 23600];
vwap = ta.vwap([[127.21, 89329], [127.17, 16137], [127.16, 23945]], 2);
assert vwap == [127.20387973375304, 127.16402599670675];
mom = ta.mom([1, 1.1, 1.2, 1.24, 1.34], 4);
assert mom == [0.24, 0.24];
mom_osc = ta.mom_osc([1, 1.2, 1.3, 1.3, 1.2, 1.4], 4);
assert mom_osc == [0.0, 3.8461538461538494];
bop = ta.bop([[4, 5, 4, 5], [5, 6, 5, 6], [6, 8, 5, 6]], 2);
assert bop == [1, 0.5];
fi = ta.fi([[1.4, 200], [1.5, 240], [1.1, 300], [1.2, 240], [1.5, 400]], 4);
assert fi == [12.00000000000001];
asi = ta.asi([[7, 6, 4], [9, 7, 5], [9, 8, 6]]);
assert asi == [0, -12.5];
ao = ta.ao([[6, 5], [8, 6], [7, 4], [6, 5], [7, 6], [9, 8]], 2, 5);
assert ao == [0.0, 0.9000000000000004];
pr = ta.pr([2, 1, 3, 1, 2], 3);
assert pr == [0, -100, -50];
lsma = ta.lsma([5, 6, 6, 3, 4, 6, 7], 6);
assert lsma == [4.714285714285714, 5.761904761904762];
don = ta.don([[6, 2], [5, 2], [5, 3], [6, 3], [7, 4], [6, 3]], 5);
assert don == [[7, 4.5, 2], [7, 4.5, 2]];
percentile = ta.percentile([[6,4,7],[5,3,6],[7,5,8]], 0.5);
assert percentile == [6,4,7];
ichimoku = ta.ichimoku([[6,3,2], [5,4,2], [5,4,3], [6,4,3], [7,6,4], [6,5,3], [7,6,5], [7,5,3], [8,6,5], [9,7,6], [8,7,6], [7,5,5],[6,5,4],[6,5,3],[6,3,2], [5,4,2]], 2, 4, 6, 4);
assert ichimoku == [[7, 6, 10.5, 6, 5], [7.5, 6, 7.5, 5.5, 6], [6.5, 7, 8, 5, 5]];
stoch = ta.stoch([[3,2,1], [2,2,1], [4,3,1], [2,2,1]], 2, 1, 1);
assert stoch == [[66.66666666666667, 66.66666666666667], [33.333333333333336, 33.333333333333336]];
ha = ta.ha([[3, 4, 2, 3], [3, 6, 3, 5], [5, 5, 2, 3]]);
assert ha == [[3.0, 4.0, 2.0, 3.0], [3.0, 4.0, 2.0, 3.0], [3.0, 6.0, 3.0, 4.25], [3.625, 5.0, 2.0, 3.75]];
ren = ta.ren([[8, 6], [9, 7], [9, 8], [13, 10]], 2);
assert ren == [[8.0, 10.0, 8.0, 10.0], [10.0, 12.0, 10.0, 12.0]];
chaikin = ta.chaikin_osc([[2,3,4,6],[5,5,5,4],[5,4,3,7],[4,3,3,4],[6,5,4,6],[7,4,3,6]],2,4);
assert chaikin == [-1.6666666666666665, -0.28888888888888886, -0.7362962962962962];
enve = ta.envelope([6,7,8,7,6,7,8,7,8,7,8,7,8], 11, 0.05);
assert enve == [[7.540909090909091, 7.181818181818182, 6.822727272727272], [7.636363636363637, 7.2727272727272725, 6.909090909090908]];
frac = ta.fractals([[7,6],[8,6],[9,6],[8,5],[7,4],[6,3],[7,4],[8,5]]);
assert frac == [[False, False], [False, False], [True, False], [False, False], [False, False], [False, True], [False, False], [False, False]];
hi = ta.recent_high([4,5,6,7,8,9,8,7,8,9,10,3,2,1], 3);
assert hi == {'index': 10, 'value': 10};
lo = ta.recent_low([1,4,5,6,4,3,2,3,4,3,5,7,8,8,5], 4);
assert lo == {'index': 6, 'value': 2};
sup = ta.support([4,3,2,5,7,6,5,4,7,8,5,4,6,7,5]);
sup = sup['calculate'](9);
assert sup == 4.0;
res = ta.resistance([5,7,5,5,4,6,5,4,6,5,4,3,2,4,3,2,1]);
res = res['calculate'](4);
assert res == 6.428571428571429;
ac = ta.ac([[6, 5], [8, 6], [7, 4], [6, 5], [7, 6], [9, 8]], 2, 4);
assert ac == [0.125, 0.5625];
fib = ta.fib(1, 2);
assert fib == [1,1.236,1.3820000000000001,1.5,1.6179999999999999,1.786,2,2.6180000000000003,3.618,4.618,5.236];
alli = ta.alligator([8,7,8,9,7,8,9,6,7,8,6,8,10,8,7,9,8,7,9,6,7,9]);
assert alli == [[7.217569412835686, 6.985078985569999, 6.456171046541722], [7.171597633136094, 7.119368115440011, 6.719144767291392]];
gato = ta.gator([8,7,8,9,7,8,9,6,7,8,6,8,10,8,7,9,8,7,9,6,7,9]);
assert gato == [[0.23249042726568714, -0.5289079390282767], [0.05222951769608297, -0.4002233481486188]];
standard = ta.standardize([6,4,6,8,6]);
assert standard == [0, -1.5811388300841895, 0, 1.5811388300841895, 0];
fisher = ta.fisher([8,6,8,9,7,8,9,8,7,8,6,7], 9);
assert fisher == [[-0.20692076425551026, 0.11044691579009712], [-0.3930108381942109, -0.20692076425551026]];
winratio = ta.winratio([0.01,0.02,-0.01,-0.03,-0.015,0.005]);
assert winratio == 0.5;
avgwin = ta.avgwin([0.01,0.02,-0.01,-0.03,-0.015,0.005]);
assert avgwin == 0.011666666666666665;
avgloss = ta.avgloss([0.01,0.02,-0.01,-0.03,-0.015,0.005]);
assert avgloss == -0.018333333333333333;
print('Test Passed');
