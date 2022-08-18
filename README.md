# Technical Analysis (ta.py)

ta.py is a Python package for dealing with financial technical analysis.

## Installation

#### pip
Use the package manager pip to install ta.py.

```bash
pip install ta_py
```
## Usage
```python
import ta_py as ta;
```
## Examples
#### Moving Averages
- [Simple Moving Average](#sma)
- [Smoothed Moving Average](#smma)
- [Weighted Moving Average](#wma)
- [Exponential Moving Average](#ema)
- [Hull Moving Average](#hull)
- [Least Squares Moving Average](#lsma)
- [Volume Weighted Moving Average](#vwma)
- [Wilder's Smoothing Moving Average](#wsma)
- [Parabolic Weighted Moving Average](#pwma)
- [Hyperbolic Weighted Moving Average](#hwma)
- [Kaufman Adaptive Moving Average](#kama)
- [Custom Weighted Moving Average](#cwma)
#### Indicators
- [Moving Average Convergence / Divergence](#macd)
- [Relative Strength Index](#rsi)
- [Wilder's Relative Strength Index](#wrsi)
- [True Strength Index](#tsi)
- [Balance Of Power](#bop)
- [Force Index](#fi)
- [Accumulative Swing Index](#asi)
- [Alligator Indicator](#alli)
- [Williams %R](#pr)
- [Stochastics](#stoch)
- [Fibonacci Retracement](#fib)
- [Bollinger Bandwidth](#bandwidth)
- [Ichimoku Cloud](#ichi)
- [Average True Range](#atr)
- [Aroon Up](#aroon-up)
- [Aroon Down](#aroon-down)
- [Money Flow Index](#mfi)
- [Rate Of Change](#roc)
- [Coppock Curve](#cop)
- [Know Sure Thing](#kst)
- [On-Balance Volume](#obv)
- [Volume-Weighted Average Price](#vwap)
- [Fractals](#fractals)
- [Crossover](#cross)
- [Momentum](#mom)
- [HalfTrend](#half)
- [ZigZag](#zigzag)
- [Parabolic SAR](#psar)
- [SuperTrend](#supertrend)
- [Elder Ray Index](#elderray)
#### Oscillators
- [Alligator Oscillator](#gator)
- [Chande Momentum Oscillator](#mom_osc)
- [Chaikin Oscillator](#chaikin_osc)
- [Aroon Oscillator](#aroon-osc)
- [Awesome Oscillator](#ao)
- [Accelerator Oscillator](#ac)
- [Fisher Transform](#fish)
#### Bands
- [Bollinger Bands](#bands)
- [Keltner Channels](#kelt)
- [Donchian Channels](#don)
- [Fibonacci Bollinger Bands](#fibbands)
- [Envelope](#env)
#### Statistics
- [Standard Deviation](#std)
- [Variance](#variance)
- [Normal CDF](#ncdf)
- [Inverse Normal Distribution](#normsinv)
- [Monte Carlo Simulation](#sim)
- [Percentile](#perc)
- [Correlation](#cor)
- [Covariance](#cov)
- [Percentage Difference](#dif)
- [Expected Return](#er)
- [Abnormal Return](#ar)
- [Kelly Criterion](#kelly)
- [Winratio](#winratio)
- [Average Win](#avgwin)
- [Average Loss](#avgloss)
- [Drawdown](#drawdown)
- [Median](#median)
- [Recent High](#rh)
- [Recent Low](#rl)
- [Median Absolute Deviation](#mad)
- [Average Absolute Deviation](#aad)
- [Standard Error](#stderr)
- [Sum Squared Differences](#ssd)
- [Logarithm](#log)
- [Exponent](#exp)
- [Normalize](#norm)
- [Denormalize](#dnorm)
- [Normalize Pair](#normp)
- [Normalize From](#normf)
- [Standardize](#standard)
- [Z-Score](#zscore)
- [K-means Clustering](#kmeans)
#### Chart Types
- [Heikin Ashi](#ha)
- [Renko](#ren)
#### Experimental
- [Support Line](#sup)
- [Resistance Line](#res)
### Moving Averages
#### <a id="sma"></a>Simple Moving Average (SMA)
```python
data = [1, 2, 3, 4, 5, 6, 10];
length = 6; # default = 14
ta.sma(data, length);
# output (array)
# [3.5, 5]
```
#### <a id="smma"></a>Smoothed Moving Average (SMMA)
```python
data = [1, 2, 3, 4, 5, 6, 10];
length = 5; # default = 14
ta.smma(data, length);
# output (array)
# [3.4, 4.92]
```
#### <a id="wma"></a>Weighted Moving Average (WMA)
```python
data = [69, 68, 66, 70, 68];
length = 4; # default = 14
ta.wma(data, length);
# output (array)
# [68.3, 68.2]
```
#### <a id="ema"></a>Exponential Moving Average (EMA)
```python
data = [1, 2, 3, 4, 5, 6, 10];
length = 6; # default = 12
ta.ema(data, length);
# output (array)
# [3.5, 5.357]
```
#### <a id="hull"></a>Hull Moving Average
```python
data = [6, 7, 5, 6, 7, 4, 5, 7];
length = 6; # default = 14
ta.hull(data, length);
# output (array)
# [4.76, 5.48]
```
#### <a id="lsma"></a>Least Squares Moving Average (LSMA)
```python
data = [5, 6, 6, 3, 4, 6, 7];
length = 6; # default = 25
ta.lsma(data, length);
# output (array)
# [4.714, 5.761]
```
#### <a id="vwma"></a>Volume Weighted Moving Average (VWMA)
```python
data = [[1, 59], [1.1, 82], [1.21, 27], [1.42, 73], [1.32, 42]]; # [price, volume (quantity)]
length = 4; # default = 20
ta.vwma(data, length);
# output (array)
# [1.185, 1.259]
```
#### <a id="wsma"></a>Wilder's Smoothing Moving Average
```python
data = [1, 2, 3, 4, 5, 6, 10];
length = 6; # default = 14
ta.wsma(data, length);
# output (array)
# [3.5, 4.58]
```
#### <a id="pwma"></a>Parabolic Weighted Moving Average
```python
 data = [17, 26, 23, 29, 20];
 length = 4; # default = 14
ta.pwma(data, length);
# output (array)
# [24.09, 25.18]
```
#### <a id="hwma"></a>Hyperbolic Weighted Moving Average
```python
data = [54, 51, 86, 42, 47];
length = 4; # default = 14
ta.hwma(data, length);
# output (array)
# [56.2, 55.0]
```
#### <a id="kama"></a>Kaufman Adaptive Moving Average (KAMA)
```python
data = [8, 7, 8, 9, 7, 9];
length1 = 2; # default = 10
length2 = 4; # default = 2
length3 = 8; # default = 30
ta.kama(data, length1, length2, length3);
# output (array)
# [8, 8.64, 8.57, 8.57]
```
#### <a id="cwma"></a>Custom Weighted Moving Average
```python
data = [69,68,66,70,68,69];
weights = [1,2,3,5,8];
ta.cwma(data, weights);
# output (array)
# [68.26315789473684, 68.52631578947368]
```
### Indicators
#### <a id="macd"></a>Moving Average Convergence / Divergence (MACD)
```python
data = [1, 2, 3, 4, 5, 6, 14];
length1 = 3; # default = 12
length2 = 6; # default = 26
ta.macd(data, length1, length2);
# output (array)
# [1.5, 3]
```
#### <a id="rsi"></a>Relative Strength Index (RSI)
```python
data = [1, 2, 3, 4, 5, 6, 7, 5];
length = 6; # default = 14
ta.rsi(data, length);
# output (array)
# [100, 100, 66.667]
```
#### <a id="wrsi"></a>Wilder's Relative Strength Index
```python
data = [1, 2, 3, 4, 5, 6, 7, 5, 6];
length = 6; # default = 14
ta.wrsi(data, length);
# output (array)
# [100, 71.43, 75.61]
```
#### <a id="tsi"></a>True Strength Index (TSI)
```python
data = [1.32, 1.27, 1.42, 1.47, 1.42, 1.45, 1.59];
longlength = 3; # default = 25
shortlength = 2; # default = 13
signallength = 2; # default = 13
ta.tsi(data, longlength, shortlength, signallength);
# output (array)
# [[0.327, 0.320], [0.579, 0.706]]
# [strength line, signal line]
```
#### <a id="bop"></a>Balance Of Power
```python
data = [[4, 5, 4, 5], [5, 6, 5, 6], [6, 8, 5, 6]]; # [open, high, low, close]
length = 2; # default = 14
ta.bop(data, length);
# output (array)
# [1, 0.5]
```
#### <a id="fi"></a>Force Index
```python
data = [[1.4, 200], [1.5, 240], [1.1, 300], [1.2, 240], [1.5, 400]]; # [close, volume]
length = 4; # default = 13
ta.fi(data, length);
# output (array)
# [0.0075]
```
#### <a id="asi"></a>Accumulative Swing Index
```python
data = [[7, 6, 4], [9, 7, 5], [9, 8, 6]]; # [high, close, low]
ta.asi(data);
# output (array)
# [0, -12.5]
```
#### <a id="alli"></a>Alligator Indicator
```python
data = [8,7,8,9,7,8,9,6,7,8,6,8,10,8,7,9,8,7,9,6,7,9];
# defaults shown
jawlength = 13;
teethlength = 8;
liplength = 5;
jawshift = 8;
teethshift = 5;
lipshift = 3;
ta.alligator(data, jawlength, teethlength, liplength, jawshift, teethshift, lipshift);
# output (array)
# [jaw, teeth, lips]
```
#### <a id="pr"></a>Williams %R
```python
data = [2, 1, 3, 1, 2];
length = 3; # default = 14
ta.pr(data, length);
# output (array)
# [-0, -100, -50]
```
#### <a id="stoch"></a>Stochastics
```python
data = [[3,2,1], [2,2,1], [4,3,1], [2,2,1]]; # [high, close, low]
length = 2; # default = 14
smoothd = 1; # default = 3
smoothk = 1; # default = 3
ta.stoch(data, length, smoothd, smoothk);
# output (array)
# [[66.667, 66.667], [33.336, 33.336]]
# [kline, dline]
```
#### <a id="fib"></a>Fibonacci Retracement
```python
start = 1;
end = 2;
ta.fib(start, end);
# output (array)
# [1, 1.236, 1.382, 1.5, 1.618, 1.786, 2, 2.618, 3.618, 4.618, 5.236]
```
#### <a id="bandwidth"></a>Bollinger Bandwidth
```python
data = [1, 2, 3, 4, 5, 6];
length = 5; # default = 14
deviations = 2; # default = 1
ta.bandwidth(data, length, deviations);
# output (array)
# [1.886, 1.344]
```
#### <a id="ichi"></a>Ichimoku Cloud
```python
data = [[6, 3, 2], [5, 4, 2], [5, 4, 3], [6, 4, 3], [7, 6, 4], [6, 5, 3]]; # [high, close, low]
length1 = 9; # default = 9
length2 = 26; # default = 26
length3 = 52; # default = 52
displacement = 26; # default = 26
ta.ichimoku(data, length1, length2, length3, displacement);
# output (array)
# [conversion line, base line, leading span A, leading span B, lagging span]
```
#### <a id="atr"></a>Average True Range (ATR)
```python
data = [[3,2,1], [2,2,1], [4,3,1], [2,2,1]]; # [high, close, low]
length = 3; # default = 14
ta.atr(data, length);
# output (array)
# [2, 1.667, 2.111, 1.741]
```
#### <a id="aroon-up"></a>Aroon Up
```python
data = [5, 4, 5, 2];
length = 3; # default = 10
ta.aroon_up(data, length);
# output (array)
# [100.0, 50.0]
```
#### <a id="aroon-down"></a>Aroon Down
```python
data = [2, 5, 4, 5];
length = 3; # default = 10
ta.aroon_down(data, length);
# output (array)
# [0.0, 50.0]
```
#### <a id="mfi"></a>Money Flow Index
```python
data = [[19, 13], [14, 38], [21, 25], [32, 17]]; # [buy volume, sell volume]
length = 3; # default = 14
ta.mfi(data, length);
# output (array)
# [41.54, 45.58]
```
#### <a id="roc"></a>Rate Of Change
```python
data = [1, 2, 3, 4];
length = 3; # default = 14
ta.roc(data, length);
# output (array)
# [2, 1]
```
#### <a id="cop"></a>Coppock Curve
```python
data = [3, 4, 5, 3, 4, 5, 6, 4, 7, 5, 4, 7, 5];
length1 = 4; # (ROC period 1) default = 11
length2 = 6; # (ROC period 2) default = 14
length3 = 5; # (WMA smoothing period) default = 10
ta.cop(data, length1, length2, length3);
# output (array)
# [0.376, 0.237]
```
#### <a id="kst"></a>Know Sure Thing
```python
data = [8, 6, 7, 6, 8, 9, 7, 5, 6, 7, 6, 8, 6, 7, 6, 8, 9, 9, 8, 6, 4, 6, 5, 6, 7, 8, 9];
# roc sma #1
r1 = 5; # default = 10
s1 = 5; # default = 10
# roc sma #2
r2 = 7; # default = 15
s2 = 5; # default = 10
# roc sma #3
r3 = 10; # default = 20
s3 = 5; # default = 10
# roc sma #4
r4 = 15; # default = 30
s4 = 7; # default = 15
# signal line
sig = 4; # default = 9
ta.kst(data, r1, s1, r2, s2, r3, s3, r4, s4, sig);
# output (array)
# [[-0.68, -0.52], [-0.29, -0.58], [0.35, -0.36]]
# [kst line, signal line]
```
#### <a id="obv"></a>On-Balance Volume
```python
data = [[25200, 10], [30000, 10.15], [25600, 10.17], [32000, 10.13]]; # [asset volume, close price]
ta.obv(data);
# output (array)
# [0, 30000, 55600, 23600]
```
#### <a id="vwap"></a>Volume-Weighted Average Price
```python
data = [[127.21, 89329], [127.17, 16137], [127.16, 23945]]; # [average price, volume (quantity)]
length = 2; # default = len(length)
ta.vwap(data, length);
# output (array)
# [127.204, 127.164]
```
#### <a id="fractals"></a>Fractals
```python
data = [[7,6],[8,6],[9,6],[8,5],[7,4],[6,3],[7,4],[8,5]]; # [high, low]
ta.fractals(data);
# output (array, same length as input)
# [[false, false],[false,false],[true,false],[false,false],[false,false],[false,true],[false,false],[false,false]]
# [upper fractal, lower fractal]
```
#### <a id="cross"></a>Crossover (golden cross)
```python
fastdata = [3,4,5,4,3]; # short period gets spliced when longer
slowdata = [4,3,2,3,4];
ta.cross(fastdata, slowdata);
# output (array)
# [{index: 1, cross True}, {index: 4, cross: False}]
# cross is true when fastdata is greater than the slowdata
```
#### <a id="mom"></a>Momentum
```python
data = [1, 1.1, 1.2, 1.24, 1.34];
length = 4; # default = 10
percentage = false; # default = false (true returns percentage)
ta.mom(data, length, percentage);
# output (array)
# [0.24, 0.24]
```
#### <a id="half"></a>HalfTrend
```python
# experimental (untested) function (may change in the future), ported from:
# https://www.tradingview.com/script/U1SJ8ubc-HalfTrend/
# data = [high, close, low]
data = [[100,97,90],[101,98,94],[103,96,92],[106,100,95],[110,101,100],[112,110,105],[110,100,90],[103,100,97],[95,90,85],[94,80,80],[90,82,81],[85,80,70]];
atrlen = 6;
amplitude = 3;
deviation = 2;
ta.halftrend(data, atrlen, amplitude, deviation);
# output (array)
# [
#   [ 115.14, 105, 94.86, 'long' ],
#   [ 100.77, 90, 79.22, 'long' ],
#   [ 116.32, 105, 93.68, 'long' ],
#   [ 101.1, 90, 78.89, 'long' ],
#   [ 116.25, 105, 93.75, 'long' ],
#   [ 99.77, 90, 80.23, 'long' ]
# ]
```
#### <a id="zigzag"></a>ZigZag
```python
# Based on high / low
data = [[10,9], [12,10], [14,12], [15,13], [16,15], [11,10], [18,15]]; # [high, low]
percentage = 0.25; # default = 0.05
ta.zigzag(data, percentage);
# output (array)
# [9, 10.75, 12.5, 14.25, 16, 10, 18]
```
```python
# Based on close
data = [6,7,8,9,10,12,9,8,5,3,3,3,5,7,8,9,11];
percentage = 0.05;
ta.zigzag(data, percentage);
# output (array)
# [6, 7.2, 8.4, 9.6, 10.8, 12.0, 10.5, 9.0, 7.5, 6.0, 4.5, 3.0, 4.6, 6.2, 7.8, 9.4, 11.0]
```
#### <a id="psar"></a>Parabolic SAR
```python
data = [[82.15,81.29],[81.89,80.64],[83.03,81.31],[83.30,82.65],[83.85,83.07],[83.90,83.11],[83.33,82.49],[84.30,82.3],[84.84,84.15],[85,84.11],[75.9,74.03],[76.58,75.39],[76.98,75.76],[78,77.17],[70.87,70.01]];
step = 0.02;
max = 0.2;
ta.psar(data, step, max);
# output (array)
# [81.29,82.15,80.64,80.64,80.7464,80.932616,81.17000672,81.3884061824,81.67956556416,82.0588176964608,85,85,84.7806,84.565588,84.35487624000001]
```
#### <a id="supertrend"></a>SuperTrend
```python
data = [[3,2,1], [2,2,1], [4,3,1], [2,2,1]]; # [high, close, low]
length = 3;
multiplier = 0.5;
ta.supertrend(data, length, multiplier);
# output (array)
# [[5.56,1.44],[3.37,0.63]]
# [up, down]
```
#### <a id="elderray"></a>Elder Ray Index
```python
data = [6,5,4,7,8,9,6,8];
length = 7;
ta.elderray(data, length);
# output (array)
# [[2.57,-2.43],[2.29,-2.71]]
# [bull, bear]
```
### Oscillators
#### <a id="gator"></a>Alligator Oscillator
```python
data = [8,7,8,9,7,8,9,6,7,8,6,8,10,8,7,9,8,7,9,6,7,9];
# defaults shown
jawlength = 13;
teethlength = 8;
liplength = 5;
jawshift = 8;
teethshift = 5;
lipshift = 3;
ta.gator(data, jawlength, teethlength, liplength, jawshift, teethshift, lipshift);
# output (array)
# [upper gator, lower gator]
```
#### <a id="mom_osc"></a>Chande Momentum Oscillator
```python
data = [1, 1.2, 1.3, 1.3, 1.2, 1.4];
length = 4; # default = 9
ta.mom_osc(data, length);
# output (array)
# [0.0, 3.85]
```
#### <a id="chaikin_osc"></a>Chaikin Oscillator
```python
data = [[2,3,4,6],[5,5,5,4],[5,4,3,7],[4,3,3,4],[6,5,4,6],[7,4,3,6]]; # [high, close, low, volume]
length1 = 2; # default = 3
length2 = 4; # default = 10
ta.chaikin_osc(data, length1, length2);
# output (array)
# [-1.667, -0.289, -0.736]
```
#### <a id="aroon-osc"></a>Aroon Oscillator
```python
data = [2, 5, 4, 5];
length = 3; # default = 25
ta.aroon_osc(data, length);
# output (array)
# [50.0, 50.0]
```
#### <a id="ao"></a>Awesome Oscillator
```python
data = [[6, 5], [8, 6], [7, 4], [6, 5], [7, 6], [9, 8]]; # [high, low]
shortlength = 2; # default = 5
longlength = 5; # default = 35
ta.ao(data, shortlength, longlength);
# output (array)
# [0, 0.9]
```
#### <a id="ac"></a>Accelerator Oscillator
```python
data = [[6, 5], [8, 6], [7, 4], [6, 5], [7, 6], [9, 8]]; # [high, low]
shortlength = 2; # default = 5
longlength = 4; # default = 35
ta.ac(data, shortlength, longlength);
# output (array)
# [-5.875, -6.125, -6.5]
```
#### <a id="fish"></a>Fisher Transform
```python
data = [8,6,8,9,7,8,9,8,7,8,6,7]; # high + low / 2
length = 9;
ta.fisher(data, length);
# output (array)
# [[-0.318, -0.11], [-0.449, -0.318], [-0.616, -0.449]] # [fisher, trigger]
```
### Bands
#### <a id="bands"></a>Bollinger Bands
```python
data = [1, 2, 3, 4, 5, 6];
length = 5; # default = 14
deviations = 2; # default = 1
ta.bands(data, length, deviations);
# output (array)
# [[5.828, 3, 0.172], [6.828, 4, 1.172]]
# [upper band, middle band, lower band]
```
#### <a id="kelt"></a>Keltner Channels
```python
data = [[3,2,1], [2,2,1], [4,3,1], [2,2,1], [3,3,1]]; # [high, close, low]
length = 5; # default = 14
deviations = 1; # default = 1
ta.keltner(data, length, deviations);
# output (array)
# [[3.93, 2.06, 0.20]]
# [upper band, middle band, lower band]
```
#### <a id="don"></a>Donchian Channels
```python
data = [[6, 2], [5, 2], [5, 3], [6, 3], [7, 4], [6, 3]]; # [high, low]
length = 5; # default = 20
ta.don(data, length);
# output (array)
# [[7, 4.5, 2], [7, 4.5, 2]]
# [upper band, base line, lower band]
```
#### <a id="fibbands"></a>Fibonacci Bollinger Bands
```python
data = [[1,59],[1.1,82],[1.21,27],[1.42,73],[1.32,42]];
length = 4;
deviations = 3;
ta.fibbands(data, length, deviations);
# output (array)
# [[highest band -> fibonacci levels -> lowest band]]
```
#### <a id="env"></a>Envelope
```python
data = [6,7,8,7,6,7,8,7,8,7,8,7,8];
length = 11, # default = 10
percentage = 0.05; # default = 0.005
ta.envelope(data, length, percentage);
# output (array)
# [[7.541, 7.182, 6.823], [7.636, 7.273, 6.909]]
# [upper band, base line, lower band]
```
### Statistics
#### <a id="std"></a>Standard Deviation
```python
data = [1, 2, 3];
length = 3; # default = len(length)
ta.std(data, length);
# output (float)
# 0.81649658092773
```
#### <a id="variance"></a>Variance
```python
data = [6, 7, 2, 3, 5, 8, 6, 2];
length = 7; # default = len(data)
ta.variance(data, length);
# output (array)
# [3.918, 5.061]
```
#### <a id="ncdf"></a>Normal CDF
```python
sample = 13;
mean = 10;
stdv = 2;
ta.ncdf(sample, mean, stdv);
# output (float)
# 0.9331737996110652
```
```python
zscore = 1.5;
ta.ncdf(zscore);
# output (float)
# 0.9331737996110652
```
#### <a id="normsinv"></a>Inverse Normal Distribution
```python
data = 0.4732;
ta.normsinv(data);
# output (float)
# -0.06722824471054376
```
#### <a id="sim"></a>Monte Carlo Simulation
```python
data = [6, 4, 7, 8, 5, 6];
length = 2; # default = 50
simulations = 100; # default = 1000
percentile = 0.5; # default = -1 (returns all raw simulations)
ta.sim(data, length, simulations, percentile)
# output (array)
# [6, 4, 7, 8, 5, 6, 5.96, 5.7]
```
#### <a id="perc"></a>Percentile
```python
data = [[6,4,7], [5,3,6], [7,5,8]];
percentile = 0.5;
ta.percentile(data, percentile);
# output (array)
# [6, 4, 7]
```
#### <a id="cor"></a>Correlation
```python
data1 = [1, 2, 3, 4, 5, 2];
data2 = [1, 3, 2, 4, 6, 3];
ta.cor(data1, data2);
# output (float)
# 0.8808929232684737
```
#### <a id="cov"></a>Covariance
```python
data1 = [12,13,25,39];
data2 = [67,45,32,21];
length = 4;
ta.covariance(data1, data2, 4);
# output (array)
# [-165.8125]
```
#### <a id="dif"></a>Percentage Difference
```python
newval = 0.75;
oldval = 0.5;
ta.dif(newval, oldval);
# output (float)
# 0.5
```
#### <a id="er"></a>Expected Return
```python
data = [0.02, -0.01, 0.03, 0.05, -0.03]; # historical return data
ta.er(data);
# output (float)
# 0.0119
```
#### <a id="ar"></a>Abnormal Return
```python
data = [0.02, -0.01, 0.03, 0.05, -0.03]; # historical return data
length = 3;
ta.ar(data, length);
# output (array)
# [0.037, -0.053]
```
#### <a id="kelly"></a>Kelly Criterion
```python
data = [0.01, 0.02, -0.01, -0.03, -0.015, 0.045, 0.005];
ta.kelly(data);
# output (float)
# 0.1443
```
#### <a id="winratio"></a>Winratio
```python
var data = [0.01, 0.02, -0.01, -0.03, -0.015, 0.005];
ta.winratio(data);
# output (float)
# 0.5
```
#### <a id="avgwin"></a> Average Win
```python
data = [0.01, 0.02, -0.01, -0.03, -0.015, 0.005];
ta.avgwin(data);
# output (float)
# 0.012
```
#### <a id="avgloss"></a> Average Loss
```python
data = [0.01, 0.02, -0.01, -0.03, -0.015, 0.005];
ta.avgloss(data);
# output (float)
# -0.018
```
#### <a id="drawdown"></a>Drawdown
```python
data = [1, 2, 3, 4, 2, 3];
ta.drawdown([1,2,3,4,2,3]);
# output (float)
# -0.5
```
#### <a id="median"></a>Median
```python
data = [4, 6, 3, 1, 2, 5];
length = 4; # default = len(data)
ta.median(data, length);
# output (array)
# [3, 2, 2]
```
#### <a id="rh"></a>Recent High
```python
data = [4,5,6,7,8,9,8,7,8,9,10,3,2,1];
lookback = 3; # No higher values after 3 periods? resets after each new high
ta.recent_high(data, lookback);
# output (dictionary)
# {'index': 10, 'value': 10}
```
#### <a id="rl"></a>Recent Low
```python
data = [1,4,5,6,4,3,2,3,4,3,5,7,8,8,5];
lookback = 4; # No lower values after 4 periods? resets after each new low
ta.recent_low(data, lookback);
# output (dictionary)
# {'index': 6, 'value': 2}
```
#### <a id="mad"></a>Median Absolute Deviation
```python
data = [3, 7, 5, 4, 3, 8, 9];
length = 6; # default = len(data)
ta.mad(data, length);
# output (array)
# [1, 2]
```
#### <a id="aad"></a>Average Absolute Deviation
```python
data = [4, 6, 8, 6, 8, 9, 10, 11];
length = 7; # default = len(data)
ta.aad(data, length);
# output (array)
# [1.673, 1.469]
```
#### <a id="stderr"></a>Standard Error
```python
data = [34, 54, 45, 43, 57, 38, 49];
size = 10; # default = len(data)
ta.se(data, size);
# output (float)
# 2.424
```
#### <a id="ssd"></a>Sum Squared Differences
```python
data = [7, 6, 5, 7, 9, 8, 3, 5, 4];
length = 7; # default = len(length)
ta.ssd(data, length);
# output (array)
# [4.87, 4.986, 5.372]
```
#### <a id="log"></a>Logarithm
```python
data = [5, 14, 18, 28, 68, 103];
ta.log(data);
# output (array)
# [1.61, 2.64, 2.89, 3.33, 4.22, 4.63]
```
#### <a id="exp"></a>Exponent
```python
data = [1.6, 2.63, 2.89, 3.33, 4.22, 4.63];
ta.exp(data);
# output (array)
# [4.95, 13.87, 17.99, 27.94, 68.03, 102.51]
```
#### <a id="norm"></a>Normalize
```python
data = [5,4,9,4];
margin = 0.1; # margin % (default = 0)
ta.normalize(data, margin);
# output (array)
# [0.22, 0.06, 0.86, 0.06]
```
#### <a id="dnorm"></a>Denormalize
```python
data = [5,4,9,4]; # original data || [highest, lowest]
norm = [0.22, 0.06, 0.86, 0.06, 0.44]; # normalized data
margin = 0.1; # margin % (default = 0)
ta.denormalize(data, norm, margin);
# output (array)
# [5 ,4, 9, 4, 6.4]
```
#### <a id="normp"></a>Normalize Pair
```python
pair1 = [10,12,11,13];
pair2 = [100,130,100,140];
ta.normalize_pair(pair1, pair2);
# output (array)
# [[55, 55], [66, 71.5], [60.5, 54.99], [71.5, 76.99]]
```
#### <a id="normf"></a>Normalize From
```python
data = [8, 12, 10, 11];
baseline = 100;
ta.normalize_from(data, baseline);
# output (array)
# [100, 150, 125, 137.5]
```
#### <a id="standard"></a>Standardize
```python
data = [6,4,6,8,6];
ta.standardize(data);
# output (array)
# [0, -1.581, 0, 1.581, 0]
```
#### <a id="zscore"></a>Z-Score
```python
data = [34,54,45,43,57,38,49];
length = 5;
ta.zscore(data, length);
# output (array)
# [1.266, -1.331, 0.408]
```
#### <a id="kmeans"></a>K-means Clustering
```python
data = [2, 3, 4, 5, 3, 5, 7, 8, 6, 8, 6, 4, 2, 6];
length = 4;
ta.kmeans(data, length);
# output (array)
# [[ 4, 5, 5, 4 ], [ 7, 6, 6, 6 ], [ 8, 8 ], [ 2, 3, 3, 2 ]]
```
### Chart types
#### <a id="ha"></a>Heikin Ashi
```python
data = [[3, 4, 2, 3], [3, 6, 3, 5], [5, 5, 2, 3]]; # [open, high, low, close]
ta.ha(data);
# output (array)
# [open, high, low, close]
# first 7-10 candles are unreliable
```
#### <a id="ren"></a>Renko
```python
data = [[8, 6], [9, 7], [9, 8]]; # [high, low]
bricksize = 3;
ta.ren(data, bricksize);
# output (array)
# [open, high, low, close]
```
### Experimental Functions
#### <a id="sup"></a>Support Line
```python
data = [4,3,2,5,7,6,5,4,7,8,5,4,6,7,5];
start = {"index": 2, "value": 2}; # default = recent_low(data, 25)
support = ta.support(data, start);
# output (dictionary)
# ['calculate'] = function(x) // calculates line at position x from start['index'] (= 0)
# ['slope'] = delta y per x
# ['lowest'] = lowest (start) value at x = 0
# ['index'] = (start) index of lowest value
# to get the line at the current candle / chart period
current = support['calculate'](len(data)-support['index']);
```
#### <a id="res"></a>Resistance Line
```python
data = [5,7,5,5,4,6,5,4,6,5,4,3,2,4,3,2,1];
start = {"index": 1, "value": 7}; # default = recent_high(data, 25)
resistance = ta.resistance(data, start);
# output (dictionary)
# ['calculate'] = function(x) // calculates line at position x from start['index'] (= 0)
# ['slope'] = delta y per x
# ['highest'] = highest (start) value
# ['index'] = (start) index of highest value
# to get the line at the current candle / chart period
current = resistance['calculate'](len(data)-resistance['index']);
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https:#choosealicense.com/licenses/mit/)
