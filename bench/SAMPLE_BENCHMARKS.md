# Benchmarks

Reference library: `ta_py`. Δ column shows max absolute delta vs reference.
Lower median is faster. `(skip)` = adapter not available or unsupported.

| Case | Size | ta_py | stdlib | talib | pandas_ta | finta |
|------|------|------|------|------|------|------|
| sma | 1000 | 58.01µs | 130.33µs Δ=3.7e-13 | (skip) | (skip) | (skip) |
| sma | 10000 | 580.56µs | 1.33ms Δ=6.4e-13 | (skip) | (skip) | (skip) |
| smma | 1000 | 71.36µs | (skip) | (skip) | (skip) | (skip) |
| smma | 10000 | 723.19µs | (skip) | (skip) | (skip) | (skip) |
| wma | 1000 | 79.05µs | (skip) | (skip) | (skip) | (skip) |
| wma | 10000 | 799.48µs | (skip) | (skip) | (skip) | (skip) |
| vwma | 1000 | 117.72µs | (skip) | (skip) | (skip) | (skip) |
| vwma | 10000 | 1.20ms | (skip) | (skip) | (skip) | (skip) |
| ema | 1000 | 142.92µs | (skip) | (skip) | (skip) | (skip) |
| ema | 10000 | 1.48ms | (skip) | (skip) | (skip) | (skip) |
| variance | 1000 | 118.22µs | (skip) | (skip) | (skip) | (skip) |
| variance | 10000 | 1.20ms | (skip) | (skip) | (skip) | (skip) |
| std | 1000 | 118.90µs | 1.08µs | (skip) | (skip) | (skip) |
| std | 10000 | 1.21ms | 1.10µs | (skip) | (skip) | (skip) |
| bands | 1000 | 236.95µs | 1.14ms Δ=1.4e-10 | (skip) | (skip) | (skip) |
| bands | 10000 | 2.54ms | 12.56ms Δ=7.1e-10 | (skip) | (skip) | (skip) |
| bandwidth | 1000 | 278.67µs | (skip) | (skip) | (skip) | (skip) |
| bandwidth | 10000 | 3.11ms | (skip) | (skip) | (skip) | (skip) |
| keltner | 1000 | 438.35µs | (skip) | (skip) | (skip) | (skip) |
| keltner | 10000 | 4.74ms | (skip) | (skip) | (skip) | (skip) |
| don | 1000 | 352.14µs | (skip) | (skip) | (skip) | (skip) |
| don | 10000 | 3.51ms | (skip) | (skip) | (skip) | (skip) |
| pr | 1000 | 308.50µs | (skip) | (skip) | (skip) | (skip) |
| pr | 10000 | 3.08ms | (skip) | (skip) | (skip) | (skip) |
| stoch | 1000 | 504.39µs | (skip) | (skip) | (skip) | (skip) |
| stoch | 10000 | 5.17ms | (skip) | (skip) | (skip) | (skip) |
| rsi | 1000 | 190.72µs | (skip) | (skip) | (skip) | (skip) |
| rsi | 10000 | 2.00ms | (skip) | (skip) | (skip) | (skip) |
| mfi | 1000 | 136.72µs | (skip) | (skip) | (skip) | (skip) |
| mfi | 10000 | 1.36ms | (skip) | (skip) | (skip) | (skip) |
| atr | 1000 | 200.30µs | (skip) | (skip) | (skip) | (skip) |
| atr | 10000 | 2.03ms | (skip) | (skip) | (skip) | (skip) |
| macd | 1000 | 334.79µs | (skip) | (skip) | (skip) | (skip) |
| macd | 10000 | 3.39ms | (skip) | (skip) | (skip) | (skip) |
