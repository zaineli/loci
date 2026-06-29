| family | rows | MAE | bias | max \|err\| | within 0.05 | module MAE | MAE, error law alone (E) | sampling floor | flagged (MAE > 0.05) |
|---|---|---|---|---|---|---|---|---|---|
| load | 36 | 0.007 | +0.005 | 0.026 | 100% | 0.003 | 0.096 | 0.007 |  |
| lifelog | 15 | 0.007 | +0.007 | 0.017 | 100% | 0.005 | 0.206 | 0.009 |  |
| cards 7/12/4 | 15 | 0.008 | +0.008 | 0.021 | 100% | 0.006 | 0.274 | 0.009 |  |
| Nh 200 | 9 | 0.004 | +0.002 | 0.011 | 100% | 0.004 | 0.366 | 0.009 |  |
| Nh 800 | 9 | 0.008 | +0.005 | 0.034 | 100% | 0.004 | 0.096 | 0.007 |  |
| periods 5,6,7 | 12 | 0.009 | +0.006 | 0.023 | 100% | 0.004 | 0.205 | 0.008 |  |
| Ns 500 | 9 | 0.007 | +0.005 | 0.020 | 100% | 0.004 | 0.140 | 0.010 |  |
| Ns 2000 | 9 | 0.010 | +0.009 | 0.022 | 100% | 0.005 | 0.247 | 0.006 |  |
| mask 0.25 | 12 | 0.006 | +0.004 | 0.017 | 100% | 0.003 | 0.142 | 0.007 |  |
| mask 0.5 | 12 | 0.004 | +0.002 | 0.011 | 100% | 0.004 | 0.202 | 0.009 |  |
| pinv read | 12 | 0.029 | +0.029 | 0.043 | 100% | 0.023 | 0.127 | 0.009 |  |
| periods 3,4,5,7 | 6 | 0.007 | +0.000 | 0.012 | 100% | 0.007 | 0.450 | 0.010 |  |
| ALL | 156 | 0.009 | +0.007 | 0.043 | 100% | 0.006 | 0.187 | 0.008 |  |

Kendall tau of the placement order within each (family, seed), mean over 33 groups: 0.99

Worst 10 conditions (T-relu predicted vs measured address accuracy):

| condition | P | predicted | measured | error | error-law-only (E) |
|---|---|---|---|---|---|
| pinv read|load 0.8|seed 30|error | 800 | 0.586 | 0.544 | +0.043 | 0.713 |
| pinv read|load 0.8|seed 30|oracle | 800 | 0.660 | 0.620 | +0.040 | 0.741 |
| pinv read|load 0.8|seed 32|error | 800 | 0.578 | 0.540 | +0.038 | 0.700 |
| pinv read|load 0.8|seed 31|error | 800 | 0.575 | 0.538 | +0.037 | 0.700 |
| pinv read|load 0.8|seed 31|kmeans | 800 | 0.394 | 0.359 | +0.035 | 0.528 |
| Nh 800|load 0.8|seed 32|random | 800 | 0.570 | 0.536 | +0.034 | 0.787 |
| pinv read|load 0.8|seed 31|oracle | 800 | 0.667 | 0.633 | +0.033 | 0.735 |
| pinv read|load 0.8|seed 30|kmeans | 800 | 0.394 | 0.367 | +0.027 | 0.527 |
| load|load 0.5|seed 30|random | 500 | 0.761 | 0.735 | +0.026 | 0.956 |
| pinv read|load 0.8|seed 32|kmeans | 800 | 0.334 | 0.309 | +0.025 | 0.500 |

Pre-stated bars: overall MAE <= 0.03 and >= 90% of rows within 0.05. Result: MAE 0.009, 100% within 0.05 -> **PASS**.
