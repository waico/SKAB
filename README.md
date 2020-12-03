![skab](skab.png)

# SKAB Skoltech Anomaly Benchmark
This repository contains the data provided by the IIot testbed system for evaluating Anomaly and Changepoint Detection algorithms.

## baselines
The baselines folder contains Ipython notebooks with the code for the initial leaderboard results reproducing.
We have calculated the results for five quite common anomaly detection algorithms:
- Hotelling's T-squared statistics;
- Hotelling's T-squared statistics + Q statistics based on PCA;
- Isolation forest;
- LSTM-based RNN;
- Feed-Forward Autoencoder.

## Leaderboard (Scoreboard)
Here we propose an initial leaderboard for SKAB v1.0 both for outlier and changepoint detection problems.

**Outlier detection problem**
| Algorithm | FAR, % | MAR, % |
|---|---|---|
Perfect detector | 0 | 0 
T-squared | 12.14 | 52.56 
T-squared+Q (PCA) | ***5.09*** | 86.1 
LSTM | 14.4 | ***40.44***
Autoencoder | 7.56 | 66.57
Isolation forest | 6.86 | 72.09 
Null detector | 0 | 100

**Changepoint detection problem**
| Algorithm | NAB (standart) | NAB (lowFP) | NAB (LowFN) |
|---|---|---|---|
Perfect detector | 100 | 100 | 100 
T-squared | 17.87 | 3.44 | 23.2
T-squared+Q (PCA) | 5.83 | 4.8 | 6.1
LSTM | 25.82 | 9.06 | 31.83
Autoencoder | 15.59 | 0.78 | 20.91
Isolation forest | ***37.53*** | ***17.09*** | ***45.02***
Null detector | 0 | 0 | 0

## Citation
Please cite our project in your publications if it helps your research.
```
Iurii D. Katser and Vyacheslav O. Kozitsin, “Skoltech Anomaly Benchmark (SKAB).” Kaggle, 2020, doi: 10.34740/KAGGLE/DSV/1693952.
```

## data
The data folder containes datasets from the benchmark. The structure of the data folder is presented in [structure](structure.md) file.


