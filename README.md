# SkAB Skoltech Anomaly Benchmark
This repository contains the data provided by the IIot testbed system for evaluating Anomaly and Changepoint Detection algorithms.

## data
The data folder containes datasets from the benchmark. The structure of the data folder is following:
- **anomaly-free** contains a normal or anomaly-free mode for algorithms tunning.
- **valve1**
- **valve2**
- **other**

## baselines
The baselines folder contains Ipython notebooks with the code for the initial leaderboard results reproducing.
We have calculated the results for five quite common anomaly detection algorithms:
- Hotelling's T-squared statistics;
- Hotelling's T-squared statistics + Q statistics based on PCA;
- Isolation forest;
- LSTM-based RNN;
- Feed-Forward Autoencoder.

## utils
The utils folder contains needed functions for the experiments and code for algorithms evaluation.

## Leaderboard (Scoreboard)
Here we propose an initial leaderboard for SkAB v1.0 both for outlier and changepoint detection problems.

*to be inserted soon*