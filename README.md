![skab](skab.png)

# About SKAB
We propose the Skoltech Anomaly Benchmark (SKAB) designed for evaluating the anomaly detection algorithms. SKAB allows working with two main problems (there are two markups for anomalies):
* Outlier detection (anomalies considered and marked up as single-point anomalies)
* Changepoint detection (anomalies considered and marked up as collective anomalies)

SKAB consists of the following artifacts:
* Datasets.
* Leaderboard (scoreboard).
* Python modules for algorithms’ evaluation.
* Baselines: python notebooks with several well-known anomaly detection algorithms.

The IIot testbed system is located in the Skolkovo Institute of Science and Technology.
All the details regarding the testbed and the experimenting process are presented in the position paper (*currently submitted for publication*).

# Datasets
The SKAB v1.0 corpus contains 35 individual data files in .csv format. Each file represents a single experiment and contains a single anomaly. The dataset represents a multivariate time series collected from the sensors installed on the testbed. The [data](data/) folder containes datasets from the benchmark. The structure of the data folder is presented in [structure](structure.md) file.

# Leaderboard (Scoreboard)
Here we propose the leaderboard for SKAB v1.0 both for outlier and changepoint detection problems. You can also present and evaluate your algorithm using SKAB on [kaggle](https://www.kaggle.com/yuriykatser/skoltech-anomaly-benchmark-skab).

## Outlier detection problem
| Algorithm | FAR, % | MAR, % |
|---|---|---|
Perfect detector | 0 | 0 
T-squared+Q (PCA) | ***5.09*** | 86.1 
Isolation forest | 6.86 | 72.09 
Autoencoder | 7.56 | 66.57
T-squared | 12.14 | 52.56 
LSTM | 14.4 | ***40.44***
Null detector | 0 | 100

## Changepoint detection problem
| Algorithm | NAB (standart) | NAB (lowFP) | NAB (LowFN) |
|---|---|---|---|
Perfect detector | 100 | 100 | 100 
Isolation forest | ***37.53*** | ***17.09*** | ***45.02***
T-squared | 17.87 | 3.44 | 23.2
Autoencoder | 15.59 | 0.78 | 20.91
LSTM | 25.82 | 9.06 | 31.83
T-squared+Q (PCA) | 5.83 | 4.8 | 6.1
Null detector | 0 | 0 | 0

# Baselines
The baselines folder contains python notebooks with the code for the initial leaderboard results reproducing.
We have calculated the results for five quite common anomaly detection algorithms:
- Hotelling's T-squared statistics;
- Hotelling's T-squared statistics + Q statistics based on PCA;
- Isolation forest;
- LSTM-based RNN;
- Feed-Forward Autoencoder.

# Citation
Please cite our project in your publications if it helps your research.
```
Iurii D. Katser and Vyacheslav O. Kozitsin, “Skoltech Anomaly Benchmark (SKAB).” Kaggle, 2020, doi: 10.34740/KAGGLE/DSV/1693952.
```



