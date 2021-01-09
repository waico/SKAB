![skab](skab.png)

# About SKAB [![DOI](https://img.shields.io/badge/DOI-10.34740/kaggle/dsv/1693952-blue.svg)](https://doi.org/10.34740/KAGGLE/DSV/1693952)
We propose the Skoltech Anomaly Benchmark (SKAB) designed for evaluating the anomaly detection algorithms. SKAB allows working with two main problems (there are two markups for anomalies):
* Outlier detection (anomalies considered and marked up as single-point anomalies)
* Changepoint detection (anomalies considered and marked up as collective anomalies)

SKAB consists of the following artifacts:
* Datasets.
* Leaderboard (scoreboard).
* Python modules for algorithms’ evaluation.
* Baselines: python notebooks with several well-known anomaly detection algorithms.

The IIot testbed system is located in the Skolkovo Institute of Science and Technology (Skoltech).
All the details regarding the testbed and the experimenting process are presented in the following artifacts:
- Position paper (*currently submitted for publication*)
- [Slides about the project](https://drive.google.com/open?id=1dHUevwPp6ftQCEKnRgB4KMp9oLBMSiDM)

# Datasets
The SKAB v0.9 corpus contains 35 individual data files in .csv format. Each file represents a single experiment and contains a single anomaly. The dataset represents a multivariate time series collected from the sensors installed on the testbed. The [data](data/) folder containes datasets from the benchmark. The structure of the data folder is presented in [structure](./data/README.md) file.

# Leaderboard (Scoreboard)
Here we propose the leaderboard for SKAB v0.9 both for outlier and changepoint detection problems. You can also present and evaluate your algorithm using SKAB on [kaggle](https://www.kaggle.com/yuriykatser/skoltech-anomaly-benchmark-skab).
The results in the tables are calculated in the python notebooks from the [baselines](baselines/) folder.

## Outlier detection problem
*Sorted by FAR, both for FAR and MAR less is better*  
| Algorithm | FAR, % | MAR, % |
|---|---|---|
Perfect detector | 0 | 0
Null detector | 0 | 100
T-squared+Q (PCA) | ***5.09*** | 86.1
Isolation forest | 6.86 | 72.09
Autoencoder | 7.56 | 66.57
T-squared | 12.14 | 52.56
LSTM | 14.4 | ***40.44***

## Changepoint detection problem
*Sorted by NAB (standart), for all metrics bigger is better*
| Algorithm | NAB (standart) | NAB (lowFP) | NAB (LowFN) |
|---|---|---|---|
Perfect detector | 100 | 100 | 100 
Isolation forest | ***37.53*** | ***17.09*** | ***45.02***
LSTM | 25.82 | 9.06 | 31.83
T-squared | 17.87 | 3.44 | 23.2
Autoencoder | 15.59 | 0.78 | 20.91
T-squared+Q (PCA) | 5.83 | 4.8 | 6.1
Null detector | 0 | 0 | 0

# Baselines
The baselines folder contains python notebooks with the code for the proposed leaderboard results reproducing.
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
