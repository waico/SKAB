![skab](docs/pictures/skab.png)

❗️❗️❗️**The current version of SKAB (v0.9) contains 34 datasets with collective anomalies. But the upcoming update to v1.0 (probably up to the winter of 2021) will contain 300+ additional files with point and collective anomalies. It will make SKAB one of the largest changepoint-containing benchmarks, especially in the technical field.**

# About SKAB [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/waico/SKAB/graphs/commit-activity) [![DOI](https://img.shields.io/badge/DOI-10.34740/kaggle/dsv/1693952-blue.svg)](https://doi.org/10.34740/KAGGLE/DSV/1693952) [![License: GPL v3.0](https://img.shields.io/badge/License-GPL%20v3.0-green.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
We propose the [Skoltech](https://www.skoltech.ru/en) Anomaly Benchmark (SKAB) designed for evaluating the anomaly detection algorithms. SKAB allows working with two main problems (there are two markups for anomalies):
1. Outlier detection (anomalies considered and marked up as single-point anomalies);
2. Changepoint detection (anomalies considered and marked up as collective anomalies).

SKAB consists of the following artifacts:
1. [Datasets](#datasets);
2. [Leaderboards](#leaderboards) for oultier detection and changepoint detection problems;
3. Python [modules](https://github.com/waico/SKAB/blob/master/utils/evaluating.py) for algorithms’ evaluation;
4. Python [notebooks](#notebooks) with anomaly detection algorithms.

The IIot testbed system is located in the Skolkovo Institute of Science and Technology (Skoltech).
All the details regarding the testbed and the experimenting process are presented in the following artifacts:
- Position paper (*currently submitted for publication*);
- [Slides](https://drive.google.com/open?id=1dHUevwPp6ftQCEKnRgB4KMp9oLBMSiDM) about the project.

<a name="datasets"></a>
# Datasets
The SKAB v0.9 corpus contains 35 individual data files in .csv format. Each file represents a single experiment and contains a single anomaly. The dataset represents a multivariate time series collected from the sensors installed on the testbed. The [data](data/) folder contains datasets from the benchmark. The structure of the data folder is presented in the [structure](./data/README.md) file. Columns in each data file are following:
- `datetime` - Represents dates and times of the moment when the value is written to the database (YYYY-MM-DD hh:mm:ss)
- `Accelerometer1RMS` - Shows a vibration acceleration (Amount of g units)
- `Accelerometer2RMS` - Shows a vibration acceleration (Amount of g units)
- `Current` - Shows the amperage on the electric motor (Ampere)
- `Pressure` - Represents the pressure in the loop after the water pump (Bar)
- `Temperature` - Shows the temperature of the engine body (The degree Celsius)
- `Thermocouple` - Represents the temperature of the fluid in the circulation loop (The degree Celsius)
- `Voltage` - Shows the voltage on the electric motor (Volt)
- `RateRMS` - Represents the circulation flow rate of the fluid inside the loop (Liter per minute)
- `anomaly` - Shows if the point is anomalous (0 or 1)
- `changepoint` - Shows if the point is a changepoint for collective anomalies (0 or 1)

Exploratory Data Analysis (EDA) for SKAB is presented at [kaggle](https://www.kaggle.com/newintown/eda-example) (Russian comments included, English version is upcoming).

<a name="leaderboards"></a>
# Leaderboards
Here we propose the leaderboards for SKAB v0.9 both for outlier and changepoint detection problems. You can also present and evaluate your algorithm using SKAB on [kaggle](https://www.kaggle.com/yuriykatser/skoltech-anomaly-benchmark-skab). Leaderboards are also available at paperswithcode.com: [CPD problem](https://paperswithcode.com/sota/change-point-detection-on-skab).

❗️All results (excl. ruptures and CPDE) are calculated for out-of-box algorithms without any hyperparameters tuning.

### Outlier detection problem
*Sorted by F1; for F1 bigger is better; both for FAR (False Alarm Rate) and MAR (Missing Alarm Rate) less is better*  
| Algorithm | F1 | FAR, % | MAR, %
|---|---|---|---|
Perfect detector | 1 | 0 | 0
Conv-AE |***0.79*** | 13.69 | ***17.77***
MSET |0.73 | 20.82 | 20.08
LSTM-AE |0.68 | 14.24 | 35.56
T-squared+Q (PCA) | 0.67 | 13.95 | 36.32
LSTM | 0.64 | 15.4 | 39.93
MSCRED | 0.64 | 13.56 | 41.16
LSTM-VAE | 0.56 | 9.13 | 55.03
T-squared | 0.56 | 12.14 | 52.56
Autoencoder | 0.45 | 7.56 | 66.57
Isolation forest | 0.4 | ***6.86*** | 72.09
Null detector | 0  | 0 | 100

### Changepoint detection problem 
*Sorted by NAB (standard); for all metrics bigger is better*  
*The current leaderboard is obtained with the window size for the NAB detection algorithm equal to 30 sec.*  
| Algorithm | NAB (standard) | NAB (lowFP) | NAB (LowFN) |
|---|---|---|---|
Perfect detector | 100 | 100 | 100 
Isolation forest | ***37.53*** | 17.09 | ***45.02***
MSCRED | 28.74 | ***23.43*** | 31.21
LSTM | 27.09 | 11.06 | 32.68
T-squared+Q (PCA) | 26.71 | 22.42 | 28.32
ruptures** | 24.1 | 21.69 | 25.04
CPDE*** | 23.07 | 20.52 | 24.35
LSTM-AE |22.12 | 20.01 | 23.21
LSTM-VAE | 19.17 | 15.39 | 20.98
T-squared | 17.87 | 3.44 | 23.2
ArimaFD | 16.06 | 14.03 | 17.12
Autoencoder | 15.59 | 0.78 | 20.91
MSET | 12.71 | 11.04 | 13.6
Conv-AE | 10.09 | 8.62 | 10.83
Null detector | 0 | 0 | 0

** The best algorithm (shown) is BinSeg with Mahalanobis cost function. The results are obtained in an unsupervised manner except for knowing by the algorithms the total amount of chagepoint to look for. The full results of various changepoint detection algorithms and ensembles are presented [here](https://github.com/YKatser/CPDE).

*** The best aggregation function (shown) is WeightedSum with MinAbs scaling function.

<a name="notebooks"></a>
# Notebooks
The [notebooks](notebooks/) folder contains python notebooks with the code for the proposed leaderboard results reproducing. This folder also contains short description of the algorithms and references to papers and code.

We have calculated the results for following common anomaly detection algorithms:
- Hotelling's T-squared statistics;
- Hotelling's T-squared statistics + Q statistics based on PCA;
- Isolation forest;
- LSTM-based NN (LSTM);
- Feed-Forward Autoencoder;
- LSTM Autoencoder (LSTM-AE);
- LSTM Variational Autoencoder (LSTM-VAE);
- Convolutional Autoencoder (Conv-AE);
- Multi-Scale Convolutional Recurrent Encoder-Decoder (MSCRED);
- Multivariate State Estimation Technique (MSET).

Additionally on the leaderboard were shown the results of the following algorithms:
- [ArimaFD](https://github.com/waico/arimafd);
- [ruptures](https://github.com/deepcharles/ruptures) changepoint detection (CPD) algorithms;
- ruptures-based [changepoint detection ensemble (CPDE) algorithms](https://github.com/YKatser/CPDE).

# Citation
Please cite our project in your publications if it helps your research.
```
Iurii D. Katser and Vyacheslav O. Kozitsin, “Skoltech Anomaly Benchmark (SKAB).” Kaggle, 2020, doi: 10.34740/KAGGLE/DSV/1693952.
```
Or in BibTeX format:
```
@misc{skab,
  author = {Katser, Iurii D. and Kozitsin, Vyacheslav O.},
  title = {Skoltech Anomaly Benchmark (SKAB)},
  year = {2020},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/dsv/1693952}},
  DOI = {10.34740/KAGGLE/DSV/1693952}
}
```

# Notable mentions
SKAB is acknowledged by some ML resources.
<details>
  <summary>List of links</summary>
  
  - [Anomaly Detection Learning Resources](https://github.com/yzhao062/anomaly-detection-resources#34-datasets)
  - [awesome-TS-anomaly-detection](https://github.com/rob-med/awesome-TS-anomaly-detection#benchmark-datasets)
  - [List of datasets for machine-learning research](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research#Anomaly_data)
  - [paperswithcode.com](https://paperswithcode.com/dataset/skab)
  - [Google datasets](https://datasetsearch.research.google.com/search?query=skoltech%20anomaly%20benchmark&docid=IIIE4VWbqUKszygyAAAAAA%3D%3D)
  - [Industrial ML Datasets](https://github.com/nicolasj92/industrial-ml-datasets)

</details>
