![skab](docs/pictures/skab.png)

🛠🛠🛠**The testbed is under repair right now. Unfortunately, we can't tell exactly when it will be ready and we be able to continue data collection. Information about it will be in the repository. Sorry for the delay.**

❗️❗️❗️The current version of SKAB (v0.9) contains 34 datasets with collective anomalies. But the update to v1.0 will contain 300+ additional files with point and collective anomalies. It will make SKAB one of the largest changepoint-containing benchmarks, especially in the technical field.

# About SKAB [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/waico/SKAB/graphs/commit-activity) [![DOI](https://img.shields.io/badge/DOI-10.34740/kaggle/dsv/1693952-blue.svg)](https://doi.org/10.34740/KAGGLE/DSV/1693952) [![License: GPL v3.0](https://img.shields.io/badge/License-GPL%20v3.0-green.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
We propose the [Skoltech](https://www.skoltech.ru/en) Anomaly Benchmark (SKAB) designed for evaluating the anomaly detection algorithms. SKAB allows working with two main problems (there are two markups for anomalies):
1. Outlier detection (anomalies considered and marked up as single-point anomalies)
2. Changepoint detection (anomalies considered and marked up as collective anomalies)

SKAB consists of the following artifacts:
1. [Datasets](#datasets)
2. [Leaderboards](#leaderboards) for oultier detection and changepoint detection problems
3. Python modules for algorithms’ evaluation (now evaluation modules are being imported from [TSAD](https://github.com/waico/tsad) framework, while the details regarding the evaluation process are presented [here](https://github.com/waico/tsad/blob/main/examples/Evaluating.ipynb))
4. Python [modules](#src) with algorithms’ implementation
5. Python [notebooks](#notebooks) with anomaly detection pipeline implementation for various algorithms

All the details about SKAB are presented in the following artifacts:
- Position paper (*currently submitted for publication*)
- Talk about the project: [English](https://youtu.be/hjzuKeNYUho) version and [Russian](https://www.youtube.com/watch?v=VLmmYGc4v2c) version
- Slides about the project: [English](https://drive.google.com/open?id=1dHUevwPp6ftQCEKnRgB4KMp9oLBMSiDM) version and [Russian](https://drive.google.com/file/d/1gThPCNbEaIxhENLm-WTFGO_9PU1Wdwjq/view?usp=share_link) version

# Datasets
The SKAB v0.9 corpus contains 35 individual data files in .csv format (datasets). The [data](data/) folder contains datasets from the benchmark. The structure of the data folder is presented in the [structure](./data/README.md) file. Each dataset represents a single experiment and contains a single anomaly. The datasets represent a multivariate time series collected from the sensors installed on the testbed. Columns in each data file are following:
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

Exploratory Data Analysis (EDA) for SKAB is presented [here](https://github.com/waico/SKAB/blob/master/notebooks/EDA.ipynb). Russian version of EDA is available on [kaggle](https://www.kaggle.com/newintown/eda-example).

ℹ️We have also made a *SKAB teaser* that is a small dataset collected separately but from the same testbed. SKAB teaser is made just for learning/teaching purposes and contains only 4 collective anomalies. All the information is available on [kaggle](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab-teaser).

# Leaderboards
Here we propose the leaderboards for SKAB v0.9 for both outlier and changepoint detection problems. You can also present and evaluate your algorithm using SKAB on [kaggle](https://www.kaggle.com/yuriykatser/skoltech-anomaly-benchmark-skab). Leaderboards are also available at paperswithcode.com: [CPD problem](https://paperswithcode.com/sota/change-point-detection-on-skab).

❗️All results (excl. ruptures and CPDE) are calculated for out-of-box algorithms without any hyperparameters tuning.

### Outlier detection problem
*Sorted by F1; for F1 bigger is better; both for FAR (False Alarm Rate) and MAR (Missing Alarm Rate) less is better*  
| Algorithm | F1 | FAR, % | MAR, %
|---|---|---|---|
Perfect detector | 1 | 0 | 0
Conv-AE |***0.79*** | 13.69 | ***17.77***
MSET |0.73 | 20.82 | 20.08
LSTM-AE |0.68 | 14.24 | 35.56
T-squared+Q (PCA-based) | 0.67 | 13.95 | 36.32
Vanilla LSTM | 0.64 | 15.4 | 39.93
MSCRED | 0.64 | 13.56 | 41.16
LSTM-VAE | 0.56 | 9.13 | 55.03
T-squared | 0.56 | 12.14 | 52.56
Vanilla AE | 0.45 | 7.56 | 66.57
Isolation forest | 0.4 | ***6.86*** | 72.09
Null detector | 0  | 0 | 100

### Changepoint detection problem 
*Sorted by NAB (standard); for all metrics bigger is better*  
*The current leaderboard is obtained with the window size for the NAB detection algorithm equal to 30 sec.*  

| Algorithm | NAB (standard) | NAB (lowFP) | NAB (LowFN) |
|---|---|---|---|
|Perfect detector | 100 | 100 | 100 |
|Isolation forest | ***37.53*** | 17.09 | ***45.02***|
|MSCRED | 28.74 | ***23.43*** | 31.21|
|Vanilla LSTM | 27.09 | 11.06 | 32.68|
|T-squared+Q (PCA-based) | 26.71 | 22.42 | 28.32|
|ruptures** | 24.1 | 21.69 | 25.04|
|CPDE*** | 23.07 | 20.52 | 24.35|
|LSTM-AE |22.12 | 20.01 | 23.21|
|LSTM-VAE | 19.17 | 15.39 | 20.98|
|T-squared | 17.87 | 3.44 | 23.2|
|ArimaFD | 07.67 | 01.97 | 11.04 |
|Vanilla AE | 15.59 | 0.78 | 20.91|
|MSET | 12.71 | 11.04 | 13.6|
|Conv-AE | 10.09 | 8.62 | 10.83|
|Null detector | 0 | 0 | 0|

** The best algorithm (shown) is BinSeg with Mahalanobis cost function. The results are obtained in an unsupervised manner except for knowing by the algorithms the total amount of chagepoint to look for. The full results of various changepoint detection algorithms and ensembles are presented [here](https://github.com/YKatser/CPDE).

*** The best aggregation function (shown) is WeightedSum with MinAbs scaling function.

# Notebooks
The [notebooks](notebooks/) folder contains jupyter notebooks with the code for the proposed leaderboard results reproducing. We have calculated the results for following commonly known anomaly detection algorithms:
- Isolation forest - *Outlier detection algorithm based on Random forest concept*
- Vanilla LSTM - *NN with LSTM layer*
- Vanilla AE - *Feed-Forward Autoencoder*
- LSTM-AE - *LSTM Autoencoder*
- LSTM-VAE - *LSTM Variational Autoencoder*
- Conv-AE - *Convolutional Autoencoder*
- MSCRED - *Multi-Scale Convolutional Recurrent Encoder-Decoder*
- MSET - *Multivariate State Estimation Technique*

Additionally on the leaderboard were shown the externally calculated results of the following algorithms:
- [ArimaFD](https://github.com/waico/arimafd) - *ARIMA-based fault detection algorithm*
- [T-squared](http://github.com/YKatser/ControlCharts/tree/main/examples) - *Hotelling's T-squared statistics*
- [T-squared+Q (PCA-based)](http://github.com/YKatser/ControlCharts/tree/main/examples) - *Hotelling's T-squared statistics + Q statistics based on PCA*
- [ruptures](https://github.com/deepcharles/ruptures) - *Changepoint detection (CPD) algorithms from ruptures package*
- [CPDE](https://github.com/YKatser/CPDE) - *Ruptures-based changepoint detection ensemble (CPDE) algorithms*

Details regarding the algorithms, including short description, references to scientific papers and code of the initial implementation is available in [this readme](https://github.com/waico/SKAB/tree/master/notebooks#anomaly-detection-algorithms).

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
  - etc.

</details>
