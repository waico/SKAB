# ![skab](docs/pictures/skab.png)

🛠🛠🛠**The testbed is under repair right now. Unfortunately, we can't tell exactly when it will be ready and we be able to continue data collection. Information about it will be in the repository. Sorry for the delay.**

❗️❗️❗️The current version of SKAB (v0.9) contains 34 datasets with collective anomalies. But the update to v1.0 will contain 300+ additional files with point and collective anomalies. It will make SKAB one of the largest changepoint-containing benchmarks, especially in the technical field.

## About SKAB [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/waico/SKAB/graphs/commit-activity) [![DOI](https://img.shields.io/badge/DOI-10.34740/kaggle/dsv/1693952-blue.svg)](https://doi.org/10.34740/KAGGLE/DSV/1693952) [![License: GPL v3.0](https://img.shields.io/badge/License-GPL%20v3.0-green.svg)](https://www.gnu.org/licenses/gpl-3.0.html)

We propose the [Skoltech](https://www.skoltech.ru/en) Anomaly Benchmark (SKAB) designed for evaluating the anomaly detection core. SKAB allows working with two main problems (there are two markups for anomalies):

1. Outlier detection (anomalies considered and marked up as single-point anomalies)
2. Changepoint detection (anomalies considered and marked up as collective anomalies)

SKAB consists of the following artifacts:

1. [Datasets](#datasets)
2. [Proposed Leaderboard](#proposed-leaderboard) for outlier detection and changepoint detection problems
3. Python modules for algorithms’ evaluation (now evaluation modules are being imported from [TSAD](https://github.com/waico/tsad) framework, while the details regarding the evaluation process are presented [here](https://github.com/waico/tsad/blob/main/examples/Evaluating.ipynb))
4. Python [core](core/) with algorithms’ implementation
5. Python [notebooks](#notebooks) with anomaly detection pipeline implementation for various algorithms

All the details about SKAB are presented in the following artifacts:

- Position paper (*currently submitted for publication*)
- Talk about the project: [English](https://youtu.be/hjzuKeNYUho) version and [Russian](https://www.youtube.com/watch?v=VLmmYGc4v2c) version
- Slides about the project: [English](https://drive.google.com/open?id=1dHUevwPp6ftQCEKnRgB4KMp9oLBMSiDM) version and [Russian](https://drive.google.com/file/d/1gThPCNbEaIxhENLm-WTFGO_9PU1Wdwjq/view?usp=share_link) version

## Datasets

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

Exploratory Data Analysis (EDA) for SKAB is presented [here (tbd)]. Russian version of EDA is available on [kaggle](https://www.kaggle.com/newintown/eda-example).

ℹ️We have also made a *SKAB teaser* that is a small dataset collected separately but from the same testbed. SKAB teaser is made just for learning/teaching purposes and contains only 4 collective anomalies. All the information is available on [kaggle](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab-teaser).

## Proposed Leaderboard

This leaderboard shows performance of algorithms on test set, unlike leaderboard for SKAB v0.9 which evaluates both training and testing data all together. Moreover, the evaluated window of change points is to the right side of actual change point occurence which is in accordance with fact, that it should be impossible to capture event before it occurs. Lastly, the window size for the NAB detection algorithm is set to 60 seconds to reflect the dynamics of the transition as presented in the slides to enable detection of the start of the transition phase which is also marked as change-point.

You can present and evaluate your algorithm using SKAB on [kaggle](https://www.kaggle.com/yuriykatser/skoltech-anomaly-benchmark-skab). Leaderboards are also available at paperswithcode.com: [CPD problem](https://paperswithcode.com/sota/change-point-detection-on-skab).

Information about the metrics for anomaly detection and intuition behind the metrics selection can be found in [this](https://medium.com/@katser/a-review-of-anomaly-detection-metrics-with-a-lot-of-related-information-736d88774712) medium article.

### Outlier detection problem

*Sorted by F1; for F1 bigger is better; both for FAR (False Alarm Rate) and MAR (Missing Alarm Rate) less is better*
*Evaluated as binary classification problem.*

| Algorithm | F1 | FAR, % | MAR, %
|---|---|---|---
|Perfect detector | 1 | 0 | 0
|Conv-AE |0.78 | 13.55 | 28.02
|MSET |0.78 | 39.73 | 14.13
|T-squared+Q (PCA-based) | 0.76 | 26.62 | 24.92
|LSTM-AE |0.74 | 29.96 | 25.92
|T-squared | 0.66 | 19.21 | 42.6
|LSTM-VAE | 0.56 | 9.13 | 55.03
|Vanilla LSTM | 0.54 | 12.54 | 59.53
|MSCRED | 0.36 | 49.94 | 69.88
|Vanilla AE | 0.39 | 2.59 | 75.15
|Isolation forest | 0.29 | 2.56 | 82.89
|Null detector | 0  | 0 | 100

### Changepoint detection problem

*Sorted by NAB (standard); for NAB (standard), NAB (LowFP), NAB (LowFN) bigger is better, for Number of Missed CPs, Number of FPs lower is better*
*The current leaderboard is obtained with the window size for the NAB detection algorithm equal to 60 sec and to the right side of true change point.*

| Algorithm | NAB (standard) | NAB (LowFP) | NAB (LowFN) | Number of Missed CPs | Number of FPs
|---|---|---|---|---|---
|Perfect detector | 100 | 100 | 100 | 0 | 0
|MSCRED | 32.42 | 16.53 | 40.28 | 55 | 342
|Isolation forest | 26.16 | 19.5 | 30.82 | 76 | 135
|T-squared+Q (PCA-based) | 25.35 | 14.51 | 31.33 | 72 | 232
|Conv-AE | 23.61 | 21.54 | 27.55 | 82 | 23
|LSTM-AE | 23.51 | 20.11 | 25.91 | 88 | 69
|T-squared | 19.54 | 10.2 | 24.31 | 70 | 106
|MSET | 13.84 | 10.22 | 17.37 | 96 | 66
|Vanilla AE | 11.41 | 6.53 | 13.91 | 103 | 106
|Vanilla LSTM | 11.31 | -3.8 | 17.25 | 90 | 342
|ArimaFD | -0.09 | -0.17 | -0.06 | 127 | 2
|Null detector | 0 | 0 | 0 | - | -

## Notebooks

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

## Installation

1. install Python 3.10+ (tested on 3.10.13)

1. install [poetry](https://python-poetry.org/docs/) package manager
   - `brew install poetry`
   > Poetry installs dependencies and locks versions for deterministic installs. Poetry uses [Python's built-in `venv` module](https://docs.python.org/3/library/venv.html) to create virtual environments. It also uses PEP [517](https://peps.python.org/pep-0517) & [518](https://peps.python.org/pep-0518) specifications to build packages without requiring `setup.py` or `requirements.txt` files.

1. LightGBM base install
   - `brew install lightgbm`

1. install SKAB dependencies, see [pyproject.toml](pyproject.toml) for details
   - `poetry install`

1. confirm installation
   - `poetry show --tree` - shows all dependencies installed
   - `poetry env info` - displays information about the current environment (Python version, path, etc)
   - `poetry list` - lists all cli commands

## Citation

Please cite our project in your publications if it helps your research.

```bibtex
@misc{skab,
  author = {Katser, Iurii D. and Kozitsin, Vyacheslav O.},
  title = {Skoltech Anomaly Benchmark (SKAB)},
  year = {2020},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/dsv/1693952}},
  DOI = {10.34740/KAGGLE/DSV/1693952}
}
```

## Notable mentions

SKAB is acknowledged by some ML resources.

- [Anomaly Detection Learning Resources](https://github.com/yzhao062/anomaly-detection-resources#34-datasets)
- [awesome-TS-anomaly-detection](https://github.com/rob-med/awesome-TS-anomaly-detection#benchmark-datasets)
- [List of datasets for machine-learning research](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research#Anomaly_data)
- [paperswithcode.com](https://paperswithcode.com/dataset/skab)
- [Google datasets](https://datasetsearch.research.google.com/search?query=skoltech%20anomaly%20benchmark&docid=IIIE4VWbqUKszygyAAAAAA%3D%3D)
- [Industrial ML Datasets](https://github.com/nicolasj92/industrial-ml-datasets)
- etc.
