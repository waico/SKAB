![skab](skab.png)

❗️❗️❗️**The current version of SKAB (v0.9) contains 34 datasets with collective anomalies. But the upcoming update to v1.0 (probably up to the summer of 2021) will contain 300+ additional files with point and collective anomalies. It will make SKAB one of the largest changepoint-containing benchmarks, especially in the technical field.**

# About SKAB [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/waico/SKAB/graphs/commit-activity) [![DOI](https://img.shields.io/badge/DOI-10.34740/kaggle/dsv/1693952-blue.svg)](https://doi.org/10.34740/KAGGLE/DSV/1693952) [![License: GPL v3.0](https://img.shields.io/badge/License-GPL%20v3.0-green.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
We propose the Skoltech Anomaly Benchmark (SKAB) designed for evaluating the anomaly detection algorithms. SKAB allows working with two main problems (there are two markups for anomalies):
* Outlier detection (anomalies considered and marked up as single-point anomalies)
* Changepoint detection (anomalies considered and marked up as collective anomalies)

SKAB consists of the following artifacts:
* Datasets.
* Leaderboard (scoreboard).
* Python modules for algorithms’ evaluation.
* Notebooks: python notebooks with anomaly detection algorithms.

The IIot testbed system is located in the Skolkovo Institute of Science and Technology (Skoltech).
All the details regarding the testbed and the experimenting process are presented in the following artifacts:
- Position paper (*currently submitted for publication*)
- [Slides about the project](https://drive.google.com/open?id=1dHUevwPp6ftQCEKnRgB4KMp9oLBMSiDM)

# Datasets
The SKAB v0.9 corpus contains 35 individual data files in .csv format. Each file represents a single experiment and contains a single anomaly. The dataset represents a multivariate time series collected from the sensors installed on the testbed. The [data](data/) folder contains datasets from the benchmark. The structure of the data folder is presented in [structure](./data/README.md) file. Columns in each data file are following:
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

# Leaderboard (Scoreboard)
Here we propose the leaderboard for SKAB v0.9 both for outlier and changepoint detection problems. You can also present and evaluate your algorithm using SKAB on [kaggle](https://www.kaggle.com/yuriykatser/skoltech-anomaly-benchmark-skab).
The results in the tables are calculated in the python notebooks from the [baselines](baselines/) folder.

## Outlier detection problem
*Sorted by FAR, both for FAR and MAR less is better*  
<<<<<<< HEAD
| Algorithm | FAR, % | MAR, % | F1
|---|---|---|---|
Perfect detector | 0 | 0 | 100 |
Null detector | 0 | 100 | 0 |
T-squared+Q (PCA) | ***5.09*** | 86.1| х |
Isolation forest | 6.86 | 72.09| х |
Autoencoder | 7.56 | 66.57| х |
T-squared | 12.14 | 52.56| х |
LSTM | 14.4 | ***40.44***| х |
=======
| Algorithm | FAR, % | MAR, % |
|---|---|---|
Perfect detector | 0 | 0
Null detector | 0 | 100
T-squared+Q (PCA) | ***5.09*** | 86.1
Isolation forest | 6.86 | 72.09
Autoencoder | 7.56 | 66.57
T-squared | 12.14 | 52.56
LSTM | 14.4 | 40.44
MSCRED | 25.17 | ***18.03***
>>>>>>> 5d281aa8d57d4ca89895acc0266780aae63e568d

## Changepoint detection problem
*Sorted by NAB (standart), for all metrics bigger is better*
| Algorithm | NAB (standart) | NAB (lowFP) | NAB (LowFN) |
|---|---|---|---|
Perfect detector | 100 | 100 | 100 
Isolation forest | ***37.53*** | ***17.09*** | ***45.02***
LSTM | 25.82 | 9.06 | 31.83
MSCRED | 18.67 | 14.92 | 20.14
T-squared | 17.87 | 3.44 | 23.2
ArimaFD | 16.06 | 14.03 | 17.12
Autoencoder | 15.59 | 0.78 | 20.91
T-squared+Q (PCA) | 5.83 | 4.8 | 6.1
Null detector | 0 | 0 | 0

# Notebooks
The Notebooks folder contains python notebooks with the code for the proposed leaderboard results reproducing.

We have calculated the results for five quite common anomaly detection algorithms:
- Hotelling's T-squared statistics;
- Hotelling's T-squared statistics + Q statistics based on PCA;
- Isolation forest;
- LSTM-based NN;
- Feed-Forward Autoencoder.

Additionaly to the repository were added the results of the following algorithms:
- ArimaFD;
- MSCRED.

# Citation
Please cite our project in your publications if it helps your research.
```
Iurii D. Katser and Vyacheslav O. Kozitsin, “Skoltech Anomaly Benchmark (SKAB).” Kaggle, 2020, doi: 10.34740/KAGGLE/DSV/1693952.
```

# Notable mentions
SKAB is acknowledged by some ML resources.
<details>
  <summary>List of links</summary>
  
  - [List of datasets for machine-learning research](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research#Anomaly_data)
  - [awesome-TS-anomaly-detection](https://github.com/rob-med/awesome-TS-anomaly-detection)

</details>
