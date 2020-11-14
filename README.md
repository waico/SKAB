# SkAB Skoltech Anomaly Benchmark
This repository contains the data provided by the IIot testbed system for evaluating Anomaly and Changepoint Detection algorithms.

## data
The data folder containes datasets from the benchmark. The structure of the data folder is following:
- **anomaly-free** contains a normal or anomaly-free mode for algorithms tunning.
- **valve1**
- **valve2**
- **other**


> **A:** Because you don't want to test the code, you want to test the *program*.


    .
    ├── data                    # Data files
	│   ├── other               # Other data files
	│   │   ├── 13.csv          # Sharply behavior of rotor imbalance
	│   │   ├── 14.csv          # Linear behavior of rotor imbalance
	│   │   ├── 15.csv          # Step behavior of rotor imbalance
	│   │   ├── 16.csv          # Dirac delta function behavior of rotor imbalance
	│   │   ├── 17.csv          # Exponential behavior of rotor imbalance
	│   │   ├── 18.csv          # The slow increase in the amount of water in the circuit
	│   │   ├── 19.csv          # The sudden increase in the amount of water in the circuit
	│   │   ├── 20.csv          # Draining water from the tank until cavitation
	│   │   ├── 21.csv          # Two-phase flow supply to the pump inlet (cavitation)
	│   │   ├── 22.csv          # Water supply of increased temperature
    ├── docs                    # Documentation files (alternatively `doc`)
    ├── src                     # Source files (alternatively `lib` or `app`)
    ├── test                    # Automated tests (alternatively `spec` or `tests`)
    ├── tools                   # Tools and utilities
    │   ├── TOC.md              # Table of contents
    │   ├── faq.md              # Frequently asked questions
    │   ├── misc.md             # Miscellaneous information
    ├── LICENSE
    └── README.md
	
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