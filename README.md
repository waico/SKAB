# SkAB Skoltech Anomaly Benchmark
This repository contains the data provided by the IIot testbed system for evaluating Anomaly and Changepoint Detection algorithms.

## data
> **A:** The data folder containes datasets from the benchmark. The structure of the data folder is following:

    .
    ├── data                    # Data files and processing Jupyter Notebook
	│   ├── Load data.ipynb     # Jupyter Notebook to load all data
	│   ├── anomaly-free        # 
	│   │   ├── anomal-free.csv # Data obtained from the experiments with normal mode
	│   ├── valve1              # Data obtained from the experiments with closing the valve at the outlet of the flow from the pump.
	│   │   ├── 1.csv           # 
	│   │   ├── 2.csv           # 
	│   │   ├── 3.csv           # 
	│   │   ├── 4.csv           # 	
	│   ├── valve2              # Data obtained from the experiments with closing the valve at the flow inlet to the pump.
	│   │   ├── 1.csv           # 
	│   │   ├── 2.csv           # 
	│   │   ├── 3.csv           # 
	│   │   ├── 4.csv           # 
	│   │   ├── 5.csv           # 
	│   │   ├── 6.csv           # 
	│   │   ├── 7.csv           # 
	│   │   ├── 8.csv           # 
	│   │   ├── 9.csv           # 
	│   │   ├── 10.csv          # 
	│   │   ├── 11.csv          # 
	│   │   ├── 12.csv          # 
	│   │   ├── 12.csv          # 
	│   │   ├── 13.csv          # 
	│   │   ├── 14.csv          # 
	│   │   ├── 15.csv          # 
	│   │   ├── 16.csv          # 
	│   ├── other               # Data obtained from the other experiments
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
    ├── test                    # Automated tests (alternatively `spec` or `tests`)
    ├── utils                   # Tools and utilities
    │   ├── evaluating.py       # 
    │   ├── t2.py               # 
    ├── .gitignore
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