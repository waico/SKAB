# SkAB Skoltech Anomaly Benchmark
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
Here we propose an initial leaderboard for SkAB v1.0 both for outlier and changepoint detection problems.

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

## data
> **A:** The data folder containes datasets from the benchmark. The structure of the data folder is following:

    .
    ├── data                        # Data files and processing Jupyter Notebook
	│   ├── Load data.ipynb         # Jupyter Notebook to load all data
	│   ├── anomaly-free         
	│   │   ├── anomal-free.csv     # Data obtained from the experiments with normal mode
	│   ├── valve1                  # Data obtained from the experiments with closing the valve at the outlet of the flow from the pump.
	│   │   ├── 1.csv            
	│   │   ├── 2.csv            
	│   │   ├── 3.csv            
	│   │   ├── 4.csv            	
	│   ├── valve2                  # Data obtained from the experiments with closing the valve at the flow inlet to the pump.
	│   │   ├── 1.csv            
	│   │   ├── 2.csv            
	│   │   ├── 3.csv            
	│   │   ├── 4.csv            
	│   │   ├── 5.csv            
	│   │   ├── 6.csv            
	│   │   ├── 7.csv            
	│   │   ├── 8.csv            
	│   │   ├── 9.csv            
	│   │   ├── 10.csv           
	│   │   ├── 11.csv           
	│   │   ├── 12.csv           
	│   │   ├── 12.csv           
	│   │   ├── 13.csv           
	│   │   ├── 14.csv           
	│   │   ├── 15.csv           
	│   │   ├── 16.csv           
	│   ├── other                   # Data obtained from the other experiments
	│   │   ├── 13.csv              # Sharply behavior of rotor imbalance
	│   │   ├── 14.csv              # Linear behavior of rotor imbalance
	│   │   ├── 15.csv              # Step behavior of rotor imbalance
	│   │   ├── 16.csv              # Dirac delta function behavior of rotor imbalance
	│   │   ├── 17.csv              # Exponential behavior of rotor imbalance
	│   │   ├── 18.csv              # The slow increase in the amount of water in the circuit
	│   │   ├── 19.csv              # The sudden increase in the amount of water in the circuit
	│   │   ├── 20.csv              # Draining water from the tank until cavitation
	│   │   ├── 21.csv              # Two-phase flow supply to the pump inlet (cavitation)
	│   │   ├── 22.csv              # Water supply of increased temperature
    ├── baselines                   # Testing algorithms using the benchmark
	│   ├── hotelling.ipynb         # Testing by using Hotelling's T-squared statistics statistic
	│   ├── hotelling and q.ipynb   # Testing by using Hotelling and Q statistic
	│   ├── isolation_forest.ipynb  # Testing by using Isolation Forest algorithm
	│   ├── autoencoder.ipynb       # Testing by using Autoencoder architecture
	│   ├── autoencoder.h5          # Obtained weights of the specific Autoencoder model
	│   ├── lstm.ipynb              # Testing by using Long Short Term Memory architecture 	
	│   ├── lstm.h5                 # Obtained weights of the specific LSTM model 	
    ├── utils                       # Tools and utilities
    │   ├── evaluating.py           # Implemenation of FAR, MAR, ADD, NAB, evaluation metrics 
    │   ├── t2.py                   # Implemenation of Hotelling and Q statistics
    ├── .gitignore
    ├── LICENSE
    └── README.md

