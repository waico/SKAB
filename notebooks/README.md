# Anomaly Detection Algorithms

### Hotelling's T-squared statistic 
Hotelling's statistic is one of the most popular statistical process control techniques. It is based on the Mahalanobis distance.
Generally, it measures the distance between the new vector of values and the previously defined vector of normal values additionally using variances.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/hotelling.ipynb) [[paper]](https://www.semanticscholar.org/paper/Multivariate-Quality-Control-illustrated-by-the-air-Hotelling/529ba6c1a80b684d2f704a7565da305bb84f14e8)

### Hotelling's T-squared statistic + Q statistic (SPE index) based on PCA
The combined index is based on PCA.
Hotelling’s T-squared statistic measures variations in the principal component subspace.
Q statistic measures the projection of the sample vector on the residual subspace.
To avoid using two separated indicators (Hotelling's T-squared and Q statistics) for the process monitoring, we use a combined one based on logical or.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/hotelling_q.ipynb) [[paper]](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.800)

### Isolation Forest
Isolation Forest or iForest builds an ensemble of iTrees for a given data set, then anomalies are those instances which have short average path lengths on the iTrees.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/isolation_forest.ipynb) [[paper]](https://ieeexplore.ieee.org/abstract/document/4781136?casa_token=kiHmrqDyGL4AAAAA:O4yM7O2WCXdQH2sQbpKUXAHiepBxUhc5odzbydmgTiz5f7ZEDYgkXltodCahlgIzArxUldce5LB9mg)

### LSTM-based NN
LSTM-based neural network for anomaly detection using reconstruction error as an anomaly score.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/lstm.ipynb) [[paper]](https://arxiv.org/abs/1612.06676)

### Feed-Forward Autoencoder
Feed-forward neural network with autoencoder architecture for anomaly detection using reconstruction error as an anomaly score.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/autoencoder.ipynb) [[paper]](https://epubs.siam.org/doi/abs/10.1137/1.9781611974973.11)


### MSCRED
MSCRED - Multi-Scale Convolutional Recurrent Encoder-Decoder first constructs multi-scale (resolution) signature matrices to characterize multiple levels of the system statuses across different time steps. 
In particular, different levels of the system statuses are used to indicate the severity of different abnormal incidents. 
Subsequently, given the signature matrices, a convolutional encoder is employed to encode the inter-sensor (time series) correlations patterns and an attention based Convolutional Long-Short Term Memory (ConvLSTM) network is developed to capture the temporal patterns. 
Finally, with the feature maps which encode the inter-sensor correlations and temporal information, a convolutional decoder is used to reconstruct the signature matrices and the residual signature matrices are further utilized to detect and diagnose anomalies. 
The intuition is that MSCRED may not reconstruct the signature matrices well if it never observes similar system statuses before.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/mscred.ipynb) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/3942)
