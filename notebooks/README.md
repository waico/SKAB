# Anomaly Detection Algorithms

### Hotelling's T-squared statistic
Hotelling's statistic is one of the most popular statistical process control techniques. It is based on the Mahalanobis distance.
Generally, it measures the distance between the new vector of values and the previously defined vector of normal values additionally using variances.

Based on the paper:
```
Hotelling, Harold. "Multivariate quality control." Techniques of statistical analysis (1947).
```

### Hotelling's T-squared statistic + Q statistic (SPE index) based on PCA
Hotelling’s T-squared statistic measures variations in the principal component subspace.
Q statistic measures the projection of the sample vector on the residual subspace.
To avoid using two separated indicators (Hotelling's T-squared and Q statistics) for the process monitoring, we use a combined one based on PCA.

Based on the paper and references therein:
```
Joe Qin, S. "Statistical process monitoring: basics and beyond." Journal of Chemometrics: A Journal of the Chemometrics Society 17.8‐9 (2003): 480-502.
```

### Isolation Forest
Isolation Forest or iForest builds an ensemble of iTrees for a given data set, then anomalies are those instances which have short average path lengths on the iTrees.

Based on the paper:
```
Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." 2008 eighth ieee international conference on data mining. IEEE, 2008.
```

### LSTM-based NN
LSTM-based neural network for anomaly detection using reconstruction error as an anomaly score.

Based on the paper:
```
Filonov, Pavel, Andrey Lavrentyev, and Artem Vorontsov. "Multivariate industrial time series with cyber-attack simulation: Fault detection using an lstm-based predictive data model." arXiv preprint arXiv:1612.06676 (2016).
```

### Feed-Forward Autoencoder
Feed-forward neural network with autoencoder architecture for anomaly detection using reconstruction error as an anomaly score.

```
Chen, Jinghui, et al. "Outlier detection with autoencoder ensembles." Proceedings of the 2017 SIAM international conference on data mining. Society for Industrial and Applied Mathematics, 2017.
```

### MSCRED
MSCRED - Multi-Scale Convolutional Recurrent Encoder-Decoder first constructs multi-scale (resolution) signature matrices to characterize multiple levels of the system statuses across different time steps. 
In particular, different levels of the system statuses are used to indicate the severity of different abnormal incidents. 
Subsequently, given the signature matrices, a convolutional encoder is employed to encode the inter-sensor (time series) correlations patterns and an attention based Convolutional Long-Short Term Memory (ConvLSTM) network is developed to capture the temporal patterns. 
Finally, with the feature maps which encode the inter-sensor correlations and temporal information, a convolutional decoder is used to reconstruct the signature matrices and the residual signature matrices are further utilized to detect and diagnose anomalies. 
The intuition is that MSCRED may not reconstruct the signature matrices well if it never observes similar system statuses before.

Based on the paper:
```
Zhang, C., Song, D., Chen, Y., Feng, X., Lumezanu, C., Cheng, W., Ni, J., Zong, B., Chen, H., & Chawla, N. V. (2019). A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 1409-1416. https://doi.org/10.1609/aaai.v33i01.33011409
```
