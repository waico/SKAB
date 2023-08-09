# Anomaly Detection Algorithms

### Hotelling's T-squared statistic 
Hotelling's statistic is one of the most popular statistical process control techniques. It is based on the Mahalanobis distance.
Generally, it measures the distance between the new vector of values and the previously defined vector of normal values additionally using variances.

[[notebook]](https://github.com/YKatser/ControlCharts/blob/main/examples/t2_SKAB.ipynb) [[paper]](https://www.semanticscholar.org/paper/Multivariate-Quality-Control-illustrated-by-the-air-Hotelling/529ba6c1a80b684d2f704a7565da305bb84f14e8)

### Hotelling's T-squared statistic + Q statistic (SPE index) based on PCA
The combined index is based on PCA.
Hotelling’s T-squared statistic measures variations in the principal component subspace.
Q statistic measures the projection of the sample vector on the residual subspace.
To avoid using two separated indicators (Hotelling's T-squared and Q statistics) for the process monitoring, we use a combined one based on logical or.

[[notebook]](https://github.com/YKatser/ControlCharts/blob/main/examples/t2_with_q_SKAB.ipynb) [[paper]](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.800)

### Isolation Forest
Isolation Forest or iForest builds an ensemble of iTrees for a given data set, then anomalies are those instances which have short average path lengths on the iTrees.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/isolation_forest.ipynb) [[paper]](https://ieeexplore.ieee.org/abstract/document/4781136?casa_token=kiHmrqDyGL4AAAAA:O4yM7O2WCXdQH2sQbpKUXAHiepBxUhc5odzbydmgTiz5f7ZEDYgkXltodCahlgIzArxUldce5LB9mg)

### LSTM-based NN (LSTM)
LSTM-based neural network for anomaly detection using reconstruction error as an anomaly score.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/LSTM.ipynb) [[paper]](https://arxiv.org/abs/1612.06676)

### Feed-Forward Autoencoder
Feed-forward neural network with autoencoder architecture for anomaly detection using reconstruction error as an anomaly score.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/autoencoder.ipynb) [[paper]](https://epubs.siam.org/doi/abs/10.1137/1.9781611974973.11)

### Convolutional Autoencoder (Conv-AE)
A reconstruction convolutional autoencoder model to detect anomalies in timeseries data using reconstruction error as an anomaly score.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/CAE.ipynb) [[paper]](https://keras.io/examples/timeseries/timeseries_anomaly_detection/)

### LSTM Autoencoder (LSTM-AE)
If you inputs are sequences, rather than vectors or 2D images, then you may want to use as encoder and decoder a type of model that can capture temporal structure, such as a LSTM. To build a LSTM-based autoencoder, first use a LSTM encoder to turn your input sequences into a single vector that contains information about the entire sequence, then repeat this vector n times (where n is the number of timesteps in the output sequence), and run a LSTM decoder to turn this constant sequence into the target sequence.

A reconstruction sequence-to-sequence (LSTM-based) autoencoder model to detect anomalies in timeseries data using reconstruction error as an anomaly score.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/LSTM-AE.ipynb) [[paper]](https://machinelearningmastery.com/lstm-autoencoders/) [[paper]](https://blog.keras.io/building-autoencoders-in-keras.html)

### LSTM Variational Autoencoder (LSTM-VAE)
A reconstruction LSTM variational autoencoder model to detect anomalies in timeseries data using reconstruction error as an anomaly score.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/LSTM-VAE.ipynb) [[paper]](https://arxiv.org/pdf/1511.06349.pdf) [[code]](https://github.com/twairball/keras_lstm_vae)

### Variational Autoencoder (VAE)
A reconstruction variational autoencoder (VAE) model to detect anomalies in timeseries data using reconstruction error as an anomaly score. VAE is an autoencoder that learns a latent variable model for its input data. So instead of letting your neural network learn an arbitrary function, you are learning the parameters of a probability distribution modeling your data. If you sample points from this distribution, you can generate new input data samples: a VAE is a "generative model".

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/VAE.ipynb) [[paper1]](https://arxiv.org/pdf/1312.6114.pdf) [[paper2]](https://dl.acm.org/doi/pdf/10.1145/3178876.3185996?casa_token=HVY_9X3NxToAAAAA%3AZzZNSpmDdI9bEbTCqC1R3fPLiP4SDHyH9l9VyHxZ9zsL_3UXblc7Fe-ZdMPI7gkyVN9orRYQ5j9C) [[code]](https://blog.keras.io/building-autoencoders-in-keras.html)


### MSCRED
MSCRED - Multi-Scale Convolutional Recurrent Encoder-Decoder first constructs multi-scale (resolution) signature matrices to characterize multiple levels of the system statuses across different time steps. 
In particular, different levels of the system statuses are used to indicate the severity of different abnormal incidents. 
Subsequently, given the signature matrices, a convolutional encoder is employed to encode the inter-sensor (time series) correlations patterns and an attention based Convolutional Long-Short Term Memory (ConvLSTM) network is developed to capture the temporal patterns. 
Finally, with the feature maps which encode the inter-sensor correlations and temporal information, a convolutional decoder is used to reconstruct the signature matrices and the residual signature matrices are further utilized to detect and diagnose anomalies. 
The intuition is that MSCRED may not reconstruct the signature matrices well if it never observes similar system statuses before.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/MSCRED.ipynb) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/3942)

### MSET 
MSET - multivariate state estimation technique is a non-parametric and statistical modeling method, which calculates the estimated values based on the weighted average of historical data. In terms of procedure, MSET is similar to some nonparametric regression methods, such as, auto-associative kernel regression.

[[notebook]](https://github.com/waico/SKAB/blob/master/notebooks/MSET.ipynb) [[paper]](https://inis.iaea.org/collection/NCLCollectionStore/_Public/32/025/32025817.pdf)
