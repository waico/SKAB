from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input, Layer, RepeatVector
from tensorflow.keras.models import Model


class KLDivergenceLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_sigma = inputs
        kl_loss = -0.5 * K.mean(
            1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1
        )
        self.add_loss(kl_loss)
        return kl_loss  # Return KL loss value


class Sampling(Layer):
    def __init__(self, latent_dim, epsilon_std=1.0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std

    def call(self, inputs):
        z_mean, z_log_sigma = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(
            shape=(batch, dim), mean=0.0, stddev=self.epsilon_std
        )
        return z_mean + z_log_sigma * epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Same shape as z_mean and z_log_sigma


class LSTM_VAE:
    """
    A reconstruction LSTM variational autoencoder model to detect anomalies in timeseries data using reconstruction error as an anomaly score.

    Parameters
    ----------
    TenserFlow_backend : bool, optional
        Flag to specify whether to use TensorFlow backend (default is False).

    Attributes
    ----------
    None

    Examples
    -------
    >>> from LSTM_VAE import LSTM_VAE
    >>> model = LSTM_VAE()
    >>> model.fit(train_data)
    >>> predictions = model.predict(test_data)
    """

    def __init__(self, params):
        self.params = params

    def _build_model(self, input_dim, timesteps, intermediate_dim, latent_dim):
        self._Random(0)

        x = Input(
            shape=(
                timesteps,
                input_dim,
            )
        )

        h = LSTM(intermediate_dim)(x)

        self.z_mean = Dense(latent_dim)(h)
        self.z_log_sigma = Dense(latent_dim)(h)

        z = Sampling(latent_dim)([self.z_mean, self.z_log_sigma])

        h_decoded = RepeatVector(timesteps)(z)
        decoder_h = LSTM(intermediate_dim, return_sequences=True)(h_decoded)
        decoder_mean = LSTM(input_dim, return_sequences=True)(decoder_h)

        vae = Model(x, decoder_mean)

        _ = Model(x, self.z_mean)

        decoder_input = Input(shape=(latent_dim,))

        _h_decoded = RepeatVector(timesteps)(decoder_input)
        _h_decoded = LSTM(intermediate_dim, return_sequences=True)(_h_decoded)

        _x_decoded_mean = LSTM(input_dim, return_sequences=True)(_h_decoded)
        _ = Model(decoder_input, _x_decoded_mean)

        vae.compile(optimizer="rmsprop", loss=self.vae_loss)

        return vae

    def _Random(self, seed_value):
        import os

        os.environ["PYTHONHASHSEED"] = str(seed_value)

        import random

        random.seed(seed_value)

        import numpy as np

        np.random.seed(seed_value)

        import tensorflow as tf

        tf.random.set_seed(seed_value)

    def vae_loss(self, x, x_decoded_mean):
        """
        Calculate the VAE loss.

        Parameters
        ----------
        x : tensorflow.Tensor
            Input data.
        x_decoded_mean : tensorflow.Tensor
            Decoded output data.

        Returns
        -------
        loss : tensorflow.Tensor
            VAE loss value.
        """
        mse = losses.MeanSquaredError()
        xent_loss = mse(x, x_decoded_mean)
        kl_loss = KLDivergenceLayer()([self.z_mean, self.z_log_sigma])
        loss = xent_loss + kl_loss
        return loss

    def fit(self, X):
        """
        Train the LSTM variational autoencoder model on the provided data.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for training.
        epochs : int, optional
            Number of training epochs (default is 20).
        validation_split : float, optional
            Fraction of the training data to be used as validation data (default is 0.1).
        BATCH_SIZE : int, optional
            Batch size for training (default is 1).
        early_stopping : bool, optional
            Whether to use early stopping during training (default is True).
        """

        self.shape = X.shape
        self.input_dim = self.shape[-1]
        self.timesteps = self.shape[1]
        self.latent_dim = 100
        self.epsilon_std = 1.0
        self.intermediate_dim = 32

        self.model = self._build_model(
            self.input_dim,
            timesteps=self.timesteps,
            intermediate_dim=self.intermediate_dim,
            latent_dim=self.latent_dim,
        )

        early_stopping = EarlyStopping(patience=5, verbose=0)

        self.model.fit(
            X,
            X,
            validation_split=self.params[2],
            epochs=self.params[0],
            batch_size=self.params[1],
            verbose=0,
            shuffle=False,
            callbacks=[early_stopping],
        )

    def predict(self, data):
        """
        Generate predictions using the trained LSTM variational autoencoder model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for making predictions.

        Returns
        -------
        predictions : numpy.ndarray
            The reconstructed output predictions.
        """

        return self.model.predict(data)
