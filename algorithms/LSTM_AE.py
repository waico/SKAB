from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

class LSTM_AE:
    """
    A reconstruction sequence-to-sequence (LSTM-based) autoencoder model to detect anomalies in timeseries data using reconstruction error as an anomaly score.

    Parameters
    ----------
    params : list
        A list of hyperparameters for the model, containing the following elements:
        - EPOCHS : int
            The number of training epochs.
        - BATCH_SIZE : int
            The batch size for training.
        - VAL_SPLIT : float
            The validation split ratio during training.

    Attributes
    ----------
    params : list
        The hyperparameters for the model.

    Examples
    --------
    >>> from LSTM_AE import LSTM_AE
    >>> PARAMS = [EPOCHS, BATCH_SIZE, VAL_SPLIT]
    >>> model = LSTM_AE(PARAMS)
    >>> model.fit(train_data)
    >>> predictions = model.predict(test_data)
    """
    
    def __init__(self, params):
        self.params = params
        
    def _Random(self, seed_value):

        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        import random
        random.seed(seed_value)

        import numpy as np
        np.random.seed(seed_value)

        import tensorflow as tf
        tf.random.set_seed(seed_value)
        
    def _build_model(self):
        self._Random(0)
        
        inputs = Input(shape=(self.shape[1], self.shape[2]))
        encoded = LSTM(100, activation='relu')(inputs)

        decoded = RepeatVector(self.shape[1])(encoded)
        decoded = LSTM(100, activation='relu', return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(self.shape[2]))(decoded)

        model = Model(inputs, decoded)
        encoder = Model(inputs, encoded)

        model.compile(optimizer='adam', loss='mae', metrics=["mse"])
        
        return model
    
    def fit(self, X):
        """
        Train the sequence-to-sequence (LSTM-based) autoencoder model on the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data for training the model.
        """

        self.shape = X.shape
        self.model = self._build_model()

        early_stopping = EarlyStopping(patience=5, 
                                       verbose=0)

        self.model.fit(X, X,
                  validation_split=self.params[2],
                  epochs=self.params[0],
                  batch_size=self.params[1],
                  verbose=0,
                  shuffle=False,
                  callbacks=[early_stopping]
                  )
    
    def predict(self, data):
        """
        Generate predictions using the trained sequence-to-sequence (LSTM-based) autoencoder model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for generating predictions.

        Returns
        -------
        numpy.ndarray
            Predicted output data.
        """
        
        return self.model.predict(data)