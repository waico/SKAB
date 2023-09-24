from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

class Vanilla_LSTM:
    """
    LSTM-based neural network for anomaly detection using reconstruction error as an anomaly score.

    Parameters
    ----------
    params : list
        A list containing various parameters for configuring the LSTM model.

    Attributes
    ----------
    model : Sequential
        The trained LSTM model.

    Examples
    --------
    >>> from Vanilla_LSTM import Vanilla_LSTM
    >>> PARAMS = [N_STEPS, EPOCHS, BATCH_SIZE, VAL_SPLIT]
    >>> lstm_model = Vanilla_LSTM(PARAMS)
    >>> lstm_model.fit(train_data, train_labels)
    >>> predictions = lstm_model.predict(test_data)
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
        
        model = Sequential()
        model.add(LSTM(100, 
                       activation='relu', 
                       return_sequences=True, 
                       input_shape=(self.params[0], self.n_features)))
        model.add(LSTM(100, 
                       activation='relu'))
        model.add(Dense(self.n_features))
        model.compile(optimizer='adam', 
                      loss='mae', 
                      metrics=["mse"])
        return model
    
    def fit(self, X, y):
        """
        Train the LSTM model on the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data for training the model.
        y : numpy.ndarray
            Target data for training the model.
        """
        self.n_features = X.shape[2]
        self.model = self._build_model()

        early_stopping = EarlyStopping(patience=10, 
                                       verbose=0)

        reduce_lr = ReduceLROnPlateau(factor=0.1, 
                                      patience=5, 
                                      min_lr=0.0001, 
                                      verbose=0)

        self.model.fit(X, y,
                  validation_split=self.params[3],
                  epochs=self.params[1],
                  batch_size=self.params[2],
                  verbose=0,
                  shuffle=False,
                  callbacks=[early_stopping, reduce_lr]
                  )
    
    def predict(self, data):
        """
        Generate predictions using the trained LSTM model.

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
