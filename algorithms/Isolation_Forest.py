from sklearn.ensemble import IsolationForest
import tensorflow as tf

class Isolation_Forest:
    """
    Isolation Forest or iForest builds an ensemble of iTrees for a given data set, then anomalies are those instances which have short average path lengths on the iTrees.

    Parameters
    ----------
    params : list
        A list containing three parameters: random_state, n_jobs, and contamination.
        
    Attributes
    ----------
    random_state : int
        The random seed used for reproducibility.
    n_jobs : int
        The number of CPU cores to use for parallelism.
    contamination : float
        The expected proportion of anomalies in the dataset.
        
    Examples
    --------
    >>> from Isolation_Forest import Isolation_Forest
    >>> PARAMS = [random_state, n_jobs, contamination]
    >>> model = Isolation_Forest(PARAMS)
    >>> model.fit(X_train)
    >>> predictions = model.predict(test_data)
    """
    
    def __init__(self, params):
        self.params = params
        self.random_state = self.params[0]
        self.n_jobs = self.params[1]
        self.contamination = self.params[2]
        
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
        
        model = IsolationForest(random_state=self.random_state, 
                                n_jobs=self.n_jobs,
                                contamination=self.contamination)
        return model
    
    def fit(self, X):
        """
        Train the Isolation Forest model on the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data for training the model.
        """
        
        self.model = self._build_model()

        self.model.fit(X)
    
    def predict(self, data):
        """
        Generate predictions using the trained Isolation Forest model.

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
