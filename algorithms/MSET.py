from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import linalg as spla

class MSET: 
    """
    MSET - multivariate state estimation technique is a non-parametric and statistical modeling method, which calculates the estimated values based on the weighted average of historical data. In terms of procedure, MSET is similar to some nonparametric regression methods, such as, auto-associative kernel regression.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Examples
    --------
    >>> from MSET import MSET
    >>> model = MSET()
    >>> model.fit(data)
    >>> prediction = model.predict(test_data)
    """
    
    def __init__(self):
        self._Random(0)
        
    def _build_model(self):
        self.SS = StandardScaler()
        
    def _Random(self, seed_value): 

        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        import random
        random.seed(seed_value)

        import numpy as np
        np.random.seed(seed_value)

        import tensorflow as tf
        tf.random.set_seed(seed_value)
        
    def calc_W(self, X_obs):
        """
        Calculate the weight matrix W.

        Parameters
        ----------
        X_obs : numpy.ndarray
            Observations for which to calculate the weight matrix.

        Returns
        -------
        numpy.ndarray
            Weight matrix W.
        """
        
        DxX_obs = self.otimes(self.D, X_obs)
        try:
            W = spla.lu_solve(self.LU_factors, DxX_obs)
        except:
            W = np.linalg.solve(self.DxD, DxX_obs)
    
        return W
    
    def otimes(self, X, Y):
        """
        Compute the outer product of two matrices X and Y.

        Parameters
        ----------
        X : numpy.ndarray
            First matrix.
        Y : numpy.ndarray
            Second matrix.

        Returns
        -------
        numpy.ndarray
            Outer product of X and Y.
        """

        m1,n = np.shape(X)
        m2,p = np.shape(Y)

        if m1!=m2:
            raise Exception('dimensionality mismatch between X and Y.')

        Z = np.zeros( (n,p) )

        if n != p:
            for i in range(n):
                for j in range(p):
                    Z[i,j] = self.kernel(X[:,i], Y[:,j])
        else:
            for i in range(n):     
                for j in range(i, p):
                    Z[i,j] = self.kernel(X[:,i], Y[:,j])
                    Z[j,i] = Z[i,j]

        return Z
    
    def kernel(self, x, y):
        """
        Compute the kernel function value.

        Parameters
        ----------
        x : numpy.ndarray
            First vector.
        y : numpy.ndarray
            Second vector.

        Returns
        -------
        float
            Kernel function s(x,y) = 1 - ||x-y||/(||x|| + ||y||) value.
        """

        if all(x==y):
            return 1.
        else:
            return 1. - np.linalg.norm(x-y)/(np.linalg.norm(x) + np.linalg.norm(y))
    
    def fit(self, df, train_start = None, train_stop = None): 
        """
        Train the MSET model on the provided data.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data for training the model.
        train_start : int, optional
            Index to start training, by default None.
        train_stop : int, optional
            Index to stop training, by default None.

        Returns
        -------
        None
        """

        self.model = self._build_model()
            
        self.D = df[train_start:train_stop].values.T.copy() 
        self.D = self.SS.fit_transform(self.D.T).T
        
        self.DxD = self.otimes(self.D, self.D)
        self.LU_factors = spla.lu_factor(self.DxD)

    def predict(self, data):
        """
        Generate predictions using the trained MSET model.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data for generating predictions.

        Returns
        -------
        pandas.DataFrame
            Predicted output data.
        """
        
        X_obs = data.values.T.copy() 
        X_obs = self.SS.transform(X_obs.T).T

        pred = np.zeros(X_obs.T.shape)
        
        for i in range(X_obs.shape[1]):
            pred[[i],:] = (self.D @ self.calc_W(X_obs[:,i].reshape([-1,1]))).T
            
        return pd.DataFrame(self.SS.inverse_transform(pred), index=data.index, columns=data.columns)