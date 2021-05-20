# -*- coding: utf-8 -*-
"""
@author: kotsoev
"""

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, butter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from scipy import special
import scipy.stats as SS
from numpy import linalg as LA
from math import sqrt
import math

# import os
import warnings
warnings.filterwarnings('ignore')

def movmean(array, window):

    n = np.size(array)
    xx = array.copy()
    y = []
    for i in range(0, window):
        y.append(np.roll(xx.tolist() + [np.nan]*window, i))
    y = np.nanmean(y, axis=0)
    l = math.ceil(window/2)

    return y[l-1:n+l-1]


def movstd(array, window):

    n = np.size(array)
    xx = array.copy()
    y = []
    for i in range(0, window):
        y.append(np.roll(xx.tolist() + [np.nan]*window, i))
    y = np.nanstd(y, axis=0)
    l = math.ceil(window/2)

    return y[l-1:n+l-1]


def filloutliers(df, window, p=5):

    df = df.copy()
    for col in df.columns:

        tf = outlierdetect(df[col].values, window=window, threshold=p)
        df[col][tf] = np.nan

    df = df.interpolate(method='linear').fillna(method = 'bfill')
    return df


def outlierdetect(array, window, threshold):
    
    array = array.copy()
    center = movmean(array, window)
    amovstd = movstd(array, window)
    lowerbound = center - threshold*amovstd
    upperbound = center + threshold*amovstd

    tf = (array > upperbound) | (array < lowerbound)

    return tf 

def preproc(df):
    
    cols = df.columns[:-2]
    df_ = df.resample('1S').asfreq()
    df_[cols].interpolate(method='linear', inplace=True)
    df_[df.columns[-2:]].fillna(method = 'ffill', inplace=True)
    df_[cols] = filloutliers(df_[cols], 60, 3)
    
    for x in ['Accelerometer1RMS', 'Accelerometer2RMS']:
        b, a = butter(8, 0.125)
        df_[x] = filtfilt(b, a, df_[x].values, method="gust")
        
    for x in ['Current', 'Pressure', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS']:
        df_[x] = movmean(df_[x].values, 30)
        
    df_.drop(['Temperature'], axis=1, inplace=True)
    df_['Power'] = df_.Current*df_.Voltage
    return df_

def Q_calculation(X, transform_matrix):
    transform_rc = np.eye(len(transform_matrix)) - \
        np.dot(transform_matrix, transform_matrix.T)
    Q = []
    for i in range(len(X)):
        Q.append(X[i] @ transform_rc @ X[i].T)
    return Q


def Q_UCL(X, n_comp, p_value=.99999):
    w, v = LA.eig(np.cov(X.T))
    Sum = 0
    for i in range(n_comp, len(w)):
        Sum += w[i]

    tetta = []
    for i in [1, 2, 3]:
        tetta.append(Sum**i)
    h0 = 1-2*tetta[0]*tetta[2]/(3*tetta[1]**2)
    C_alpha = np.linspace(0, 15, 10000)[SS.norm.cdf(
        np.linspace(0, 15, 10000)) < p_value][-1]
    Q_UCL = tetta[0]*(1+(C_alpha*h0*sqrt(2*tetta[1])/tetta[0]) +
                      tetta[1]*h0*(h0-1)/tetta[0]**2)**(1/h0)

    return Q_UCL


def t2_my(Y, latent, pc_comp):
    """latent is explained varinance from PCA
    """

    standscore = Y[:, :pc_comp]*(1/np.sqrt(latent[:pc_comp])).T
    tsq = np.sum(standscore**2, axis=1)

    return tsq


def T2_UCL(X, p_value=.99999):
    m = X.shape[1]
    n = len(X)
    C_alpha = np.linspace(0, 15, 10000)[SS.f.cdf(
        np.linspace(0, 15, 10000), m, n-m) < p_value][-1]
    #koef = m*(n-1)/(n-m)
    koef = m*(n-1)*(n+1)/(n*(n-m))
    T2_UCL = koef*C_alpha

    return T2_UCL


class AnomalyDetection():
    
    def __init__(self):
        pass
    
    def fit_predict(self, df, fit_interval, exp_var=0.85):
        
        #df_ = preproc(df)
        data = df.copy()#.drop(['anomaly', 'changepoint'], axis=1)
        StSc = StandardScaler()
        StSc.fit(data[:fit_interval])
        data_norm = StSc.transform(data)
        
        pca = PCA(exp_var).fit(data_norm[:fit_interval])
        data_pca = pca.transform(data_norm)
        latent = pca.explained_variance_
        transform_matrix = pca.components_.T
        self.tt = t2_my(data_pca,
                        latent,
                        len(latent))
        #self.tt = movmean(self.tt, 20)
        self.tt_ucl = T2_UCL(data_norm[:fit_interval])
        self.q = Q_calculation(data_norm, transform_matrix)
        self.q = movmean(np.array(self.q), 20)
        self.q_ucl = Q_UCL(data_norm[:fit_interval], n_comp=len(latent))
        pred = np.logical_or(self.tt > self.tt_ucl,
                             self.q > self.q_ucl)
        pred = pd.Series(pred.astype(int),
                         index=df_.index)
        
        return pred
    
