# Author: Iurii Katser

from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
import scipy.stats as SS
from math import sqrt
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame

class T2:
    '''
    Calculation of the Hotelling's 1-dimensional T-squared statistics or T-squared statistics+Q-statistics based on PCA for fault detection.
    
    Статистический критерий Хотеллинга показывает отклонение состояния оборудования в каждый момент времени записи анализируемых сигналов, сравнивая значения с эталонными, полученными предварительно. Каждое значение критерия характеризует отклонение состояния контролируемого оборудования от нормального.
    Зачастую статистический критерий Хотеллинга применяется совместно с методом главных компонент: для подпространства главных компонент (подпространства признаков с наибольшей дисперсией) вычисляется T2-критерий, а для подпространства оставшихся (подпространство разностей) применяется Q-критерий [1-3], значение которого равно евклидовой норме вектора в подпространстве разностей. Так как Q-критерий применяется к подпространству разностей, он позволяет обнаружить отклонение зависимостей между измеряемыми параметрами, неучтенное при получении главных компонент для тестовой выборки. Появление возмущений в Q-критерии говорит о нарушении зависимостей, что, в свою очередь, может говорить о возникновении неисправности. Так как в подпространстве главных компонент содержатся сигналы с наибольшей дисперсией, а в подпространстве оставшихся компонент в основном шум, — контрольные пределы для T2-критерия часто больше соответствующих пределов в подпространстве оставшихся компонент. По этой причине требуется гораздо более высокая амплитуда возмущений, вносимая неисправностью, чтобы обнаружить ее с помощью T2-критерия.
    T2-критерий и Q-критерий применяются совместно для лучшего качества обнаружения. Они позволяют обнаруживать развитие аномалии на раннем этапе, однако Q-критерий является чувствительным к изменению зависимостей между контролируемыми параметрами, а T2-критерий зависит от эталонной выборки, выбор которой сказывается на работе алгоритма. Отдельной задачей является выбор контрольных пределов.
    
    В текущей библиотеке реализованы два подхода: T2 для исходного пространства признаков и T2+Q на основе метода главных компонент.
    
    [1] - Q-statistic and T2-statistic PCA-based measures for damage assessment in structures / LE Mujica, J. Rodellar, A. Ferna ́ndez, A. Gu ̈emes // Structural Health Monitoring: An International Journal. — 2010. — nov. — Vol. 10, no. 5. — Pp. 539–553.
    [2] - Zhao Chunhui, Gao Furong. Online fault prognosis with relative deviation analysis and vector autoregressive modeling // Chemical Engineering Science. — 2015. — dec. — Vol. 138. — Pp. 531–543.
    [3] - Li Wei, Peng Minjun, Wang Qingzhong. False alarm reducing in PCA method for sensor fault detection in a nuclear power plant // Annals of Nuclear Energy. — 2018. — aug. — Vol. 118. — Pp. 131–139.
    
    Examples
    --------
    T2+Q based on PCA:
    
    from statistics import T2
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    t2 = T2()
    t2.fit(df.iloc[:20])
    t2.predict(df)
    
    
    T2 without PCA:
    
    df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    t2 = T2(using_PCA=False)
    t2.fit(df.iloc[:20])
    t2.predict(df)

    '''
    def __init__(self, scaling=False, using_PCA=True, explained_variance=0.85, p_value=0.999):
        self.explained_variance = explained_variance
        self.using_PCA = using_PCA
        self.p_value = p_value
        self.scaling = scaling
        
    #T2 and Q statistics calculations
    #-----------------------------------------------------------
    def _T2_calculation(self,X):      
        T2 = []
        for i in range(len(X)):
            T2.append(X[i] @ self.inv_cov @ X[i].T)
        return T2

    def _Q_calculation(self,X):
        Q = []
        for i in range(len(X)):
            Q.append(X[i] @ self.transform_rc @ X[i].T)
        return Q
    #-----------------------------------------------------------
    
    #CALCULATING UPPER CONTROL LIMITS
    #-----------------------------------------------------------
    def _T2_UCL(self, X):
        if self.using_PCA:
            m = self.n_components
        else:
            m = X.shape[1]
        n = len(X)
        C_alpha = np.linspace(0,15,10000)[SS.f.cdf(np.linspace(0,15,10000), m, n-m)<self.p_value][-1]
        #koef = m*(n-1)/(n-m)
        koef = m*(n-1)*(n+1)/(n*(n-m))
        self.T2_UCL = koef*C_alpha

    def _Q_UCL(self, X):
        w, v = LA.eig(np.cov(X.T))

        Sum = 0
        for i in range(self.n_components,len(w)):
            Sum += w[i]

        tetta = []
        for i in [1,2,3]:
            tetta.append(Sum**i)
        h0 = 1-2*tetta[0]*tetta[2]/(3*tetta[1]**2)
        C_alpha = np.linspace(0,15,10000)[SS.norm.cdf(np.linspace(0,15,10000))<self.p_value][-1]
        self.Q_UCL = tetta[0]*(1+(C_alpha*h0*sqrt(2*tetta[1])/tetta[0])+tetta[1]*h0*(h0-1)/tetta[0]**2)**(1/h0)
    #-----------------------------------------------------------
    
    def _PCA(self, X):
        self.pca = PCA(n_components=self.explained_variance).fit(X)
        self.n_components = self.pca.n_components_
        self._EV = self.pca.components_.T
        return self.pca.transform(X)
    
    #PLOTTING AND SAVING RESULTS
    #-----------------------------------------------------------
    def plot_T2(self, T2=None, window_size=200):
        if T2 is None:
            T2 = self.T2
        plt.figure(figsize=(12, 6))
        plt.plot(T2, label='$T^2$-statistics')
        #for i in self.final_list:
        #    plt.axvspan(i[0], i[1], facecolor='green', alpha=0.2, zorder=0, label='Обучающая выборка')
        plt.grid(True)
        plt.axhline(self.T2_UCL, zorder=10, color='r', label='UCL')
        #plt.vlines(X.index[2320000],0,3*max(min(T2),self.T2_UCL), zorder = 20,colors='g',linestyles='dotted',label='fault beginning')
        plt.ylim(0, 3*max(min(T2), self.T2_UCL))
        plt.xlim((T2.index.values[0], T2.index.values[-1]))
        plt.title('$T^2$-statistics chart')
        plt.xlabel('Time')
        plt.ylabel('$T^2$-statistics value')
        plt.legend(['$T^2$-statistics','UCL','Обучающая выборка'])
        plt.tight_layout()
        
    def plot_Q(self, Q=None, window_size=200):
        if Q is None:
            Q = self.Q
        plt.figure(figsize=(12, 6))
        plt.plot(Q, label='$Q$-statistics')
        #for i in self.final_list:
        #    plt.axvspan(i[0], i[1], facecolor='green', alpha=0.2, zorder=0, label='Обучающая выборка')
        plt.grid(True)
        plt.axhline(self.Q_UCL, zorder=10, color='r', label='UCL')
        #plt.vlines(X.index[2320000],0,3*max(min(Q),self.Q_UCL), zorder = 20,colors='g',linestyles='dotted',label='fault beginning')
        plt.ylim(0, 3*max(min(Q), self.Q_UCL))
        plt.xlim((Q.index.values[0], Q.index.values[-1]))
        plt.title('$Q$-statistics chart')
        plt.xlabel('Time')
        plt.ylabel('$Q$-statistics value')
        plt.legend(['$Q$-statistics','UCL','Обучающая выборка'])
        plt.tight_layout()
    
    def _save(self, name='', fmt='png'):
        pwd = os.getcwd()
        iPath = pwd+'/pictures/'
        if not os.path.exists(iPath):
            os.mkdir(iPath)
        os.chdir(iPath)
        plt.savefig('{}.{}'.format(name, fmt), fmt='png', dpi=150, bbox_inches='tight')
        os.chdir(pwd)
    #-----------------------------------------------------------
    
    def fit(self, X):
        '''
        Computation of the inversed covariance matrix, matrix of transformation to the residual space (in case of using_PCA=True) and standart scaler fitting (in case of using scaling=True).
        
        Parameters
        ----------
        X : pandas.DataFrame()
            Training set.
        
        Returns
        -------
        self : object.
        '''
        X=X.copy()
        if self.scaling:
            #fitting PCA and calculation of scaler, EV
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            X_ = self.scaler.transform(X)
        else:
            X_ = X.values
        
        self.n_components = X.shape[1]
        if self.using_PCA:
            X_pc = self._PCA(X_)
        if self.n_components == X.shape[1]:
            #preparing inv_cov for T2
            self.inv_cov = LA.inv(np.cov(X_.T))
            
            #calculating T2_ucl
            self._T2_UCL(X_)
            if self.using_PCA:
                print('Number of principal components is equal to dataset shape. Q-statistics is unavailable.')
        else:
            #preparing inv_cov for T2 (principal space)
            self.inv_cov = LA.inv(np.cov(X_pc.T))
        
            #preparing transform matrix for Q (to residual space)
            self.transform_rc = np.eye(len(self._EV))-np.dot(self._EV,self._EV.T)
        
            #calculating T2_ucl and Q_ucl
            self._T2_UCL(X_)
            self._Q_UCL(X_)
            
        # calculating train indices
        indices = X.index.tolist()
        diff = X.index.to_series().diff()
        list_of_ind = diff[diff>diff.mean()+3*diff.std()].index.tolist()
        
        #self.final_list = [(X.index[0], indices[indices.index(list_of_ind[0])-1])] + \
        #[(list_of_ind[i], indices[indices.index(list_of_ind[i+1])-1]) for i in range(len(list_of_ind)-2)] + \
        #[(indices[indices.index(list_of_ind[-1])], X.index[-1])]

    
    def predict(self, X, plot_fig=True, save_fig=False, fig_name=['T2','Q'], window_size=1):
        '''
        Computation of T2-statistics or T2-statistics+Q-statistics for the testing dataset.
        
        Parameters
        ----------
        X : pandas.DataFrame()
            Testing dataset.
            
        plot_fig : boolean, True by default
            If True there will be plotted T2-statistics or T2-statistics+Q-statistics chart.
            
        save_fig : boolean, False by default
            If True there will be saved T2 and Q charts as .png to the current folder.
        
        fig_name : list of one or two str, ['T2','Q'] by default
            Names of the saved figures.
            
        Returns
        -------
        self : object
            Plotting and saving T2 or T2+Q charts; to get numpy arrays with T2 or Q values call self.T2 or self.Q.
        '''
        X=X.copy()
        if self.scaling:
            X_ = self.scaler.transform(X)
        else:
            X_ = X.values
        
        if self.n_components != X.shape[1]:
            #calculating T2
            self.T2 = DataFrame(self._T2_calculation(self.pca.transform(X_)), index=X.index).rolling(window_size).median()
            
            #calculating Q
            self.Q = DataFrame(self._Q_calculation(X_), index=X.index).rolling(window_size).median()
            
            #plotting
            if plot_fig:
                self.plot_T2(self.T2)
                if save_fig:
                    self._save(name=fig_name[0])
                    
                self.plot_Q(self.Q)
                if save_fig:
                    self._save(name=fig_name[1])
                
        else:
            #calculating T2
            self.T2 = self._T2_calculation(X_)
            
            #plotting
            if plot_fig:
                self.plot_T2(self.T2)
                if save_fig:
                    self._save(name=fig_name[0])
                
    def fit_predict(self, X, plot_fig=True, save_fig=False, fig_name=['T2','Q']):
        '''
        Computation of the inversed covariance matrix, matrix of transformation to the residual space (in case of using_PCA=True) and standart scaler fitting (in case of using scaling=True).
        Computation of T2-statistics or T2-statistics+Q-statistics for the training dataset.
        
        Parameters
        ----------
        X : pandas.DataFrame()
            Training dataset.
            
        plot_fig : boolean, True by default
            If True there will be plotted T2-statistics or T2-statistics+Q-statistics chart.
            
        save_fig : boolean, False by default
            If True there will be saved T2 and Q charts as .png to the current folder.
        
        fig_name : list of one or two str, ['T2','Q'] by default
            Names of the saved figures.
            
        Returns
        -------
        self : object
            Plotting and saving T2 or T2+Q charts; to get numpy arrays with T2 or Q values call self.T2 or self.Q.
        '''
        #fit
        self.fit(X)
        
        #predict
        self.predict(X, plot_fig=True, save_fig=False, fig_name=['T2','Q'])
       