# Author: Iurii Katser

import os
from math import sqrt

import numpy as np
import scipy.stats as SS
from matplotlib import pyplot as plt
from numpy import linalg as LA
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class T2:
    """Calculation of the Hotelling's 1-dimensional T-squared
    statistic or T-squared statistic+Q-statistic based on PCA for
    anomaly detection in multivariate data.

    Based on the following papers:
    [1] - Q-statistic and T2-statistic PCA-based measures for damage
    assessment in structures / LE Mujica, J. Rodellar, A. Ferna ́ndez,
    A. Gu ̈emes // Structural Health Monitoring: An International
    Journal. — 2010. — nov. — Vol. 10, no. 5. — Pp. 539–553.
    [2] - Zhao Chunhui, Gao Furong. Online fault prognosis with
    relative deviation analysis and vector autoregressive modeling //
    Chemical Engineering Science. — 2015. — dec. — Vol. 138. — Pp.
    531–543.
    [3] - Li Wei, Peng Minjun, Wang Qingzhong. False alarm reducing in
    PCA method for sensor fault detection in a nuclear power plant //
    Annals of Nuclear Energy. — 2018. — aug. — Vol. 118. — Pp. 131–139.

    Parameters
    ----------
    scaling : boolean, default = False
        If True StandartScaler is used in the pipeline.
        If False no scaling procedures are used.

    using_pca : boolean, default = True
        If True T2+Q based on PCA is used as anomaly detection method.
        If False T2 without PCA is used as anomaly detection method.

    explained_variance : object, default = 0.85
        Proportion of the explained variance for principal components
        selection. Relevant only if using_pca=True.

    p_value : object, default = 0.999
        P value for upper control limits selection. Shows the proportion
        of the number of points in train set perceived as normal.

    Examples
    --------
    T2+Q based on PCA:

    from ControlCharts import T2
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    t2 = T2()
    t2.fit(df.iloc[:20])
    t2.predict(df)

    T2 without PCA:

    from ControlCharts import T2
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    t2 = T2(using_pca=False)
    t2.fit(df.iloc[:20])
    t2.predict(df)

    More examples at:
    https://github.com/YKatser/control-charts/tree/main/examples
    """

    def __init__(
        self,
        scaling=False,
        using_pca=True,
        explained_variance=0.85,
        p_value=0.999,
    ):
        self.scaling = scaling
        self.using_pca = using_pca
        self.explained_variance = explained_variance
        self.p_value = p_value

    # T2 and Q statistics calculations
    def _t2_calculation(self, x):
        t2 = []
        for i in range(len(x)):
            t2.append(x[i] @ self.inv_cov @ x[i].T)
        return t2

    def _q_calculation(self, x):
        q = []
        for i in range(len(x)):
            q.append(x[i] @ self.transform_rc @ x[i].T)
        return q

    # CALCULATING UPPER CONTROL LIMITS
    def _t2_ucl(self, x):
        if self.using_pca:
            m = self.n_components
        else:
            m = x.shape[1]

        n = len(x)
        linspace = np.linspace(0, 15, 10000)
        c_alpha = linspace[SS.f.cdf(linspace, m, n - m) < self.p_value][-1]
        # koef = m * (n-1) / (n-m)
        koef = m * (n - 1) * (n + 1) / (n * (n - m))

        self.t2_ucl = koef * c_alpha

    def _q_ucl(self, x):
        w, v = LA.eig(np.cov(x.T))
        sum_ = 0
        for i in range(self.n_components, len(w)):
            sum_ += w[i]

        tetta = []
        for i in [1, 2, 3]:
            tetta.append(sum_**i)
        h0 = 1 - 2 * tetta[0] * tetta[2] / (3 * tetta[1] ** 2)
        linspace = np.linspace(0, 15, 10000)
        c_alpha = linspace[SS.norm.cdf(linspace) < self.p_value][-1]

        self.q_ucl = tetta[0] * (
            1
            + (c_alpha * h0 * sqrt(2 * tetta[1]) / tetta[0])
            + tetta[1] * h0 * (h0 - 1) / tetta[0] ** 2
        ) ** (1 / h0)

    # applying pca
    def _pca_applying(self, x):
        self.pca = PCA(n_components=self.explained_variance).fit(x)
        self.n_components = self.pca.n_components_
        self._EV = self.pca.components_.T
        return self.pca.transform(x)

    # PLOTTING AND SAVING RESULTS
    def plot_t2(self, t2=None, t2_ucl=None, save_fig=False, fig_name="T2"):
        """Plotting results of T2-statistic calculation with matplotlib

        Parameters
        ----------
        t2 : pandas.DataFrame(), default = None
            Results of T2-statistic calculation.

        t2_ucl : float or int, default = None
            Upper control limit for T2.

        save_fig : boolean, default = False
            If True there will be saved T2 chart as .png to the
            current folder.

        fig_name : str, default = 'T2'
            Name of the saved figure.

        Returns
        -------
        self : object.
        """

        if t2 is None:
            t2 = self.t2
        if t2_ucl is None:
            t2_ucl = self.t2_ucl
        plt.figure(figsize=(12, 4))
        plt.plot(t2, label="$T^2$-statistic")
        # for i in self.final_list:
        #    plt.axvspan(i[0], i[1], facecolor='green', alpha=0.2, zorder=0,
        #                label='Train set')
        plt.grid(True)
        plt.axhline(t2_ucl, zorder=10, color="r", label="UCL")
        plt.ylim(0, 3 * max(t2.min().values, t2_ucl))
        plt.xlim(t2.index.values[0], t2.index.values[-1])
        plt.title("$T^2$-statistic chart")
        plt.xlabel("Time")
        plt.ylabel("$T^2$-statistic value")
        plt.legend(["$T^2$-statistic", "UCL", "Train set"])
        plt.tight_layout()
        if save_fig:
            self._save(name=fig_name)

    def plot_q(self, q=None, q_ucl=None, save_fig=False, fig_name="Q"):
        """Plotting results of Q-statistic calculation with matplotlib

        Parameters
        ----------
        q : pandas.DataFrame(), default = None
            Results of Q-statistic calculation.

        q_ucl : float or int, default = None
            Upper control limit for Q.

        save_fig : boolean, default = False
            If True there will be saved Q chart as .png to the
            current folder.

        fig_name : str, default = 'Q'
            Name of the saved figure.

        Returns
        -------
        self : object.
        """

        if q is None:
            q = self.q
        if q_ucl is None:
            q_ucl = self.q_ucl
        plt.figure(figsize=(12, 4))
        plt.plot(q, label="$Q$-statistic")
        # for i in self.final_list:
        #    plt.axvspan(i[0], i[1], facecolor='green', alpha=0.2, zorder=0,
        #                label='Train set')
        plt.grid(True)
        plt.axhline(q_ucl, zorder=10, color="r", label="UCL")
        plt.ylim(0, 3 * max(q.min().values, q_ucl))
        plt.xlim(q.index.values[0], q.index.values[-1])
        plt.title("$Q$-statistic chart")
        plt.xlabel("Time")
        plt.ylabel("$Q$-statistic value")
        plt.legend(["$Q$-statistic", "UCL", "Train set"])
        plt.tight_layout()
        if save_fig:
            self._save(name=fig_name)

    @staticmethod
    def _save(name="", fmt="png"):
        pwd = os.getcwd()
        iPath = pwd + "/pictures/"
        if not os.path.exists(iPath):
            os.mkdir(iPath)
        os.chdir(iPath)
        plt.savefig(f"{name}.{fmt}", fmt="png", dpi=150, bbox_inches="tight")
        os.chdir(pwd)

    def fit(self, x):
        """Computation of the inversed covariance matrix, matrix of
        transformation to the residual space (in case of
        using_pca=True) and standart scaler fitting (in case of using
        scaling=True).

        Parameters
        ----------
        x : pandas.DataFrame()
            Training set.

        Returns
        -------
        self : object.
        """

        x = x.copy()

        # removing constant columns
        initial_cols_number = len(x.columns)
        x = x.loc[:, (x != x.iloc[0]).any()]
        if initial_cols_number > len(x.columns):
            print("Constant columns removed")

        if self.scaling:
            # fitting PCA and calculation of scaler, EV
            self.scaler = StandardScaler()
            self.scaler.fit(x)
            x_ = self.scaler.transform(x)
        else:
            x_ = x.values

        if self.using_pca:
            x_pc = self._pca_applying(x_)
        else:
            self.n_components = x.shape[1]

        if self.n_components == x.shape[1]:
            # preparing inv_cov for T2
            self.inv_cov = LA.inv(np.cov(x_.T))

            # calculating T2_ucl
            self._t2_ucl(x_)
            if self.using_pca:
                print("""Number of principal components is equal to dataset \
                         shape. Q-statistics is unavailable.""")
        else:
            # preparing inv_cov for T2 (principal space)
            self.inv_cov = LA.inv(np.cov(x_pc.T))

            # preparing transform matrix for Q (to residual space)
            self.transform_rc = np.eye(len(self._EV)) - np.dot(
                self._EV, self._EV.T
            )

            # calculating t2_ucl and q_ucl
            self._t2_ucl(x_)
            self._q_ucl(x_)

        # calculating train indices
        # indices = x.index.tolist()
        # diff = x.index.to_series().diff()
        # list_of_ind = diff[diff > diff.mean() + 3 * diff.std()].index.tolist()

    def predict(
        self,
        x,
        plot_fig=True,
        save_fig=False,
        fig_name=["T2", "Q"],
        window_size=1,
    ):
        """Computation of T2-statistic or T2-statistic+Q-statistic for
        the test dataset.

        Parameters
        ----------
        x : pandas.DataFrame()
            Testing dataset.

        plot_fig : boolean, default = True
            If True there will be plotted T2-statistics or
            T2-statistics+Q-statistics chart.

        save_fig : boolean, default = False
            If True there will be saved T2 and Q charts as .png to the
            current folder.

        fig_name : list of one or two str, default = ['T2','Q']
            Names of the saved figures.

        window_size : int, default = 1
            Size of the window for median filter as a postprocessing.

        Returns
        -------
        self : object
            Plotting and saving T2 or T2+Q charts. To get DataFrames
            with T2 or Q values call self.t2 or self.q.
        """

        x = x.copy()
        if self.scaling:
            x_ = self.scaler.transform(x)
        else:
            x_ = x.values

        if self.n_components != x.shape[1]:
            # calculating T2
            self.t2 = (
                DataFrame(
                    self._t2_calculation(self.pca.transform(x_)),
                    index=x.index,
                    columns=["T2"],
                )
                .rolling(window_size)
                .median()
            )

            # calculating Q
            self.q = (
                DataFrame(
                    self._q_calculation(x_), index=x.index, columns=["Q"]
                )
                .rolling(window_size)
                .median()
            )

            # plotting
            if plot_fig:
                self.plot_t2(
                    t2=self.t2,
                    t2_ucl=self.t2_ucl,
                    save_fig=save_fig,
                    fig_name=fig_name[0],
                )
                self.plot_q(
                    q=self.q,
                    q_ucl=self.q_ucl,
                    save_fig=save_fig,
                    fig_name=fig_name[1],
                )

        else:
            # calculating T2
            self.t2 = (
                DataFrame(
                    self._t2_calculation(x_), index=x.index, columns=["T2"]
                )
                .rolling(window_size)
                .median()
            )

            # plotting
            if plot_fig:
                self.plot_t2(self.t2)
                if save_fig:
                    self._save(name=fig_name[0])
