"""EZH: EWMA-z score with hysteresis.

A one-class, causal outlier detector for SKAB. It is fitted on the first 400
rows of a dataset (the anomaly-free part) and then predicts the remaining rows
one by one. There is no training loop and no random number.

Fitted per dataset (15 numbers): mean and standard deviation of the six stable
channels, one reference score, and the reference rolling standard deviation of
the two thermal channels.

Licensed GPL-3.0-or-later like the rest of the SKAB repository.
"""

from collections import deque

import numpy as np

ALPHA = 0.2  # EMA smoothing factor of the level signal
ON, OFF = 2.5, 2.0  # hysteresis thresholds, in units of the reference score
HOLD = 40  # rows kept flagged after the hysteresis switches off
WSTD = 30  # rolling window of the thermal variability feature
VRATIO = 5.0  # thermal variability ratio that is mapped onto the ON threshold
# Accelerometer1RMS, Accelerometer2RMS, Current, Pressure, Voltage,
# Volume Flow RateRMS
STABLE = (0, 1, 2, 3, 6, 7)
THERMAL = (4, 5)  # Temperature, Thermocouple
TRAIN = 400  # rows used for fitting (SKAB convention)
EPS = 1e-9


def _ema(x, alpha):
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def _roll_std(x, w):
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = x[max(0, i - w + 1) : i + 1].std(0)
    return out


def fit_params(Xtr):
    """Everything the detector stores. Xtr: first 400 rows, shape [400, 8]."""
    mu = Xtr.mean(0)
    E = _ema(Xtr, ALPHA)
    sd = E[TRAIN // 2 :].std(0) + EPS * (1 + np.abs(mu))
    s = (np.abs(E - mu) / sd)[:, list(STABLE)].max(1)
    ref = float(s[TRAIN // 2 :].max())
    S = _roll_std(Xtr, WSTD)
    vref = np.percentile(S[WSTD:], 95, axis=0) + EPS * (1 + np.abs(mu))
    return {"mu": mu, "sd": sd, "ref": ref, "vref": vref}


def predict_batch(X):
    """Boolean prediction for rows 400.. of X (shape [n_rows >= 400, 8]).

    The alarm state starts switched off at row 400.
    """
    p = fit_params(X[:TRAIN])
    E = _ema(X, ALPHA)
    s = (np.abs(E - p["mu"]) / p["sd"])[:, list(STABLE)].max(1) / p["ref"]
    v = (_roll_std(X, WSTD) / p["vref"])[:, list(THERMAL)].max(1)
    score = np.maximum(s, v / VRATIO * ON)
    cur, last, out = False, -(10**9), np.zeros(len(X) - TRAIN, dtype=bool)
    for k, i in enumerate(range(TRAIN, len(X))):
        if not cur and score[i] > ON:
            cur = True
        elif cur and score[i] < OFF:
            cur = False
        if cur:
            last = i
        out[k] = cur or (i - last <= HOLD)
    return out


class EZH:
    """fit(first 400 rows), then predict(later rows) -> boolean array.

    The detector is causal: ``update(x)`` consumes one row and returns the
    decision for it, ``predict`` calls it for every row.
    """

    def fit(self, Xtr):
        Xtr = np.asarray(Xtr, dtype=float)
        self.p = fit_params(Xtr)
        # replay the training rows to bring the EMA and the ring buffer to
        # their state at row 399
        self.ema = None
        self.buf = deque(maxlen=WSTD)
        for x in Xtr:
            self._advance(x)
        self.cur, self.since_last = False, 10**9
        return self

    def _advance(self, x):
        if self.ema is None:
            self.ema = x.copy()
        else:
            self.ema = ALPHA * x + (1 - ALPHA) * self.ema
        self.buf.append(x)

    def score(self):
        z = np.abs(self.ema - self.p["mu"]) / self.p["sd"]
        s = z[list(STABLE)].max() / self.p["ref"]
        rolling_std = np.std(np.asarray(self.buf), axis=0)
        v = (rolling_std / self.p["vref"])[list(THERMAL)].max()
        return max(s, v / VRATIO * ON)

    def update(self, x):
        self._advance(np.asarray(x, dtype=float))
        sc = self.score()
        if not self.cur and sc > ON:
            self.cur = True
        elif self.cur and sc < OFF:
            self.cur = False
        self.since_last = 0 if self.cur else self.since_last + 1
        return self.cur or self.since_last <= HOLD

    def predict(self, X_test):
        X = np.asarray(X_test, dtype=float)
        return np.array([self.update(x) for x in X], dtype=bool)

    def n_parameters(self):
        """Fitted numbers that influence the prediction (15)."""
        return 2 * len(STABLE) + 1 + len(THERMAL)
