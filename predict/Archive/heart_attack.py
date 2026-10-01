import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/home/w/AI/predict/Archive/Medicaldataset.csv")

age = df["Age"]
heart = df["Heart rate"]


# Index(['Age', 'Gender', 'Heart rate', 'Systolic blood pressure',
#        'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin',
#        'Result'],

# print(np.zeros((2,3)))
# [[0. 0. 0.]
#  [0. 0. 0.]]

class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.means_ = None
        self.vars_ = None
        self.priors_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c] 
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + self.var_smoothing
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self

    def _log_likelihood(self, X):
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        log_proba = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            diff = X - self.means_[i]
            log_prob_features = (
                -0.5 * np.log(2 * np.pi * self.vars_[i])
                - 0.5 * (diff**2) / self.vars_[i]
            )
            log_proba[:, i] = log_prob_features.sum(axis=1) + np.log(self.priors_[i])

        return log_proba

    def predict(self, X):
        log_proba = self._log_likelihood(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def predict_proba(self, X):
        log_proba = self._log_likelihood(X)
        log_proba -= log_proba.max(axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
