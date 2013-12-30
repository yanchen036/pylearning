# -*- coding: utf-8 -*-
# yanchen036@gmail.com

import math
import numpy as np

from .base import LinearModel


class LogisticRegression(LinearModel):
    def __init__(self, X, y, alpha=0.01, num_iters=100):
        assert isinstance(X, np.matrix)
        assert isinstance(y, np.matrix)
        # n is number of samples, m is the dimension of feature
        (self.n, self.m) = X.shape
        self.X = np.append(np.ones((self.n, 1)), X, 1)
        self.theta = np.zeros((1, self.m + 1))
        self.y = y
        self.alpha = alpha
        self.num_iters = num_iters

    def _sigmoid(self, z):
        return 1.0 / (1 + math.exp(-z))

    def _cost(self):
        hx = self.X * self.theta.T
        for i in range(0, hx.shape[0]):
            hx[i, 0] = self._sigmoid(hx[i, 0])
        J = 0.0
        for i in range(0, self.n):
            J += self.y[i, 0] * math.log(hx[i, 0]) + (1.0 - self.y[i, 0] * math.log(1.0 - hx[i, 0]))
        J *= -1.0 / self.n
        return J

    def _calc_gradient(self):
        hx = self.X * self.theta.T
        for i in range(0, hx.shape[0]):
            hx[i, 0] = self._sigmoid(hx[i, 0])
        new_theta = np.zeros((1, self.m + 1))
        for col in range(0, self.m + 1):
            new_theta[0, col] = self.theta[0, col] - self.alpha / self.n \
                * np.sum((np.asarray(hx - self.y) * np.asarray(self.X[:, col])))
        return new_theta

    def fit(self):
        J_history = []
        for iter in range(0, self.num_iters):
            self.theta = self._calc_gradient()
            J_history.append(self._cost())
        return J_history

    def predict(self, X):
        assert isinstance(X, np.matrix)
        assert X.shape[1] == self.m
        X_plus = np.append(np.ones((X.shape[0], 1)), X, 1)
        return self._sigmoid(X_plus * self.theta.T)
