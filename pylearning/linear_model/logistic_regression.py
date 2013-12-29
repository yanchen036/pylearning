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
        J = 0.0
        for i in range(0, self.n):
            J += self.y[i, 0] * math.log(hx[i, 0]) + (1.0 - self.y[i, 0] * math.log(1.0 - hx[i, 0]))
        J *= -1.0 / self.n
        return J

    def _calc_gradient(self):
        hx = self.X * self.theta.T
        new_theta = np.zeros((1, self.m + 1))
        for col in range(0, self.m + 1):
            new_theta[0, col] = self.theta[0, col] - self.alpha / self.n \
                                * np.sum((np.asarray(hx - self.y) * np.asarray(self.X[:, col])))
        return new_theta


    def fit(self):
        pass

    def predict(self, X):
        pass