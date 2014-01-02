# -*- coding: utf-8 -*-
# yanchen036@gmail.com

import math
import numpy as np

from .base import LinearModel


class LogisticRegression(LinearModel):
    '''
    Parameters
    ----------
    X: array-like
        training set, shape = [n_samples, n_features]

    y: array-like
        lables, shape = [n_samples, 1]

    alpha: float
        gradient descent step length

    num_iters: int
        iteration numbers

    penalty: string, 'l1' or 'l2'
        specify the norm

    Lambda: float
        regularization strength
    '''
    def __init__(self, X, y, penalty='l2', Lambda=1.0, alpha=0.01, num_iters=100):
        assert isinstance(X, np.matrix)
        assert isinstance(y, np.matrix)
        # n is number of samples, m is the dimension of feature
        (self.n, self.m) = X.shape
        self.X = np.append(np.ones((self.n, 1)), X, 1)
        self.theta = np.zeros((1, self.m + 1))
        self.y = y
        self.penalty = penalty
        self.Lambda = Lambda
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
            #TODO avoid this assert, may need another rule to stop the iteration
            assert hx[i, 0] > 0
            assert 1.0 - hx[i, 0] > 0
            J += self.y[i, 0] * math.log(hx[i, 0]) + (1.0 - self.y[i, 0]) * math.log(1.0 - hx[i, 0])
        J *= -1.0 / self.n + self.Lambda / (2.0 * self.n) * (self.theta * self.theta.T)[0, 0]
        return J

    def _calc_gradient(self):
        hx = self.X * self.theta.T
        for i in range(0, hx.shape[0]):
            hx[i, 0] = self._sigmoid(hx[i, 0])
        new_theta = np.zeros((1, self.m + 1))
        new_theta[0, 0] = self.theta[0, 0] - self.alpha / self.n \
                * np.sum((np.asarray(hx - self.y) * np.asarray(self.X[:, 0])))
        for col in range(1, self.m + 1):
            new_theta[0, col] = self.theta[0, col] - self.alpha \
                * (
                    1.0 / self.n * np.sum((np.asarray(hx - self.y) * np.asarray(self.X[:, col])))
                    -
                    self.Lambda / self.n * self.theta[0, col]
                )
        return new_theta

    def fit(self):
        J_history = []
        for iter in range(0, self.num_iters):
            print 'iter: %d' % iter
            self.theta = self._calc_gradient()
            J_history.append(self._cost())
        return J_history

    def predict(self, X):
        assert isinstance(X, np.matrix)
        assert X.shape[1] == self.m
        X_plus = np.append(np.ones((X.shape[0], 1)), X, 1)
        return self._sigmoid(X_plus * self.theta.T)
