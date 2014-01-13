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
        labels, shape = [n_samples, 1]

    penalty: string, 'l1' or 'l2'
        specify the norm

    Lambda: float
        regularization strength

    grad_descent: boolean
        if true, use gradient descent, which is simple but much slower than other advanced optimization method,
        such as BFGS for l2 and OWQN for l1
        this option is just for efficiency compare

    alpha: float, only for gradient descent use
        gradient descent step length

    max_iterations: int, only for gradient descent use
        max iteration numbers

    stop_diff: float,
        when the last two costs' diff less than stop_diff, iterate stop
    '''
    def __init__(self, X, y, penalty='l2', Lambda=1.0, grad_descent=False, alpha=0.01, max_iterations=200, stop_diff=1e-6):
        assert isinstance(X, np.matrix)
        assert isinstance(y, np.matrix)
        # n is number of samples, m is the dimension of features
        (self.n, self.m) = X.shape
        self.X = np.append(np.ones((self.n, 1)), X, 1)
        self.theta = np.zeros((1, self.m + 1))
        self.y = y
        self.penalty = penalty
        self.Lambda = Lambda
        self.grad_descent = grad_descent
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.stop_diff = stop_diff

    def _sigmoid(self, z):
        # avoid large number
        if (z <= -20.0):
            z = -20.0
        if (z >= 30.0):
            z = 30.0
        return 1.0 / (1 + math.exp(-z))

    def _cost(self):
        hx = self.X * self.theta.T
        for i in range(0, hx.shape[0]):
            hx[i, 0] = self._sigmoid(hx[i, 0])
        J = 0.0
        for i in range(0, self.n):
            if (hx[i, 0] <= 0.0):
                hx[i, 0] = 1e-6
            elif (hx[i, 0] >= 1.0):
                hx[i, 0] = 0.999999
            assert hx[i, 0] > 0
            assert 1.0 - hx[i, 0] > 0

            J += self.y[i, 0] * math.log(hx[i, 0]) + (1.0 - self.y[i, 0]) * math.log(1.0 - hx[i, 0])
        J = -1.0 / self.n * J + self.Lambda / (2.0 * self.n) * (self.theta * self.theta.T)[0, 0]
        return J

    def _calc_gradient(self):
        hx = self.X * self.theta.T
        for i in range(0, hx.shape[0]):
            hx[i, 0] = self._sigmoid(hx[i, 0])
        grad = np.zeros((1, self.m + 1))
        grad[0, 0] = 1.0 / self.n * np.sum((np.asarray(hx - self.y) * np.asarray(self.X[:, 0])))
        for col in range(1, self.m + 1):
            grad[0, col] = 1.0 / self.n * np.sum((np.asarray(hx - self.y) * np.asarray(self.X[:, col]))) \
                           + self.Lambda / self.n * self.theta[0, col]
        return grad

    def fit(self):
        J_history = []
        for iter in range(0, self.max_iterations):
            if (iter >= self.max_iterations):
                break
            self.theta -= self.alpha * self._calc_gradient()
            J_history.append(self._cost())
            if (iter > 0 and J_history[-1] - J_history[-2] <= self.stop_diff):
                break
        return J_history

    def predict(self, X):
        assert isinstance(X, np.matrix)
        assert X.shape[1] == self.m
        X_plus = np.append(np.ones((X.shape[0], 1)), X, 1)
        return self._sigmoid(X_plus * self.theta.T)
