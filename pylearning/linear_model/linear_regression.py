__author__ = 'yanchen036@gmail.com'

import numpy as np

from .base import LinearModel


class LinearRegressionModel(LinearModel):
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

    def _cost(self):
        hx = self.X * self.theta.T
        least_square = (hx - self.y).T * (hx - self.y)
        return least_square[0, 0] / (2 * self.n)

    def _calc_gradient(self):
        hx = self.X * self.theta.T
        new_theta = np.zeros((1, self.m + 1))
        for col in range(0, self.m + 1):
            new_theta[0, col] = self.theta[0, col] - self.alpha / self.n \
                * np.sum((np.asarray(hx - self.y) * np.asarray(self.X[:, col])))
        return new_theta

    def gradient_descent(self):
        J_history = []
        for iter in range(0, self.num_iters):
            self.theta = self.calc_gradient()
            J_history.append(self.cost())

        return [self.theta, J_history]