__author__ = 'yanchen036@gmail.com'

import numpy as np

from linear_model.base import LinearModel


class LinearRegressionModel(LinearModel):
    def __init__(self, X, y, alpha=0.01, num_iters=100):
        assert isinstance(X, np.matrix)
        assert isinstance(y, np.matrix)
        # n is number of samples, m is the dimension of feature
        (self.n, self.m) = X.shape
        self.X = np.append(X, np.ones((self.n, 1)))
        self.theta = np.zeros((1, self.m + 1))
        self.y = y
        self.alpha = alpha
        self.num_iters = num_iters

    def cost(self):
        hx = self.X * self.theta.T
        least_square = (hx - self.y).T * (hx - self.y)
        return least_square[0, 0] / (2 * self.n)

    def gradient_descent(self):
        J_history = []
        for iter in range(0, self.num_iters):
            hx = self.X * self.theta.T - self.y
            new_theta = np.zeros((1, self.m + 1))
            for col in range(0, self.m + 1):
                new_theta[0, col] = self.theta[0, col] - self.alpha / self.m * \
                    np.sum((np.asarray(hx - self.y) * np.asarray(self.X[:, col])))
            self.theta = new_theta
            J_history.append(self.cost())

        return [self.theta, J_history]