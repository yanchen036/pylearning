# -*- coding: utf-8 -*-
# yanchen036@gmail.com

import math
import numpy as np
import scipy.optimize
from ..util import owlqn
from .base import LinearModel


'''bfgs will call this function'''
def _obj_func(x0, *args):
    assert isinstance(x0, np.ndarray)
    theta = np.asmatrix(x0)
    lr = args[0]
    assert isinstance(lr, LogisticRegression)
    norm = args[1]
    hx = lr.X * theta.T
    for i in range(0, hx.shape[0]):
        hx[i, 0] = lr._sigmoid(hx[i, 0])
    J = 0.0
    for i in range(0, lr.n):
        if (hx[i, 0] <= 0.0):
            hx[i, 0] = 1e-9
        elif (hx[i, 0] >= 1.0):
            hx[i, 0] = 0.999999
        assert hx[i, 0] > 0
        assert 1.0 - hx[i, 0] > 0

        J += lr.y[i, 0] * math.log(hx[i, 0]) + (1.0 - lr.y[i, 0]) * math.log(1.0 - hx[i, 0])
    if norm == 'l2':
        J = -1.0 / lr.n * J + lr.Lambda / (2.0 * lr.n) * (theta * theta.T)[0, 0]
    # l1 norm
    else:
        th_sum = 0.0
        row, col = theta.shape
        for i in range(0, row):
            for j in range(0, col):
                th_sum += math.fabs(theta[i, j])
        J = -1.0 / lr.n * J + lr.Lambda / (2.0 * lr.n) * th_sum
    return J

'''bfgs will call this function'''
def _fprime(x0, *args):
    assert isinstance(x0, np.ndarray)
    theta = np.asmatrix(x0)
    lr = args[0]
    assert isinstance(lr, LogisticRegression)
    norm = args[1]
    hx = lr.X * theta.T
    for i in range(0, hx.shape[0]):
        hx[i, 0] = lr._sigmoid(hx[i, 0])
    grad = np.zeros(x0.shape, x0.dtype)
    grad[0] = 1.0 / lr.n * np.sum((np.asarray(hx - lr.y) * np.asarray(lr.X[:, 0])))
    for col in range(1, lr.m + 1):
        if norm == 'l2':
            grad[col] = 1.0 / lr.n * np.sum((np.asarray(hx - lr.y) * np.asarray(lr.X[:, col]))) + lr.Lambda / lr.n * theta[0, col]
        else:
            if theta[0, col] > 0:
                grad[col] = 1.0 / lr.n * np.sum((np.asarray(hx - lr.y) * np.asarray(lr.X[:, col]))) + lr.Lambda / lr.n
            elif theta[0, col] < 0:
                grad[col] = 1.0 / lr.n * np.sum((np.asarray(hx - lr.y) * np.asarray(lr.X[:, col]))) - lr.Lambda / lr.n
            else:
                grad[col] = 1.0 / lr.n * np.sum((np.asarray(hx - lr.y) * np.asarray(lr.X[:, col])))
    return -grad

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
    '''
    def __init__(self, X, y, penalty='l2', Lambda=1.0):
        assert isinstance(X, np.matrix)
        assert isinstance(y, np.matrix)
        # n is number of samples, m is the dimension of features
        (self.n, self.m) = X.shape
        self.X = np.append(np.ones((self.n, 1)), X, 1)
        self.theta = np.zeros((1, self.m + 1))
        self.y = y
        self.penalty = penalty
        self.Lambda = Lambda

    def _sigmoid(self, z):
        # avoid large number
        if z <= -20.0:
            z = -20.0
        if z >= 30.0:
            z = 30.0
        return 1.0 / (1 + math.exp(-z))

    '''gradient descent will call this function'''
    def _cost(self):
        hx = self.X * self.theta.T
        for i in range(0, hx.shape[0]):
            hx[i, 0] = self._sigmoid(hx[i, 0])
        J = 0.0
        for i in range(0, self.n):
            if hx[i, 0] <= 0.0:
                hx[i, 0] = 1e-9
            elif hx[i, 0] >= 1.0:
                hx[i, 0] = 0.999999
            assert hx[i, 0] > 0
            assert 1.0 - hx[i, 0] > 0

            J += self.y[i, 0] * math.log(hx[i, 0]) + (1.0 - self.y[i, 0]) * math.log(1.0 - hx[i, 0])
        J = -1.0 / self.n * J + self.Lambda / (2.0 * self.n) * (self.theta * self.theta.T)[0, 0]
        return J

    '''gradient descent will call this function'''
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

    def fit(self, max_iter=None):
        if self.penalty == 'l2':
            res = \
                scipy.optimize.fmin_bfgs(_obj_func, np.zeros((self.m + 1,)), fprime=_fprime, args=(self, 'l2'), maxiter=max_iter, full_output=1, retall=1)
            xopt = res[0]
            flatten_theta = xopt.flatten()
            for idx in range(0, self.m + 1):
                self.theta[0, idx] = flatten_theta[idx]
        # if not l2, treated as l1
        else:
            res = owlqn.fmin_owlqn(_obj_func, np.zeros((self.m + 1,)), fprime=_fprime, args=(self, 'l1'), maxiter=max_iter, full_output=1, retall=1)
            xopt = res[0]
            flatten_theta = xopt.flatten()
            for idx in range(0, self.m + 1):
                self.theta[0, idx] = flatten_theta[idx]

    '''gradient descent fit'''
    def gd_fit(self, alpha=0.01, max_iter=200, stop_diff=1e-6):
        '''
        alpha: float, only for gradient descent use
            gradient descent step length

        max_iterations: int, only for gradient descent use
            max iteration numbers

        stop_diff: float,
            when the last two costs' diff less than stop_diff, iterate stop
        '''
        J_history = []
        for iter in range(0, max_iter):
            if iter >= max_iter:
                break
            self.theta -= alpha * self._calc_gradient()
            J_history.append(self._cost())
            if iter > 0 and J_history[-2] - J_history[-1] <= stop_diff:
                print 'objective function value descent less than stop_diff or less than 0'
                break
        return J_history

    def predict(self, X):
        assert isinstance(X, np.matrix)
        assert X.shape[1] == self.m
        X_plus = np.append(np.ones((X.shape[0], 1)), X, 1)
        return self._sigmoid(X_plus * self.theta.T)
