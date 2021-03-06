#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import numpy as np
import pylab as pl

from pylearning.linear_model.linear_regression import LinearRegression

pl.figure(1, figsize=(8, 6))
pl.clf()

# target function is y = 2x-1, generate 20 samples
X = []
y = []
for x in range(1, 20, 1):
    X.append(float(x))
    y.append(2 * x - 1.0 + random.gauss(0, 2))
    X.append(float(x) + 0.5)
    y.append(2 * (float(x) + 0.5) - 1.0 + random.gauss(0, 2))
pl.scatter(X, y, color='red', marker='x')
pl.ylabel('y')
pl.xlabel('x')

mat_x = np.matrix(X)
(row_x, col_x) = mat_x.shape
mat_x = mat_x.reshape((col_x, row_x))
mat_y = np.matrix(y)
(row_y, col_y) = mat_y.shape
mat_y = mat_y.reshape((col_y, row_y))

model = LinearRegression(mat_x, mat_y)
J_history = model.gd_fit(alpha=0.001, max_iterations=500)
X_test = np.linspace(0, 20, 20)
y_test = []
for x in X_test:
    y_test.append(model.predict(np.matrix(x))[0, 0])
pl.plot(X_test, y_test, linewidth=1)

pl.figure(2, figsize=(8, 6))
iter = range(1, J_history.__len__() + 1)
pl.plot(iter, J_history, color='green', linewidth=1)
pl.show()
