# -*- coding: utf-8 -*-
# yanchen036@gmail.com

import random
import numpy as np
import pylab as pl

from pylearning.linear_model.logistic_regression import LogisticRegression

pl.figure(1, figsize=(8, 6))
pl.clf()

x1 = []
y1 = []
x2 = []
y2 = []
for i in range(0, 20):
    x1.append(random.gauss(3, 1.0))
    y1.append(random.gauss(5, 1.6))
    x2.append(random.gauss(5, 0.3))
    y2.append(random.gauss(3, 0.7))
pl.scatter(x1, y1, color='red', marker='x')
pl.scatter(x2, y2, color='blue', marker='+')

arr_x = []
arr_y = []
for i in range(0, 10):
    arr_x.append([x1[i], y1[i]])
    arr_y.append(0)
for i in range(0, 10):
    arr_x.append([x2[i], y2[i]])
    arr_y.append(1)
mat_x = np.matrix(arr_x)
mat_y = np.matrix(arr_y)
(row_y, col_y) = mat_y.shape
mat_y = mat_y.reshape((col_y, row_y))

model = LogisticRegression(mat_x, mat_y, alpha=0.2, Lambda=1.0, num_iters=100)
J_history = model.fit()
print model.theta
X_test = np.linspace(0, 10, 10)
y_test = []
for x in X_test:
    y_test.append((-model.theta[0, 0] - model.theta[0, 1] * x) / model.theta[0, 2])
pl.plot(X_test, y_test, linewidth=1)

pl.figure(2, figsize=(8, 6))
iter = range(1, J_history.__len__() + 1)
pl.plot(iter, J_history, color='green', linewidth=1)
pl.show()
