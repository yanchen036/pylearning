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

data_fp = open('./data/lr_data1.txt')
data = data_fp.readlines()
for line in data:
    [x, y, label] = line.strip().split(',')
    if label == '0':
        x1.append(float(x))
        y1.append(float(y))
    else:
        x2.append(float(x))
        y2.append(float(y))
pl.scatter(x1, y1, color='yellow', marker='o')
pl.scatter(x2, y2, color='black', marker='+')

arr_x = []
arr_y = []
for i in range(0, x1.__len__()):
    arr_x.append([x1[i], y1[i]])
    arr_y.append(0)
for i in range(0, x2.__len__()):
    arr_x.append([x2[i], y2[i]])
    arr_y.append(1)
mat_x = np.matrix(arr_x)
mat_y = np.matrix(arr_y)
(row_y, col_y) = mat_y.shape
mat_y = mat_y.reshape((col_y, row_y))

# this case at least need 1000000 iterations
model = LogisticRegression(mat_x, mat_y, alpha=0.001, Lambda=0.0, max_iterations=1000000, stop_diff=1e-9)
J_history = model.fit()
print J_history.__len__()
print model.theta
X_test = np.linspace(30, 100, 70)
y_test = []
for x in X_test:
    y_test.append((-model.theta[0, 0] - model.theta[0, 1] * x) / model.theta[0, 2])
pl.plot(X_test, y_test, linewidth=1)

pl.figure(2, figsize=(8, 6))
iter = range(1, J_history.__len__() + 1)
pl.plot(iter, J_history, color='green', linewidth=1)
pl.show()
