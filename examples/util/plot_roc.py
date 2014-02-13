#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import pylab as pl
from pylearning.util import measurement

gnr_arr = []
for i in range(0, 500):
    gnr_arr.append(measurement.pred_gnd(random.uniform(0, 1), 1, 0))
for i in range(0, 500):
    gnr_arr.append(measurement.pred_gnd(random.uniform(0, 1), 0, 0))
roc = measurement.ROC(gnr_arr)
print measurement.AUC(roc)
tpr = []
fpr = []
for i in range(0, roc.__len__()):
    tpr.append(roc[i][0])
    fpr.append(roc[i][1])
pl.figure(1, figsize=(8, 6))
pl.clf()
pl.plot(fpr, tpr, color='green', linewidth=1)
pl.show()
