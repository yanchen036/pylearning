#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import pylab as pl
from pylearning.util import utility

gn_arr = []
for i in range(0, 500):
    gn_arr.append(utility.pred_gnd(random.uniform(0.1, 1), 1))
for i in range(0, 500):
    gn_arr.append(utility.pred_gnd(random.uniform(0, 0.9), 0))
roc = utility.ROC(gn_arr)

tpr = []
fpr = []
for i in range(0, roc.__len__()):
    tpr.append(roc[i][0])
    fpr.append(roc[i][1])
pl.figure(1, figsize=(8, 6))
pl.clf()
pl.plot(fpr, tpr, color='green', linewidth=1)
pl.show()
