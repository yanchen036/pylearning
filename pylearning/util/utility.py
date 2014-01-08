# -*- coding: utf-8 -*-
# yanchen036@gmail.com

import numpy as np

def normalize(fea):
    assert isinstance(fea, np.matrix)
    mean = fea.mean(0)
    std = fea.std(0)
    new_fea = np.matrix(fea)
    for row in range(0, fea.shape[0]):
        new_fea[row] = (fea[row] - mean) / std
    return new_fea

'''struct stored predict value and ground truth value'''
class pred_gnd():
    def __init__(self, pred=0.0, gnd=0):
        self.pred = pred
        self.gnd = gnd

def ROC(pg_arr):
    assert isinstance(pg_arr, list)
    for pg in pg_arr:
        assert isinstance(pg, pred_gnd)
    if (pg_arr.__len__() == 0):
        return list()
    ret = [[0.0, 0.0]]
    for i in range(99, 0, -1):
        threshold = float(i) / 100
        tp, fp, fn, tn = 0.0, 0.0, 0.0, 0.0
        for pg in pg_arr:
            if (pg.pred >= threshold):
                if (pg.gnd == 0):
                    fp += 1
                else:
                    tp += 1
            else:
                if (pg.gnd == 0):
                    tn += 1
                else:
                    fn += 1
        tp_rate = tp / (tp + fn)
        fp_rate = fp / (fp + tn)
        ret.append([tp_rate, fp_rate])
    ret.append([1.0, 1.0])
    return ret