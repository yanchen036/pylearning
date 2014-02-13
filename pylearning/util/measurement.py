# -*- coding: utf-8 -*-
# yanchen036@gmail.com

from operator import attrgetter

'''struct stored predict value and ground truth value'''
class Pgr():
    def __init__(self, pred=0.0, gnd=0, rank=0):
        self.pred = pred
        self.gnd = gnd
        self.rank = rank

def ROC(pgr_arr):
    assert isinstance(pgr_arr, list)
    for pgr in pgr_arr:
        assert isinstance(pgr, Pgr)
    if pgr_arr.__len__() == 0:
        return list()
    ret = [[0.0, 0.0]]
    for i in range(99, 0, -1):
        threshold = float(i) / 100
        tp, fp, fn, tn = 0.0, 0.0, 0.0, 0.0
        for pgr in pgr_arr:
            if pgr.pred >= threshold:
                if pgr.gnd == 0:
                    fp += 1
                else:
                    tp += 1
            else:
                if pgr.gnd == 0:
                    tn += 1
                else:
                    fn += 1
        tp_rate = tp / (tp + fn)
        fp_rate = fp / (fp + tn)
        ret.append([tp_rate, fp_rate])
    ret.append([1.0, 1.0])
    return ret

def AUC(pgr_arr):
    assert isinstance(pgr_arr, list)
    for pgr in pgr_arr:
        assert isinstance(pgr, Pgr)
    if pgr_arr.__len__() == 0:
        return 0.0
    pgr_arr = sorted(pgr_arr, key=attrgetter(pred), reverse=True)
    # rank
    for i, item in enumerate(triple):
        item.rank = pgr_arr.__len__() - i
    # for same predict, average their rank
    start, end = 0, 0
    while end < pgr_arr.__len__():
        sum = pgr_arr[start].rank
        end += 1
        while end < pgr_arr.__len__() and pgr_arr[start].pred == pgr_arr[end].pred:
            sum += pgr_arr[end].rank
            end += 1
        len = end - start
        if len > 1:
            avg = float(sum) / len
            for i in range(start, end):
                pgr_arr[i].rank = avg
        start = end
    pos_num = 0
    neg_num = 0
    pos_rank_sum = 0.0
    for item in pgr_arr:
        if item.gnd == 1:
            pos_num += 1
            pos_rank_sum += item.rank
        else:
            neg_num += 1

    return (pos_rank_sum - pos_num * (pos_num + 1) / 2.0) / (pos_num * neg_num)
