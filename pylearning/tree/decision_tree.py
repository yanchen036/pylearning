# -*- coding: utf-8 -*-
# yanchen036@gmail.com

from ..util.dataset import Instance, Dataset
import math

class Node():
    def __init__(self):
        self.children = None
        self.children_split_val = None
        self.data = []
        self.split_fea = None
        self._label = None

    def split(self, dataset):
        assert self.data != None
        max_ratio = 0.
        sp_fea = None
        for feaidx in range(0, self.dataset.get_fea_num()):
            ratio = self.calc_infogain_ratio(feaidx, dataset)
            if ratio > max_ratio:
                max_ratio = ratio
                sp_fea = feaidx
        self.split_fea = sp_fea

        self.children = []
        self.children_split_val = []
        for idx in self.data:
            ins = dataset.get_ins(idx)
            if ins.fea[self.split_fea] in self.children_feaval:
                chidx = self.children_split_val.index(ins.fea[self.split_fea])
                self.children[chidx].data_list.append(idx)
            else:
                self.children.append(Node())
                self.children[-1].data_list.append(idx)

    def _calc_label_dict(self, dataset):
        label_dict = {}
        for idx in self.data:
            ins = dataset.get_ins(idx)
            if ins.label in label_dict:
                label_dict[ins.label] += 1
            else:
                label_dict[ins.label] = 1
        return label_dict

    def get_label(self, dataset):
        if self._label is None:
            label_dict = self._calc_label_dict(dataset)
            max_cnt = 0
            max_label = None
            for l, cnt in label_dict.iteritems():
                if cnt > max_cnt:
                    max_cnt = cnt
                    max_label = l
            self._label = max_label
        return self._label

    def calc_infogain_ratio(self, fea_x, dataset):
        d = len(self.data)
        ck = self._calc_label_dict(dataset)
        # number of instance whose fea_x value is i and label is k
        dik = {}
        for idx in self.data:
            ins = dataset.get_ins(idx)
            feaxval = ins.fea[fea_x]
            if feaxval in dik:
                dik[feaxval][0] += 1
                if ins.label in dik[feaxval][1]:
                    dik[feaxval][1][ins.label] += 1
                else:
                    dik[feaxval][1][ins.label] = 1
            else:
                dik[feaxval] = [1, {ins.label:1}]
        entropy = 0.0
        for l,c in ck.iteritems():
            entropy += -(float(ck[l]) / d) * math.log(float(ck[l]) / d, 2)
        condition_entropy = 0.0
        for f,li in dik.iteritems():
            single_fea_entropy = 0.0
            for l,c in li[1].iteritems():
                single_fea_entropy += -(float(c) / li[0]) * math.log(float(c) / li[0], 2)
            condition_entropy += -(float(li[0]) / d * single_fea_entropy)
        infogain = entropy - condition_entropy
        infogain_ratio = infogain / entropy
        return infogain_ratio

class DTree():
    def __init__(self):
        self._root = None
        self._dataset = None

    def initialize(self, dataset):
        self._dataset = dataset
        self._root = Node()
        self._root.sample_idx = range(0, self._dataset.getsize())