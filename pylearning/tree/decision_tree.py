# -*- coding: utf-8 -*-
# yanchen036@gmail.com

from ..util.dataset import Instance, Dataset
import math

class Node():
    def __init__(self):
        self.children = []
        self.children_split_val = []
        self.data = []
        self.split_fea_idx = None
        self.split_infogain_ratio = 0.0
        self.label = None
        self.is_leaf = False

    def split(self, dataset, eps):
        assert self.data != None
        max_ratio = 0.
        sp_fea = None
        for feaidx in range(0, dataset.fea_num):
            ratio = self.calc_infogain_ratio(feaidx, dataset)
            if ratio > max_ratio:
                max_ratio = ratio
                sp_fea = feaidx
        self.split_fea_idx = sp_fea
        self.split_infogain_ratio = max_ratio
        if max_ratio >= eps:
            for feaval in dataset.get_fea_val_domain(self.split_fea_idx):
                self.children_split_val.append(feaval)
                self.children.append(Node())
            for idx in self.data:
                ins = dataset.get_ins(idx)
                if ins.fea[self.split_fea_idx] in self.children_split_val:
                    chidx = self.children_split_val.index(ins.fea[self.split_fea_idx])
                    self.children[chidx].data.append(idx)
                else:
                    self.children.append(Node())
                    self.children_split_val.append(ins.fea[self.split_fea_idx])
                    self.children[-1].data.append(idx)

    def calc_label_dict(self, dataset):
        label_dict = {}
        for idx in self.data:
            ins = dataset.get_ins(idx)
            if ins.label in label_dict:
                label_dict[ins.label] += 1
            else:
                label_dict[ins.label] = 1
        return label_dict

    def get_label(self, dataset):
        if self.label is None:
            label_dict = self.calc_label_dict(dataset)
            max_cnt = 0
            max_label = None
            for l, cnt in label_dict.iteritems():
                if cnt > max_cnt:
                    max_cnt = cnt
                    max_label = l
            self.label = max_label
        return self.label

    def calc_infogain_ratio(self, fea_x, dataset):
        d = len(self.data)
        ck = self.calc_label_dict(dataset)
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
                single_fea_entropy += (float(c) / li[0]) * math.log(float(c) / li[0], 2)
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
        self._root.data = range(0, self._dataset.ins_num)

    def train(self, eps=0.1):
        st = [self._root]
        while len(st) > 0:
            cur = st.pop()
            # current node is empty
            if len(cur.data) == 0:
                cur.is_leaf = True
                continue
            ld = cur.calc_label_dict(self._dataset)
            # current node only has one class data
            if len(ld) <= 1:
                cur.is_leaf = True
                continue
            cur.split(self._dataset, eps)
            if len(cur.children) == 0:
                cur.is_leaf = True
            for i in range(0, len(cur.children)):
                st.append(cur.children[i])

    def predict(self, ins):
        assert isinstance(ins, Instance)
        cur = self._root
        while True:
            if cur.is_leaf:
                break
            for i in xrange(0, len(cur.children_split_val)):
                if ins.fea[cur.split_fea_idx] == cur.children_split_val[i]:
                    cur = cur.children[i]
                    break
        return cur.get_label(self._dataset)

    def serialize(self):
        model = ''
        st = [self._root]
        while len(st) > 0:
            cur = st.pop()

            fea_idx = '_'
            if cur.split_fea_idx:
                fea_idx = str(cur.split_fea_idx)
            val_list = ''
            for v in cur.children_split_val:
                val_list += str(v) + ','
            val_list = val_list[0:-1]
            leaf = '0'
            label = '_'
            if cur.is_leaf:
                leaf = '1'
                label = str(cur.get_label(self._dataset))
                val_list = '_'

            # format
            # is_leaf:label:split_fea:val1,val2,...;
            model +=  leaf + ':' + label +':' + fea_idx + ':' + val_list + ';'
            for node in cur.children:
                st.append(node)
        return model
