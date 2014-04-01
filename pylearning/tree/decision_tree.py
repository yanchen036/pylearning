# -*- coding: utf-8 -*-
# yanchen036@gmail.com

from ..util.dataset import Instance, Dataset

class Node():
    def __init__(self):
        self.children = None
        self.children_split_val = None
        self.data_list = []
        self.split_fea = None
        self._label = None

    def split(self, dataset):
        assert self.data_list != None
        for feaidx in range(0, self.dataset.get_fea_num()):
            # calc info gain
            pass

        self.children = []
        self.children_split_val = []
        for idx in self.data_list:
            ins = dataset.get_ins(idx)
            if ins.fea[self.split_fea] in self.children_feaval:
                chidx = self.children_split_val.index(ins.fea[self.split_fea])
                self.children[chidx].data_list.append(idx)
            else:
                self.children.append(Node())
                self.children[-1].data_list.append(idx)

    def get_label(self, dataset):
        if self._label is None:
            label_dict = {}
            max_cnt = 0
            max_label = None
            for idx in self.data_list:
                ins = dataset.get_ins(idx)
                if ins.label in label_dict.keys():
                    label_dict[ins.label] += 1
                else:
                    label_dict[ins.label] = 1
            for l, cnt in label_dict.iteritems():
                if cnt > max_cnt:
                    max_cnt = cnt
                    max_label = l
            self._label = max_label
        return self._label




class DTree():
    def __init__(self):
        self._root = None
        self._dataset = None

    def initialize(self, dataset):
        self._dataset = dataset
        self._root = Node()
        self._root.sample_idx = range(0, self._dataset.getsize())