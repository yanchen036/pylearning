# -*- coding: utf-8 -*-
# yanchen036@gmail.com

from ..util.dataset import Sample, Dataset

class Node():
    def __init__(self):
        lch = None
        rch = None
        sample_idx = None
        split_fea = None
        split_val = None

class DTree():
    def __init__(self):
        self._root = None
        self._dataset = None

    def initialize(self, dataset):
        self._dataset = dataset
        self._root = Node()
        self._root.sample_idx = range(0, self._dataset.getsize())
