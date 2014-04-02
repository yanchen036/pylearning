# -*- coding: utf-8 -*-
# yanchen036@gmail.com

class Instance():
    def __init__(self, lable=0, fea=[]):
        self.label = lable
        self.fea = fea

class Dataset():
    def __init__(self):
        self._data = []
        # key: fea_idx, value: list of fea_val
        self._feaset = {}

    def load(self, file):
        pass

    def get_ins(self, idx):
        return self._data[idx]

    def get_ins_len(self):
        return len(self._data)

    def get_fea_val_domain(self, feaidx):
        return self._feaset[feaidx]

    def get_fea_num(self):
        assert self.data is not None and len(self.data) > 0
        return len(self.data[0].fea)