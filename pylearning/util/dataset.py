# -*- coding: utf-8 -*-
# yanchen036@gmail.com

class Instance():
    def __init__(self, lable, fea=[]):
        self.label = lable
        self.fea = fea

class Dataset():
    def __init__(self):
        self._data = []
        # key: fea_idx, value: set of fea_val
        self._feaset = {}
        self.ins_num = 0
        self.fea_num = 0

    def load(self, file):
        fp = open(file)
        for line in fp:
            label, rest = line.strip().split('^')
            fea_s = rest.split(';')
            fea = []
            for i in xrange(0, len(fea_s)):
                _,val = fea_s[i].split(':')
                fea.append(int(val))
                if i in self._feaset:
                    self._feaset[i].add(int(val))
                else:
                    self._feaset[i] = set([int(val)])
            ins = Instance(label, fea)
            self._data.append(ins)
        fp.close()
        self.ins_num = len(self._data)
        self.fea_num = len(self._feaset)

    def get_ins(self, idx):
        return self._data[idx]

    def get_fea_val_domain(self, feaidx):
        return self._feaset[feaidx]
