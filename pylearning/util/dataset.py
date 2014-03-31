# -*- coding: utf-8 -*-
# yanchen036@gmail.com

class Sample():
    def __init__(self, lable=0, fea=[]):
        self.label = lable
        self.fea = fea

class Dataset():
    def __init__(self):
        _samples = []

    def insert(self, s):
        self.samples.append(s)

    def get(self, idx):
        return self.samples[idx]

    def getsize(self):
        return len(self.samples)