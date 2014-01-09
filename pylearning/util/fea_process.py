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