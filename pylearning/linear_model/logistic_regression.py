# -*- coding: utf-8 -*-
# yanchen036@gmail.com

import math
import numpy as np

from .base import LinearModel


class LogisticRegression(LinearModel):
    def __init__(self):
        pass

    def _sigmoid(self, z):
        return 1.0 / (1 + math.exp(-z))

    def fit(self):
        pass

    def predict(self, X):
        pass