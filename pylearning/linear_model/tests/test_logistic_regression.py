import unittest
import numpy as np

from pylearning.linear_model.logistic_regression import LogisticRegression


class LogisticRegressionTestCase(unittest.TestCase):
    def setUp(self):
        self.model = LogisticRegression(x, y, 0.1, 120)

    def tearDown(self):
        self.model = None

    def test_sigmoid(self):
        self.assertEqual(self.model._sigmoid(0), 0.5)


if __name__ == '__main__':
    unittest.main()
