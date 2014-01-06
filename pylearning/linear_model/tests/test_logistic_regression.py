import unittest
import numpy as np

from pylearning.linear_model.logistic_regression import LogisticRegression


class LogisticRegressionTestCase(unittest.TestCase):
    def setUp(self):
        X = np.matrix('1;2;3')
        y = np.matrix('1;0;1')
        self.model = LogisticRegression(X, y)

    def tearDown(self):
        self.model = None

    def test_sigmoid(self):
        self.assertEqual(self.model._sigmoid(0), 0.5)

    def test_cost(self):
        self.model.theta = np.matrix('1,2')
        self.model.Lambda = 0.0
        # -1.0 / 3 * (log(1/(1+exp(-3.0))) + log(1.0-1/(1+exp(-5.0))) + log(1/(1+exp(-7.0)))) = 1.6854047221722177
        self.assertEqual(self.model._cost(), 1.6854047221722177)
        self.model.Lambda = 1.0
        self.assertEqual(self.model._cost(), 2.518738055505551)

    def test_calc_gradient(self):
        self.model.theta = np.matrix('1,2')
        self.model.Lambda = 1.0
        grad = self.model._calc_gradient()
        # 1.0/3*(((1/(1+exp(-3.0)))-1.0)*1 + ((1/(1+exp(-5.0)))-0.0)*1 + ((1/(1+exp(-7.0)))-1.0)*1) = 0.3149900749012493
        self.assertEqual(grad[0, 0], 0.3149900749012493)
        # 1.0/3*(((1/(1+exp(-3.0)))-1.0)*1 + ((1/(1+exp(-5.0)))-0.0)*2 + ((1/(1+exp(-7.0)))-1.0)*3) + 1.0/3*2 = 1.3121517571302206
        self.assertEqual(grad[0, 1], 1.3121517571302206)

if __name__ == '__main__':
    unittest.main()
