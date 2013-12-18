import unittest

import numpy as np

from pylearning.linear_model.linear_regression import LinearRegressionModel

class LinearRegressionModelTestCase(unittest.TestCase):
    def setUp(self):
        x = np.matrix('1;0;2')
        y = np.matrix('1;-1;3')
        self.model = LinearRegressionModel(x, y)

    def tearDown(self):
        self.model = None

    def test_cost(self):
        cost = self.model._cost()
        self.assertEqual(cost, 11./6)

    def test_calc_gradient(self):
        theta = self.model._calc_gradient()
        self.assertEqual(theta[0, 0], 0.01)
        self.assertEqual(theta[0, 1], 7./3*0.01)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(LinearRegressionModelTestCase('test_cost'))
    suite.addTest(LinearRegressionModelTestCase('test_calc_gradient'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
