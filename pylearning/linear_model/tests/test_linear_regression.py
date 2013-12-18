import unittest
import numpy as np

from pylearning.linear_model.linear_regression import LinearRegression

class LinearRegressionModelTestCase(unittest.TestCase):
    def setUp(self):
        x = np.matrix('1;0;2')
        y = np.matrix('1;-1;3')
        self.model = LinearRegression(x, y, 0.1, 120)

    def tearDown(self):
        self.model = None

    def test_cost(self):
        cost = self.model._cost()
        self.assertEqual(cost, 11./6)

    def test_calc_gradient(self):
        theta = self.model._calc_gradient()
        self.assertEqual(theta[0, 0], self.model.alpha)
        self.assertAlmostEqual(theta[0, 1], 7./3*self.model.alpha)

    def test_gradient_descent(self):
        [theta, J_history] = self.model.gradient_descent()
        self.assertAlmostEqual(theta[0, 0], -1., delta=0.1)
        self.assertAlmostEqual(theta[0, 1], 2., delta=0.1)
        self.assertAlmostEqual(J_history[-1], 0., places=2)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(LinearRegressionModelTestCase('test_cost'))
    suite.addTest(LinearRegressionModelTestCase('test_calc_gradient'))
    suite.addTest(LinearRegressionModelTestCase('test_gradient_descent'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
