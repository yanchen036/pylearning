import unittest
import numpy as np

from pylearning.linear_model.linear_regression import LinearRegression

class LinearRegressionTestCase(unittest.TestCase):
    def setUp(self):
        # y = 2x-1
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

    def test_fit(self):
        J_history = self.model.fit()
        self.assertAlmostEqual(self.model.theta[0, 0], -1., delta=0.1)
        self.assertAlmostEqual(self.model.theta[0, 1], 2., delta=0.1)
        self.assertAlmostEqual(J_history[-1], 0., places=2)

    def test_predict(self):
        x = 3.0
        X = np.matrix(x)
        self.model.fit()
        y = self.model.predict(X)
        self.assertAlmostEqual(y[0, 0], 5., delta=0.1)

if __name__ == '__main__':
    #suite = unittest.TestSuite()
    #suite.addTest(LinearRegressionTestCase('test_cost'))
    #suite.addTest(LinearRegressionTestCase('test_calc_gradient'))
    #suite.addTest(LinearRegressionTestCase('test_fit'))
    #suite.addTest(LinearRegressionTestCase('test_predict'))

    #runner = unittest.TextTestRunner()
    #runner.run(suite)

    unittest.main()
