# -*- coding: utf-8 -*-
# yanchen036@gmail.com

import unittest
import pylearning.tree.decision_tree as dt

class DecisionTreeTestCase(unittest.TestCase):
    def setUp(self):
        self.node = dt.Node()
        self.dataset = dt.Dataset()
        self.dataset.load('pylearning/tree/tests/dt_data.txt')
        self.node.data = range(0, self.dataset.ins_num)

    def tearDown(self):
        pass

    def test_calc_infogain_ratio(self):
        self.assertAlmostEqual(self.node.calc_infogain_ratio(0, self.dataset), 0.083/0.971, places=4)

if __name__ == '__main__':
    unittest.main()