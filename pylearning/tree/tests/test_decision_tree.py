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

        self.t = dt.DTree()
        self.t.initialize(self.dataset)

    def tearDown(self):
        pass

    def test_calc_infogain_ratio(self):
        self.assertAlmostEqual(self.node.calc_infogain_ratio(0, self.dataset), 0.083/0.971, places=4)

    def test_train(self):
        self.t.train()
        self.assertEqual(self.t._root.split_fea_idx, 2)
        self.assertAlmostEqual(self.t._root.split_infogain_ratio, 0.4325, places=4)
        self.assertEqual(self.t._root.is_leaf, False)
        self.assertEqual(self.t._root.children[1].is_leaf, True)
        self.assertEqual(self.t._root.children[0].split_fea_idx, 1)
        self.assertAlmostEqual(self.t._root.children[0].split_infogain_ratio, 1.0, places=4)
        self.assertEqual(self.t._root.children[0].is_leaf, False)
        self.assertEqual(self.t._root.children[0].children[0].is_leaf, True)
        self.assertEqual(self.t._root.children[0].children[1].is_leaf, True)


if __name__ == '__main__':
    unittest.main()