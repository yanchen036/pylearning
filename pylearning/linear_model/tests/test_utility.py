import unittest
import numpy as np

from pylearning.util.fea_process import normalize

class UtilityTestCase(unittest.TestCase):
    def test_normalize(self):
        fea = np.matrix("1.0, 2.0, 3.0; 2.0, 3.0, 4.0")
        fea = normalize(fea)
        self.assertTrue(np.array_equal(fea, np.matrix("-1.0, -1.0, -1.0; 1.0, 1.0, 1.0")))

if __name__ == '__main__':
    unittest.main()
