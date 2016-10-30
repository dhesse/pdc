import unittest
import scipy
from DataFrame import DataFrame, n, mean, sd
import numpy.testing as npt

class ArrayTest(unittest.TestCase):
    def assertAllClose(self, a, b):
        try:
            npt.assert_allclose(a, b)
        except AssertionError:
            self.fail("{0} and {1} not close!".format(a, b))

class TestGroupBySummarizeN(ArrayTest):

    def test_sum_n(self):
        a = DataFrame(globals(),
                      x = scipy.random.randint(0, 10, 100),
                      y = scipy.random.randint(0, 10, 100))
        s = a.group_by(x, y).summarize(n = n())
        self.assertEqual(s.n.sum(), 100)
    def test_mean(self):
        df = DataFrame(x = scipy.random.randint(0, 10, 100))
        s = df.group_by('x').summarize(mu=mean('x'))
        self.assertAllClose(s.x, s.mu)
    def test_sd(self):
        df = DataFrame(x = scipy.random.randint(0, 10, 100))
        s = df.group_by('x').summarize(sd=sd('x'))
        self.assertAllClose(0, s.sd)
    def test_mean_add(self):
        x = scipy.random.randint(0, 10, 100)
        twox = x * 2
        df = DataFrame(x = x, tx = 2*x)
        glob = df.scope
        s = df.group_by('tx').summarize(txo = mean(glob['x'] +
                                                   glob['x']))
        self.assertAllClose(s.txo, s.tx)

if __name__ == "__main__":
    unittest.main()
