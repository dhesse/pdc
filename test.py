import unittest
import scipy
from DataFrame import DataFrame, n, mean, sd, TEMP_COLS
import numpy.testing as npt

class TestGroupBySummarizeN(unittest.TestCase):

    def test_sum_n(self):
        a = DataFrame(globals(),
                      x = scipy.random.randint(0, 10, 100),
                      y = scipy.random.randint(0, 10, 100))
        s = a.group_by(x, y).summarize(n = n())
        self.assertEqual(s.n.sum(), 100)
    def test_mean(self):
        df = DataFrame(x = scipy.random.randint(0, 10, 100))
        s = df.group_by('x').summarize(mu=mean('x'))
        try:
            npt.assert_allclose(s.x, s.mu)
        except AssertionError:
            self.fail("s.x and s.mu not close")
    def test_sd(self):
        df = DataFrame(x = scipy.random.randint(0, 10, 100))
        s = df.group_by('x').summarize(sd=sd('x'))
        try:
            npt.assert_allclose(0, s.sd)
        except AssertionError:
            self.fail("s.x and s.sd not close")
    def test_mean_add(self):
        x = scipy.random.randint(0, 10, 100)
        twox = x * 2
        df = DataFrame(x = x, tx = 2*x)
        glob = df.scope
        print glob
        s = df.group_by('tx').summarize(txo = mean(glob['x'] +
                                                   glob['x']))
        try:
            npt.assert_allclose(s.txo, s.tx)
        except AssertionError:
            self.fail("txo and tx not close")

if __name__ == "__main__":
    unittest.main()
