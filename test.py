import unittest
import scipy
from DataFrame import DataFrame, n

class TestGroupBySummarizeN(unittest.TestCase):

    def test_sum_n(self):
        a = DataFrame(x = scipy.random.randint(0, 10, 100),
                      y = scipy.random.randint(0, 10, 100))
        s = a.group_by('x', 'y').summarize(n = n())
        self.assertEqual(s.n.sum(), 100)

if __name__ == "__main__":
    unittest.main()
