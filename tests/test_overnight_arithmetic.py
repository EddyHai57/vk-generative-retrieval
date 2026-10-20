import unittest
from analysis.overnight_arithmetic import pairs, utilization


class ArithmeticTests(unittest.TestCase):
    def test_short_sequences_exact_enumeration(self):
        for length in range(12):
            expected = sum(1 for a in range(length) for b in range(length) if 0 < abs(a-b) <= 5)
            self.assertEqual(pairs({str(length): 1}), expected)

    def test_frozen_utilization(self):
        self.assertEqual(utilization(120415, 48), [1., 1., 1.])
        self.assertLess(utilization(120415, 64)[-1], .8)
        self.assertAlmostEqual(utilization(12012895, 256)[-1], 12012895/256**3)


if __name__ == '__main__':
    unittest.main()
