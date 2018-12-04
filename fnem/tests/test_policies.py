'''
Usage:
    python test_policies.py -v
    python test_policies.py
'''
import logging
import unittest
import numpy as np
from policies.rule_based import SimpleRuleBased


class TestSimpleRuleBased(unittest.TestCase):

    def setUp(self):
        start = 0.0
        amplitude = 10.0
        period = np.pi
        self.policy = SimpleRuleBased(start, amplitude, period)

    def tearDown(self):
        pass

    def test_basic(self):
        pass


if __name__ == '__main__':
    unittest.main()
