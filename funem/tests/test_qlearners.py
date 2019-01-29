'''
Usage:
    python test_qlearners.py -v
    python test_qlearners.py
'''
import unittest

import qlearners.qbase as qbase
import qlearners.simple_mlp as simple_mlp
import qlearners.simple_rulebased as simple_rulebased
from utils.common_defs import DEFAULT_COMMANDS


class TestQBase(unittest.TestCase):

    def setUp(self):
        self.learner = qbase.BaseQ(DEFAULT_COMMANDS)

    def tearDown(self):
        pass

    def test_compute_qvalues(self):
        with self.assertRaises(NotImplementedError):
            observation = None
            self.learner.compute_qvalues(observation)


class TestSimpleMLP(unittest.TestCase):

    def setUp(self):
        d = {}
        d['commands_array'] = DEFAULT_COMMANDS
        self.learner = simple_mlp.SimpleMLP(train_pars_dict=d)

    def tearDown(self):
        pass

    def test_compute_qvalues(self):
        self.fail('Finish the test...')


class TestSimpleRuleBased(unittest.TestCase):

    def setUp(self):
        d = {}
        d['commands_array'] = DEFAULT_COMMANDS
        self.learner = simple_rulebased.SimpleRuleBased(train_pars_dict=d)

    def tearDown(self):
        pass

    def test_compute_qvalues(self):
        self.fail('Finish the test...')


if __name__ == '__main__':
    unittest.main()
