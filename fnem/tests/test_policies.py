'''
Usage:
    python test_policies.py -v
    python test_policies.py
'''
import unittest

from utils.common_defs import DEFAULT_COMMANDS
from policies.base import BasePolicy
from policies.rule_based import SimpleRuleBased
from policies.simple_mlp import SimpleMLP


class TestBasePolicy(unittest.TestCase):

    def setUp(self):
        self.policy = BasePolicy(commands_array=DEFAULT_COMMANDS)

    def test_api_methods(self):
        with self.assertRaises(NotImplementedError):
            state = [[10.0, 1.0, 0.5, 0.1, 5.0, 5.0, 0.1]]
            self.policy.set_state(state)
        with self.assertRaises(NotImplementedError):
            self.policy.compute_action()
        with self.assertRaises(NotImplementedError):
            self.policy.train()
        with self.assertRaises(NotImplementedError):
            self.policy.build_or_restore_model_and_optimizer()


class TestSimpleMLP(unittest.TestCase):

    def setUp(self):
        self.policy = SimpleMLP(commands_array=DEFAULT_COMMANDS)

    def tearDown(self):
        pass

    def test_set_state(self):
        pass


class TestSimpleRuleBased(unittest.TestCase):

    def setUp(self):
        start = 0.0
        amplitude = 10.0
        period = 2.0
        self.policy = SimpleRuleBased(
            time=start, amplitude=amplitude, period=period,
            commands_array=DEFAULT_COMMANDS
        )

    def tearDown(self):
        pass

    def test_set_state(self):
        # `state` is a batche of size N of lists of sensor and settings data.
        state = [[10.0, 1.0, 0.5, 0.1, 5.0, 10.0, 0.1]]
        self.policy.set_state(state)
        self.assertEqual(self.policy._state, state[0][0:4])
        self.assertEqual(self.policy._setting, state[0][-2])
        self.assertEqual(self.policy._time, state[0][-1])

    def test_compute_action(self):
        pass


if __name__ == '__main__':
    unittest.main()
