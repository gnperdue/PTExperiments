'''
Usage:
    python test_policies.py -v
    python test_policies.py
'''
import unittest

from utils.common_defs import DEFAULT_COMMANDS
from policies.base import BasePolicy
from policies.rule_based import SimpleRuleBased


class TestBasePolicy(unittest.TestCase):

    def setUp(self):
        start = 0.0
        amplitude = 10.0
        period = 2.0
        self.policy = BasePolicy(
            time=start, amplitude=amplitude, period=period,
            commands_array=DEFAULT_COMMANDS
        )

    def test_api_methods(self):
        with self.assertRaises(NotImplementedError):
            state = [10.0, 1.0, 0.5, 0.1, 0.1]
            self.policy.set_state(state)
        with self.assertRaises(NotImplementedError):
            self.policy.compute_action()
        with self.assertRaises(NotImplementedError):
            self.policy.build_or_restore_model_and_optimizer()


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
        state = [10.0, 1.0, 0.5, 0.1, 5.0, 10.0, 0.1]
        self.policy.set_state(state)
        self.assertEqual(self.policy._state, state[0:4])
        self.assertEqual(self.policy._setting, state[-2])
        self.assertEqual(self.policy._time, state[-1])

    def test_compute_action(self):
        pass
        # setting0 = self.policy._setting
        # action = self.policy.compute_action()
        # setting1 = self.policy._setting
        # self.assertIsNotNone(setting1 - setting0)


if __name__ == '__main__':
    unittest.main()
