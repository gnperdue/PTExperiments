'''
Usage:
    python test_utils.py -v
    python test_utils.py
'''
import logging
import unittest
import utils.util_funcs as utils
from utils.common_defs import DEFAULT_COMMANDS


class TestUtils(unittest.TestCase):

    def test_get_logging_level(self):
        self.assertEqual(logging.INFO,
                         utils.get_logging_level('info'))
        self.assertEqual(logging.INFO,
                         utils.get_logging_level('INFO'))
        self.assertEqual(logging.DEBUG,
                         utils.get_logging_level('debug'))
        self.assertEqual(logging.DEBUG,
                         utils.get_logging_level('DEBUG'))
        self.assertEqual(logging.WARNING,
                         utils.get_logging_level('warning'))
        self.assertEqual(logging.WARNING,
                         utils.get_logging_level('WARNING'))
        self.assertEqual(logging.ERROR,
                         utils.get_logging_level('error'))
        self.assertEqual(logging.ERROR,
                         utils.get_logging_level('ERROR'))
        self.assertEqual(logging.CRITICAL,
                         utils.get_logging_level('critical'))
        self.assertEqual(logging.CRITICAL,
                         utils.get_logging_level('CRITICAL'))

    def test_create_policy(self):
        arguments_dict = {
            'start': 0.0, 'setting': 10.0, 'amplitude': 10.0, 'period': 2.0,
            'commands_array': DEFAULT_COMMANDS
        }
        policy = utils.create_policy('SimpleRuleBased', arguments_dict)
        state = [10.0, 1.0, 0.5, 0.1, 0.1]
        policy.set_state(state)
        self.assertEqual(policy._state, state[0:4])
        self.assertEqual(policy._time, state[-1])
        setting0 = policy._setting
        action = policy.compute_action()
        policy.update_setting(action)
        setting1 = policy._setting
        self.assertIsNotNone(setting1 - setting0)

        with self.assertRaises(ValueError):
            policy = utils.create_policy('NoSuchPolicy', arguments_dict)


if __name__ == '__main__':
    unittest.main()
