'''
Usage:
    python test_utils.py -v
    python test_utils.py
'''
import logging
import unittest
import utils.util_funcs as utils
from utils.common_defs import DEFAULT_COMMANDS
from utils.common_defs import MACHINE_WITH_RULE_REFERNECE_LOG
from utils.common_defs import RUN_MODES

TEST_RULEBASED_ARG_DICT = {
    'start': 0.0, 'amplitude': 10.0, 'period': 2.0,
    'commands_array': DEFAULT_COMMANDS
}


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
        self.assertEqual(logging.INFO,
                         utils.get_logging_level('NoSuchLevel'))

    def test_create_default_arguments_dict(self):
        def test_d(d):
            for k in ['start', 'amplitude', 'period', 'commands_array']:
                self.assertIsNotNone(d[k])

        for i in range(len(RUN_MODES)):
            d = utils.create_default_arguments_dict(
                'NoSuchPolicy', RUN_MODES[i]
            )
            self.assertIsNone(d)

        for i in range(len(RUN_MODES)):
            d = utils.create_default_arguments_dict(
                'SimpleRuleBased', RUN_MODES[i]
            )
            test_d(d)

    def test_create_policy(self):
        policy = utils.create_policy(
            'SimpleRuleBased', TEST_RULEBASED_ARG_DICT
        )
        # `state` is a batch of size N of lists of sensor and settings data.
        state = [[10.0, 1.0, 0.5, 0.1, 5.0, 5.0, 0.1]]
        policy.set_state(state)
        self.assertEqual(policy._state, state[0][0:4])
        self.assertEqual(policy._setting, state[0][-2])
        self.assertEqual(policy._time, state[0][-1])

        with self.assertRaises(ValueError):
            policy = utils.create_policy(
                'NoSuchPolicy', TEST_RULEBASED_ARG_DICT
            )

    def test_create_trainer(self):
        data_source = MACHINE_WITH_RULE_REFERNECE_LOG
        policy = utils.create_policy(
            'SimpleRuleBased', TEST_RULEBASED_ARG_DICT
        )
        mode = 'RUN-TRAINED-HISTORICAL'
        trainer = utils.create_trainer(data_source, policy, mode, 1, 1, 1, 1)
        self.assertEqual(trainer.num_epochs, 1)
        self.assertEqual(trainer.num_steps, 1)
        self.assertEqual(trainer.sequence_size, 1)
        self.assertEqual(trainer.replay_buffer_size, 1)

        with self.assertRaises(ValueError):
            trainer = utils.create_trainer(
                data_source, policy, 'NoSuchMode', 1, 1
            )

    def test_create_data_source(self):
        with self.assertRaises(ValueError):
            data_source = utils.create_data_source('TRAIN-HISTORICAL')

        with self.assertRaises(ValueError):
            data_source = utils.create_data_source('NoSuchMode')


if __name__ == '__main__':
    unittest.main()
