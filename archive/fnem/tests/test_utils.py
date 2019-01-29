'''
Usage:
    python test_utils.py -v
    python test_utils.py
'''
import logging
import unittest

import torch
import torch.nn as nn

import utils.util_funcs as utils
from utils.common_defs import DEFAULT_COMMANDS
from utils.common_defs import MACHINE_WITH_RULE_REFERNECE_LOG
from utils.common_defs import RUN_MODES

TEST_RULEBASED_ARG_DICT = {
    'start': 0.0, 'amplitude': 10.0, 'period': 2.0,
    'commands_array': DEFAULT_COMMANDS
}
TEST_MLP_ARG_DICT = {
    'ckpt_path': '/tmp/simple_mlp/ckpt.tar', 'learning_rate': 1e-4,
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
        with self.assertRaises(ValueError):
            for i in range(len(RUN_MODES)):
                d = utils.create_default_arguments_dict(
                    'NoSuchPolicy', RUN_MODES[i]
                )

        for i in range(len(RUN_MODES)):
            d = utils.create_default_arguments_dict(
                'SimpleRuleBased', RUN_MODES[i]
            )
            for k in ['start', 'amplitude', 'period', 'commands_array']:
                self.assertIsNotNone(d[k])

        for i in range(len(RUN_MODES)):
            d = utils.create_default_arguments_dict(
                'SimpleMLP', RUN_MODES[i]
            )
            for k in ['learning_rate', 'ckpt_path', 'commands_array']:
                self.assertIsNotNone(d[k])

    def test_create_policy(self):
        policy = utils.create_policy(
            'SimpleRuleBased', TEST_RULEBASED_ARG_DICT
        )
        # `state` is a batch of size N of lists of sensor and settings data.
        state = [torch.Tensor([10.0, 1.0, 0.5, 0.1, 5.0, 0.1])]
        heat = [torch.Tensor([5.0])]
        policy.set_state(state, heat)
        for a, b in zip(policy._state, state[0].numpy()[0:4]):
            self.assertEqual(a, b)
        self.assertEqual(policy._setting, state[0].numpy()[-2])
        self.assertEqual(policy._time, state[0].numpy()[-1])

        policy = utils.create_policy(
            'SimpleMLP', TEST_MLP_ARG_DICT
        )
        self.assertIsNotNone(policy)

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
        trainer = utils.create_trainer(data_source, policy, mode, 1, 1, 1)
        self.assertEqual(trainer.num_epochs, 1)
        self.assertEqual(trainer.num_steps, 1)
        self.assertEqual(trainer.sequence_size, 1)

        with self.assertRaises(ValueError):
            trainer = utils.create_trainer(
                data_source, policy, 'NoSuchMode', 1, 1
            )

    def test_create_data_source(self):
        with self.assertRaises(ValueError):
            data_source = utils.create_data_source('TRAIN-HISTORICAL')

        with self.assertRaises(ValueError):
            data_source = utils.create_data_source('NoSuchMode')

        data_source = utils.create_data_source(
            'TRAIN-HISTORICAL', MACHINE_WITH_RULE_REFERNECE_LOG
        )
        self.assertEqual(data_source.get_setting(), 10.0)
        data_source.adjust_setting(-1.0)
        self.assertEqual(data_source.get_setting(), 9.0)

    def test_count_parameters(self):

        class Net(nn.Module):
            '''sizes for 28x28 image'''

            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3)
                self.conv2 = nn.Conv2d(32, 64, 3)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout1 = nn.Dropout(0.25)
                self.dropout2 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(64 * 12 * 12, 128)
                self.fc2 = nn.Linear(128, 10)

            # def forward(self, x):
            #     x = F.relu(self.conv1(x))
            #     x = F.relu(self.conv2(x))
            #     x = self.dropout1(self.pool(x))
            #     x = x.view(-1, 64 * 12 * 12)
            #     x = F.relu(self.fc1(x))
            #     x = self.dropout2(x)
            #     x = self.fc2(x)
            #     return x

        net = Net()
        self.assertEqual(1199882, utils.count_parameters(net))


if __name__ == '__main__':
    unittest.main()
