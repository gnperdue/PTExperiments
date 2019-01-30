'''
Usage:
    python test_datasources.py -v
    python test_datasources.py
'''
import unittest
import os

import numpy as np
import torch

import datasources.live as live
from utils.common_defs import DEFAULT_COMMANDS
from utils.common_defs import MAX_SETTING
from utils.common_defs import MIN_SETTING

TEST_LOG = 'test_tmplog.csv'
TEST_LOG_GZ = TEST_LOG + '.gz'


class TestLive(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.live_data = live.LiveData(setting=MIN_SETTING, logname=TEST_LOG)

    def tearDown(self):
        if os.path.isfile(TEST_LOG_GZ):
            os.remove(TEST_LOG_GZ)

    def test_iter(self):
        it = iter(self.live_data)
        for _ in range(3):
            obs, setting, t, heat = next(it)
            self.assertEqual(obs.shape, torch.Size([80]))
            for x in [setting, t, heat]:
                self.assertEqual(x.shape, torch.Size([1]))

        self.live_data.close_dataset_logger()

        with self.assertRaises(StopIteration):
            next(it)

    def test_update_setting(self):
        '''should not go below min setting or above max'''
        self.assertEqual(MIN_SETTING, self.live_data.get_setting())
        self.live_data.update_setting(0)
        new_setting = MIN_SETTING + DEFAULT_COMMANDS[0]
        self.assertEqual(MIN_SETTING, self.live_data.get_setting())
        self.live_data.update_setting(8)
        new_setting = MIN_SETTING + DEFAULT_COMMANDS[8]
        self.assertEqual(new_setting, self.live_data.get_setting())


if __name__ == '__main__':
    unittest.main()
