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

TEST_LOG = 'test_tmplog.csv'
TEST_LOG_GZ = TEST_LOG + '.gz'


class TestLive(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.live_data = live.LiveData(logname=TEST_LOG)

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
