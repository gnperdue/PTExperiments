'''
Usage:
    python test_datasources.py -v
    python test_datasources.py
'''
import unittest
import time
import os

import torch
import numpy as np

from datasources.live import LiveData
from datasources.historical import HistoricalData
from utils.common_defs import DATASET_MACHINE_LOG_TEMPLATE
from utils.common_defs import DATASET_MACHINE_REFERENCE_LOG
from utils.common_defs import MACHINE_WITH_RULE_REFERNECE_LOG
from utils.common_defs import DEFAULT_COMMANDS


class TestLiveData(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.test_time = time.time()
        self.maxsteps = 100
        self.logname = './' + DATASET_MACHINE_LOG_TEMPLATE % self.test_time
        self.setting = 10.0
        self.dataset = LiveData(
            setting=self.setting, maxsteps=self.maxsteps, logname=self.logname
        )

    def tearDown(self):
        pass

    def test_data_gen(self):
        for i, sample in enumerate(self.dataset):
            self.assertEqual(sample.shape, torch.Size([7]))
        self.assertEqual(i, self.maxsteps - 1)
        self.dataset.close_dataset_logger()
        reference_log_size = os.stat(DATASET_MACHINE_REFERENCE_LOG).st_size
        new_log_size = os.stat(self.logname + '.csv.gz').st_size
        self.assertEqual(reference_log_size, new_log_size)

    def test_dataset_len(self):
        self.assertEqual(self.maxsteps, len(self.dataset))

    def test_update_settings(self):
        self.dataset.update_setting(0)
        self.assertEqual(self.setting + DEFAULT_COMMANDS[0],
                         self.dataset.get_setting())


class TestHistoricalData(unittest.TestCase):

    def setUp(self):
        self.epochs = 2
        self.dataset = HistoricalData(
            setting=10.0, source_file=MACHINE_WITH_RULE_REFERNECE_LOG
        )

    def tearDown(self):
        pass

    def test_data_read(self):
        print('\n')
        for ep in range(self.epochs):
            for i, data in enumerate(self.dataset):
                # if i == 10:
                #     print(i, data[0], data[1])
                self.assertEqual(data[0].shape, torch.Size([7]))
                self.assertEqual(data[1].shape, torch.Size([4]))
            self.assertEqual(i, 1999)
        self.assertEqual(ep, self.epochs - 1)


if __name__ == '__main__':
    unittest.main()
