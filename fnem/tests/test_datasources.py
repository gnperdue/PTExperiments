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
from utils.common_defs import DATASET_MACHINE_LOG_TEMPLATE
from utils.common_defs import DATASET_MACHINE_REFERENCE_LOG


class TestLiveData(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.test_time = time.time()
        self.maxsteps = 100
        self.logname = './' + DATASET_MACHINE_LOG_TEMPLATE % self.test_time
        self.dataset = LiveData(maxsteps=self.maxsteps, logname=self.logname)

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


if __name__ == '__main__':
    unittest.main()
