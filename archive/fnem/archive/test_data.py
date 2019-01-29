'''
Usage:
    python test_data.py -v
    python test_data.py
'''
import unittest
import os
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets.historical_data import HistoricalDataset
from datasets.live_data import LiveDataset
from datasets.live_data import LiveToTensor
from utils.common_defs import DATASET_MACHINE_LOG_TEMPLATE
from utils.common_defs import DATASET_MACHINE_REFERENCE_LOG


class TestHistoricalDataset(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_live_data(self):
        pass


class TestLiveDataset(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.test_time = time.time()
        self.maxsteps = 100
        self.logname = './' + DATASET_MACHINE_LOG_TEMPLATE % self.test_time
        trnsfrms = transforms.Compose([
            LiveToTensor()
        ])
        self.dataset = LiveDataset(
            maxsteps=self.maxsteps, logname=self.logname, transform=trnsfrms
        )

    def tearDown(self):
        pass

    def test_dataset_gen(self):
        batch_size = 10
        dl = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, num_workers=1
        )
        for i, sample in enumerate(dl):
            self.assertEqual(sample.shape, torch.Size([10, 7]))
        self.assertEqual(i, self.maxsteps / batch_size - 1)
        # we need to kill some time before closing the logger since the
        # dataloader is doing its thing on a different thread
        time.sleep(1)
        self.dataset.close_dataset_logger()
        reference_log_size = os.stat(DATASET_MACHINE_REFERENCE_LOG).st_size
        new_log_size = os.stat(self.logname + '.csv.gz').st_size
        self.assertEqual(reference_log_size, new_log_size)

    def test_dataset_len(self):
        self.assertEqual(self.maxsteps, len(self.dataset))


if __name__ == '__main__':
    unittest.main()
