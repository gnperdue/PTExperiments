'''
Usage:
    python test_data.py -v
    python test_data.py
'''
import unittest
import time

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets.historical_data import HistoricalDataset
from datasets.live_data import LiveDataset
from datasets.live_data import LiveToTensor
from utils.common_defs import DATASET_MACHINE_LOG_TEMPLATE


class TestHistoricalDataset(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_live_data(self):
        pass


class TestLiveDataset(unittest.TestCase):

    def setUp(self):
        self.test_time = time.time()
        self.maxsteps = 100
        logname = './' + DATASET_MACHINE_LOG_TEMPLATE % self.test_time
        trnsfrms = transforms.Compose([
            LiveToTensor()
        ])
        self.dataset = LiveDataset(
            maxsteps=self.maxsteps, logname=logname, transform=trnsfrms
        )

    def tearDown(self):
        pass

    def test_dataset_gen(self):
        dl = DataLoader(
            self.dataset, batch_size=10, shuffle=False, num_workers=1
        )
        for i, sample in enumerate(dl):
            print(i, sample)
        self.dataset.close_machine_logger()

    def test_dataset_len(self):
        self.assertEqual(self.maxsteps, len(self.dataset))


if __name__ == '__main__':
    unittest.main()
