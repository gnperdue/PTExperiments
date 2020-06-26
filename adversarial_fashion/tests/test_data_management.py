'''
Usage:
    python test_data_management.py -v
    python test_data_management.py
'''
import unittest
import torch
import math

import tests.utils as utils


class TestDataManagers(unittest.TestCase):

    def setUp(self):
        self.dm = utils.configure_and_get_testing_data_manager()

    def test_dataloaders(self):
        batch_size = 100
        train_loader, test_loader = self.dm.get_data_loaders(
            batch_size=batch_size)
        for i, (inputs, labels) in enumerate(train_loader):
            inputs_l = list(inputs.shape)
            labels_l = list(labels.shape)
            for idx, j in enumerate([batch_size, 1, 28, 28]):
                self.assertEqual(inputs_l[idx], j)
            self.assertEqual(labels_l[0], batch_size)
            self.assertTrue(math.isclose(
                torch.mean(inputs).item(), 0, rel_tol=0.05, abs_tol=0.05))
            self.assertTrue(math.isclose(
                torch.std(inputs).item(), 1.0, rel_tol=0.05, abs_tol=0.05))


if __name__ == '__main__':
    unittest.main()
