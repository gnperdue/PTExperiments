'''
Usage:
    python test_models.py -v
    python test_models.py
'''
import unittest
import numpy as np
import torch
from torch.autograd import Variable

import gridrl.models as models


class TestBasicModels(unittest.TestCase):

    def setUp(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        self.model = models.build_basic_model()
        self.model.to(self.device)
        state_ = np.random.rand(1, 64)
        self.random_state = Variable(
            torch.from_numpy(state_).float()
        ).to(self.device)

    def test_output_shape(self):
        actions = self.model(self.random_state)
        self.assertEqual(actions.shape, (1, 4))


class TestConvModels(unittest.TestCase):

    def setUp(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        self.model = models.build_conv_model()
        self.model.to(self.device)
        state_ = np.random.rand(1, 4, 4, 4)
        self.random_state = Variable(
            torch.from_numpy(state_).float()
        ).to(self.device)

    def test_output_shape(self):
        actions = self.model(self.random_state)
        self.assertEqual(actions.shape, (1, 4))


if __name__ == '__main__':
    unittest.main()
