'''
Usage:
    python test_models.py -v
    python test_models.py
'''
import unittest
import numpy as np
import torch
from torch.autograd import Variable
import ptlib.models as models


class TestBasicModels(unittest.TestCase):

    def setUp(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        self.model = models.SimpleConvNet()
        self.model.to(self.device)
        image_ = np.random.rand(1, 1, 28, 28)
        self.random_image = Variable(
            torch.from_numpy(image_).float()
        ).to(self.device)

    def test_output_shape(self):
        output = self.model(self.random_image)
        self.assertEqual(output.shape, (1, 10))


if __name__ == '__main__':
    unittest.main()
