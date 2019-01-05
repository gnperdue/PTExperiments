import torch
import numpy as np
from .base import BasePolicy


class SimpleMLP(BasePolicy):

    def __init__(self, commands_array):
        super(SimpleMLP, self).__init__(commands_array)
        self.pytorch = True
        l1 = 4
        l2 = 150
        l3 = 2
        self.model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.Softmax(dim=0)
        )
        learning_rate = 0.0009
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)

    def set_state(self, sensor_array_sequence):
        '''
        1. flatten the array sequence
        2. convert to Tensor
        '''
        pass

    def train(self):
        '''
        '''
        pass

    def compute_action(self):
        '''
        '''
        pass

    def build_or_restore_model_and_optimizer(self):
        '''
        '''
        pass

    def loss_fn(preds, r):
        return -1 * torch.sum(r * torch.log(preds))
