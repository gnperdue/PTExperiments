import logging
import torch
import numpy as np
from .base import BasePolicy
import utils.util_funcs as utils


LOGGER = logging.getLogger(__name__)


class SimpleMLP(BasePolicy):

    def __init__(self, commands_array, learning_rate=None, ckpt_path=None):
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
        self._ckpt_path = ckpt_path or '/tmp/simple_mlp/ckpt.tar'
        self._learning_rate = learning_rate or 0.0009
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self._learning_rate)
        self.lossfn = torch.nn.MSELoss()

    def set_state(self, sensor_array_sequence):
        '''flatten the array sequence'''
        state = torch.stack(sensor_array_sequence)
        return state.view(-1)

    def train(self):
        '''
        train should:
        * zero gradiaents
        * compute loss, add loss to a monitoring log
        * call `backward()`
        * call `optimizer.step()`
        '''
        pass

    def compute_action(self):
        '''
        call forward pass on the NN model
        '''
        return 0

    def build_or_restore_model_and_optimizer(self):
        '''
        '''
        LOGGER.info('model has {} parameters'.format(
            utils.count_parameters(self.model)
        ))
        try:
            checkpoint = torch.load(self._ckpt_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            LOGGER.info('Loaded checkpoint from {}'.format(self._ckpt_path))
        except FileNotFoundError:
            LOGGER.info('No checkpoint found...')

        LOGGER.debug('Model state dict:')
        for param_tensor in self.model.state_dict():
            LOGGER.debug(str(param_tensor) + '\t'
                         + str(self.model.state_dict()[param_tensor].size()))
        LOGGER.debug('Optimizer state dict:')
        for var_name in self.optimizer.state_dict():
            LOGGER.debug(str(var_name) + '\t'
                         + str(self.optimizer.state_dict()[var_name]))

        self.model.to(self.device)

    def loss_fn(self, heats):
        target = torch.zeros(heats.shape)
        output = self.lossfn(heats, target)
        return output
