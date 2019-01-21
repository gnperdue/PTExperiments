import logging
import torch
import numpy as np
from .qbase import BaseQ
import utils.util_funcs as utils


LOGGER = logging.getLogger(__name__)


class SimpleMLP(BaseQ):

    def __init__(self, commands_array, learning_rate=None, ckpt_path=None):
        super(SimpleMLP, self).__init__(commands_array)
        self.pytorch = True
        l1 = 4
        l2 = 150
        l3 = len(commands_array)
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
        # TODO - just sum squares by hand and avoid making a vector of zeros?
        self.lossfn = torch.nn.MSELoss()

    def set_state(self, sensor_array_sequence, heats_sequence):
        '''the sequences are batches of observations'''
        # state = torch.stack(sensor_array_sequence)
        # state = state[0:5]  # remove time from the state - not meaningful here
        # self._state = state.view(-1)
        # heats = torch.stack(heats_sequence).view(-1)
        # self._heats = heats
        pass

    def train(self):
        '''
        train should:
        * zero gradiaents
        * compute loss, add loss to a monitoring log
        * call `backward()`
        * call `optimizer.step()`
        '''
        # targets = torch.zeros(self._heats.shape)
        # loss = self.lossfn(self._heats, targets)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # return loss.item()
        pass

    def compute_action(self):
        '''
        call forward pass on the NN model
        '''
        preds = self.model(self._state)
        action = np.random.choice(self._command_idcs, p=preds.data.numpy())
        return action

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
