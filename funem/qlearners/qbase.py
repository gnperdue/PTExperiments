import logging

import torch
import numpy as np
from utils.common_defs import DTYPE


LOGGER = logging.getLogger(__name__)


class BaseQ(object):
    '''
    intended operation:
    1. observe the machine state
    2. compute the Q-value of taking an action
    3. use the maximum Q to choose a new setting value
    '''

    def __init__(self, commands_array):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        LOGGER.info('Device = {}'.format(self.device))
        self.noutputs = len(commands_array)

        self._batch_size = 20  # should be smaller than trainer replay buffer
        self._setting = None
        self._state = None
        self._heats = None
        self._commands = np.asarray(commands_array, dtype=DTYPE)
        self._command_idcs = list(range(len(self._commands)))

    def get_adjustment_value(self, command):
        return self._commands[command]

    def compute_qvalues(self):
        '''return an array of q-values for each action in the commands_array'''
        raise NotImplementedError

    def compute_action(self, qvalues):
        '''compute the action index'''
        raise NotImplementedError

    def build_trainbatch(self, replay_buffer):
        '''build a training batch and targets from the replay buffer'''
        raise NotImplementedError

    def train(self, X_train, y_train):
        '''
        train should:
        * zero gradiaents
        * compute loss, add loss to a monitoring log
        * call `backward()`
        * call `optimizer.step()`
        '''
        raise NotImplementedError

    def build_or_restore_model_and_optimizer(self):
        raise NotImplementedError
