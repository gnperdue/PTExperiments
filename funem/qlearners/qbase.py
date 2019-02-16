import logging
import random

import numpy as np
import torch
from torch.autograd import Variable

from utils.common_defs import DTYPE
from utils.common_defs import MAX_HEAT


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
        self.epsilon = -1.0
        self.gamma = None
        self._batch_size = 20  # should be smaller than trainer replay buffer
        self._setting = None
        self._state = None
        self._heats = None
        self._commands = np.asarray(commands_array, dtype=DTYPE)
        self._command_idcs = list(range(len(self._commands)))

    def get_adjustment_value(self, command):
        return self._commands[command]

    def compute_qvalues(self, observation, use_target=False):
        '''return an array of q-values for each action in the commands_array'''
        raise NotImplementedError

    def compute_action(self, qvalues):
        '''compute the action index'''
        raise NotImplementedError

    def build_trainbatch(self, replay_buffer):
        '''build a training batch and targets from the replay buffer'''
        X_train, y_train = None, None
        minibatch = random.sample(replay_buffer, self._batch_size)
        X_train = Variable(torch.empty(
            self._batch_size, self.noutputs, dtype=torch.float
        ))
        y_train = Variable(torch.empty(
            self._batch_size, self.noutputs, dtype=torch.float
        ))
        # Fill X_train and y_train minibatch tensors by index `h` by
        # looping through memory and computing the Q-values before (X)
        # and after (y) each move
        h = 0
        for memory in minibatch:
            # reward is heat, and we want head to be small, but here we need
            # a large "reward", so use max_heat - heat instead of the raw heat
            # value.
            old_state, action_m, reward_m, new_state_m = memory
            old_qval = self.compute_qvalues(old_state)
            # TODO - use a target model...
            # new_qval = self.target_model(new_state_m).cpu().data.numpy()
            new_qval = self.compute_qvalues(new_state_m, use_target=True)
            max_qval = np.max(new_qval.cpu().data.numpy())
            y = torch.zeros((1, self.noutputs))
            y[:] = old_qval[:]
            update = (MAX_HEAT - reward_m) + (self.gamma * max_qval)
            y[0][action_m] = update
            X_train[h] = old_qval
            y_train[h] = Variable(y)
            h += 1
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        return X_train, y_train

    def train(self, X_train, y_train):
        '''
        train should:
        * compute loss
        * zero gradiaents
        * call `backward()`
        * call `optimizer.step()`
        '''
        raise NotImplementedError

    def restore_model_and_optimizer(self):
        # set gamma to a vanilla value so `build_trainbatch` will function.
        if self.gamma is None:
            self.gamma = 0.99

    def anneal_epsilon(self, step):
        raise NotImplementedError

    def save_model(self, epoch, step):
        raise NotImplementedError
