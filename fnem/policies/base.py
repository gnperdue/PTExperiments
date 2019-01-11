import logging

import torch
import numpy as np
from utils.common_defs import DTYPE


LOGGER = logging.getLogger(__name__)


class BasePolicy(object):
    '''
    intended operation:
    1. observe the machine state
    2. compute a new setting value and apply it to the machine
    '''

    def __init__(self, commands_array):
        '''
        amplitude = max value, period = max -> min -> max t
        '''
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        LOGGER.info('Device = {}'.format(self.device))

        self._setting = None
        self._state = None
        self._commands = np.asarray(commands_array, dtype=DTYPE)

    def get_adjustment_value(self, command):
        return self._commands[command]

    def set_state(self, sensor_array_sequence):
        '''
        sensor_array should be a sequence compoesed of 7 element arrays - one
        for each sensor, the heat, the setting, and t.
        '''
        raise NotImplementedError

    def train(self):
        '''
        train should:
        * zero gradiaents
        * compute loss, add loss to a monitoring log
        * call `backward()`
        * call `optimizer.step()`
        '''
        raise NotImplementedError

    def compute_action(self):
        raise NotImplementedError

    def build_or_restore_model_and_optimizer(self):
        raise NotImplementedError

    # def discount_rewards(self, rewards, gamma=0.99):
    #     '''
    #     rewards should be a `torch.Tensor`
    #     '''
    #     lenr = float(len(rewards))
    #     d_rewards = torch.pow(gamma, torch.arange(lenr)) * rewards
    #     d_rewards = (d_rewards - d_rewards.mean()) / (d_rewards.std() + 1e-07)
    #     return d_rewards

    def loss_fn(self, heats):
        raise NotImplementedError
