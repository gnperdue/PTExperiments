import numpy as np
from utils.common_defs import DTYPE


class BasePolicy(object):
    '''
    intended operation:
    1. observe the machine state
    2. compute a new setting value and apply it
    '''

    def __init__(
        self, time, setting, amplitude, period, commands_array
    ):
        '''
        amplitude = max value, period = max -> min -> max t
        '''
        self._time = time
        self._setting = setting
        self._amplitude = amplitude
        self._period = period
        self._state = None
        self._commands = np.asarray(commands_array, dtype=DTYPE)

    def set_state(self, sensor_array):
        '''
        sensor_array should be a minibatch compoesed of 5 element arrays - one
        for each sensor, t.
        '''
        raise NotImplementedError

    def compute_action(self):
        raise NotImplementedError

    def update_setting(self, command_index):
        '''command_index is the index of the step change'''
        self._setting = self._setting + self._commands[command_index]

    def build_or_restore_model_and_optimizer(self):
        raise NotImplementedError
