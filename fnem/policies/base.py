import numpy as np
from utils.common_defs import DTYPE


class BasePolicy(object):
    '''
    intended operation:
    1. observe the machine state
    2. compute a new setting value and apply it to the machine
    '''

    def __init__(
        self, time, amplitude, period, commands_array
    ):
        '''
        amplitude = max value, period = max -> min -> max t
        '''
        self._time = time
        self._setting = None
        self._amplitude = amplitude
        self._period = period
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
