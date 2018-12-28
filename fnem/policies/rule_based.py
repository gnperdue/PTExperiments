import numpy as np
from .base import BasePolicy


class SimpleRuleBased(BasePolicy):

    def __init__(
        self, time, amplitude, period, commands_array
    ):
        super(SimpleRuleBased, self).__init__(
            time, amplitude, period, commands_array
        )

    def set_state(self, sensor_array_sequence):
        '''
        this policy ignores the full state and only uses the most recent t
        in the sequence
        '''
        sensor_array = sensor_array_sequence[-1]
        self._state = sensor_array[0:4]
        self._setting = sensor_array[-2]
        self._time = sensor_array[-1]

    def train(self):
        '''
        this policy does not train and only uses the most recent t to predict
        a setting
        '''
        pass

    def compute_action(self):
        '''
        pure time-based, want to go from ampl to -ampl as t goes from
        0->1, then from -ampl to ampl as t goes from 1->2
        '''
        t = self._time % 2
        if t < (self._period / 2.0):
            slope = -2.0 * self._amplitude
            intercept = self._amplitude
        if t >= (self._period / 2.0):
            slope = 2.0 * self._amplitude
            intercept = -3.0 * self._amplitude
        target = slope * t + intercept
        delta = target - self._setting
        diffs = np.abs(self._commands - delta)
        command = np.argmin(diffs)
        return command

    def build_or_restore_model_and_optimizer(self):
        '''function for API compatability only'''
        pass
