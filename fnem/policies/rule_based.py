import numpy as np
from sim.data_model import DTYPE


class SimpleRuleBased(object):
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
        sensor_array should be a 5 element array - one for each sensor, t.
        this policy ignores state and only uses t

        TODO - this should take a minibatch, not a single point
        '''
        self._state = sensor_array[0:4]
        self._time = sensor_array[-1]

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

    def update_setting(self, command):
        '''command is the index of the step change'''
        self._setting = self._setting + self._commands[command]
