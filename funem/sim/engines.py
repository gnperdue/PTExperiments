import numpy as np
from collections import deque

# from utils.common_defs import DTYPE
from utils.common_defs import DEFAULT_COMMANDS
from utils.common_defs import MAX_HEAT
from utils.common_defs import MAX_SETTING
from utils.common_defs import MIN_SETTING


class SimulationMachine(object):
    '''
    intended operation...

    when finished, call `close_logger()` to zip the log file.
    '''

    def __init__(
        self, setting, data_generator, noise_model, logger=None, commands=None
    ):
        self._data_generator = data_generator
        self._noise_model = noise_model
        self._setting = setting
        self._commands = commands or DEFAULT_COMMANDS
        self._logger = logger
        self._true_instantaneous_sensor_vals = None
        self._observation = deque([], maxlen=80)
        self._initialize()

    def _initialize(self):
        for i in range(20):
            data = self._data_generator.step()
            noise = self._noise_model.gen_noise(data)
            measured = data + noise
            for m in measured:
                self._observation.append(m)
        self._true_instantaneous_sensor_vals = list(data)

    def _true_state(self):
        return sum(self._true_instantaneous_sensor_vals)

    def update_machine(self, command):
        '''command is the index of the step change'''
        self._setting = self._setting + self._commands[command]
        self._setting = min(self._setting, MAX_SETTING)
        self._setting = max(self._setting, MIN_SETTING)

    def step(self):
        data = self._data_generator.step()
        self._true_instantaneous_sensor_vals = list(data)
        noise = self._noise_model.gen_noise(data)
        measured = data + noise
        for m in measured:
            self._observation.append(m)
        heat = self.get_heat()

        return_value = list(self._observation) + \
            [self._setting, self._data_generator.t, heat]

        if self._logger is not None:
            self._logger.write_data(return_value)

        return return_value

    def get_heat(self):
        return min((self._true_state() - self._setting) ** 2, MAX_HEAT)

    def get_time(self):
        return self._data_generator.t

    def get_setting(self):
        return self._setting

    def get_commands(self):
        return list(self._commands)

    def get_sensor_values(self):
        return list(self._observation)[-4:]

    def close_logger(self):
        self._logger.close()
