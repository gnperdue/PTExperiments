import numpy as np
from sim.data_model import DTYPE


class SimulationMachine(object):
    '''
    intended operation:
    1. update the machine setting
    2. step the machine
        a. step the data generator (generate data and advance the time)
        b. add noise
        c. update sensor values
        d. if logging, record machine state
    3. report the "heat" (difference between machine setting and true state)
    '''
    # default_commands = np.linspace(-0.5, 0.5, num=9, dtype=DTYPE)
    default_commands = np.array([-0.5, -0.375, -0.25, -0.125, 0.0,
                                 0.125, 0.25, 0.375, 0.5], dtype=DTYPE)

    def __init__(
        self, setting, data_generator, noise_model, logger=None, commands=None
    ):
        self._data_generator = data_generator
        self._noise_model = noise_model
        self._setting = setting
        self._heat = 0.0
        self._true_state = 0.0
        self._commands = commands or SimulationMachine.default_commands
        self._sensors = np.zeros(4, dtype=DTYPE)
        if logger is None:
            self._live_mode = True
        else:
            self._live_mode = False
            self._logger = logger

    def update_machine(self, command):
        '''command is the index of the step change'''
        self._setting = self._setting + self._commands[command]

    def step(self):
        data = self._data_generator.step()
        self._true_state = np.sum(data)
        noise = self._noise_model.gen_noise(data)
        measured = data + noise
        self._sensors = measured

    def get_heat(self):
        return self._true_state - self._setting

    def get_time(self):
        return self._data_generator.t

    def get_setting(self):
        return self._setting

    def get_commands(self):
        return list(self._commands)

    def get_sensor_values(self):
        return list(self._sensors)
