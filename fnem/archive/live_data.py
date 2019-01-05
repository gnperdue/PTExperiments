import numpy as np
import torch
from torch.utils.data import Dataset

from sim.recorders import MachineStateTextRecorder as Recorder
from sim.data_model import DataGenerator as Generator
from sim.data_model import NoiseModel as Noise
from sim.engines import SimulationMachine


class LiveToTensor(object):
    '''transform for moving live data to tensors'''

    def __call__(self, sample):
        return torch.from_numpy(sample).float()


class LiveDataset(Dataset):

    def __init__(self, setting=10.0, maxsteps=2000, logname='log.txt',
                 transform=None):
        super(LiveDataset, self).__init__()
        self._maxsteps = maxsteps
        self._machine_log = logname
        self.transform = transform
        dgen = Generator()
        nosgen = Noise()
        recorder = Recorder(self._machine_log)
        self.machine = SimulationMachine(
            setting=setting, data_generator=dgen, noise_model=nosgen,
            logger=recorder, maxsteps=self._maxsteps
        )

    def close_dataset_logger(self):
        self.machine.close_logger()

    def update_setting(self, command):
        self.machine.update_machine(command)

    def __len__(self):
        return self._maxsteps

    def __getitem__(self, idx):
        self.machine.step()
        t = self.machine.get_time()
        sensor_vals = self.machine.get_sensor_values()
        setting = self.machine.get_setting()
        heat = self.machine.get_heat()
        # TODO - can we go straight from list to tensor?
        state = np.array(sensor_vals + [heat, setting, t])
        if self.transform:
            state = self.transform(state)
        return state
