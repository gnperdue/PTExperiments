import torch

from sim.recorders import MachineStateTextRecorder as Recorder
from sim.data_model import DataGenerator as Generator
from sim.data_model import NoiseModel as Noise
from sim.engines import SimulationMachine


class LiveData(object):

    def __init__(self, setting=10.0, logname='tmplog', random_seed=0):
        self._machine_log = logname
        self._continue = True
        dgen = Generator()
        nosgen = Noise(random_seed=random_seed)
        recorder = Recorder(self._machine_log)
        self.machine = SimulationMachine(setting=setting, data_generator=dgen,
                                         noise_model=nosgen, logger=recorder)

    def close_dataset_logger(self):
        self._continue = False
        self.machine.close_logger()

    def update_setting(self, command):
        self.machine.update_machine(command)

    def get_setting(self):
        return self.machine.get_setting()

    def __iter__(self):
        while self._continue:
            data = self.machine.step()
            observation = torch.Tensor(data[:-3])
            setting = torch.Tensor([data[-3]])
            time = torch.Tensor([data[-2]])
            heat = torch.Tensor([data[-1]])
            yield observation, setting, time, heat
