'''
Usage:
    python test_sim.py -v
    python test_sim.py
'''
import unittest
import time
import glob
import os

import numpy as np
# import matplotlib.pyplot as plt
# import torch

from sim.recorders import MachineStateTextRecorder as Recorder
from sim.data_model import DataGenerator as Generator
from sim.data_model import NoiseModel as Noise
from sim.engines import SimulationMachine
from utils.common_defs import MACHINE_LOG_TEMPLATE
from utils.common_defs import MACHINE_LOG_REFERNECE


TEST_LOG = 'test_tmplog.csv'
TEST_LOG_GZ = TEST_LOG + '.gz'


class TestEngine(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.test_time = time.time()
        self.machine_log = './' + \
            MACHINE_LOG_TEMPLATE % self.test_time

        dgen = Generator()
        nosgen = Noise()
        recorder = Recorder(self.machine_log)
        self.machine = SimulationMachine(setting=10.0, data_generator=dgen,
                                         noise_model=nosgen, logger=recorder)

    def tearDown(self):
        files = glob.glob(self.machine_log + '*')
        for f in files:
            if os.path.isfile(f):
                os.remove(f)

    def test_step_and_log(self):
        for i in range(100):
            data = self.machine.step()
            self.assertEqual(len(data), 83)
        self.machine.close_logger()
        reference_log_size = os.stat(MACHINE_LOG_REFERNECE).st_size
        new_log_size = os.stat(self.machine_log + '.gz').st_size
        self.assertEqual(reference_log_size, new_log_size)


class TestRecorders(unittest.TestCase):

    def setUp(self):
        self.recorder = Recorder(log_name=TEST_LOG)

    def tearDown(self):
        if os.path.isfile(TEST_LOG_GZ):
            os.remove(TEST_LOG_GZ)

    def test_recorder(self):
        data = [[10, 11, 12], [13, 14, 15]]
        for d in data:
            self.recorder.write_data(d)
        self.recorder.close()
        read = self.recorder.read_data()
        for i, r in enumerate(read):
            parts = list(map(int, r.split(',')))
            self.assertEqual(data[i], parts)


if __name__ == '__main__':
    unittest.main()
