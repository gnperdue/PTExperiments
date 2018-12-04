'''
Usage:
    python test_machine.py -v
    python test_machine.py
'''
import logging
import unittest
import numpy as np
from sim.recorders import MachineStateTextRecorder as Recorder
from sim.data_model import DataGenerator as Generator
from sim.data_model import NoiseModel as Noise


class TestMachine(unittest.TestCase):

    def setUp(self):
        gen = Generator()
        noise = Noise()
        recorder = Recorder('./test_log')

    def tearDown(self):
        pass

    def test_basic(self):
        pass


if __name__ == '__main__':
    unittest.main()
