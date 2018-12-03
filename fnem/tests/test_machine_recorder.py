'''
Usage:
    python test_utils.py -v
    python test_utils.py
'''
import logging
import unittest
import numpy as np
import sim.recorders as recorders


class TestMachineRecorder(unittest.TestCase):

    def setUp(self):
        self.log_name = './test_log'
        self.recorder = recorders.MachineStateTextRecorder(self.log_name)

    def tearDown(self):
        self.recorder.cleanup_files()

    def test_write_data(self):
        t = 1.0
        meas = np.asarray([10.1, 1.1, 0.6, 0.2], dtype=np.float32)
        targ = np.asarray([10.0, 1.0, 0.5, 0.1], dtype=np.float32)
        success = self.recorder.write_data(t, meas, targ)
        self.assertTrue(success)
        t = 1.1
        meas = np.asarray([10.2, 0.9, 0.4, 0.2], dtype=np.float32)
        targ = np.asarray([10.3, 1.1, 0.3, 0.1], dtype=np.float32)
        success = self.recorder.write_data(t, meas, targ)
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
