'''
Usage:
    python test_machine_recorder.py -v
    python test_machine_recorder.py
'''
import unittest
import numpy as np
import sim.recorders as recorders


class TestMachineRecorder(unittest.TestCase):

    def setUp(self):
        self.log_name = './test_log'
        self.recorder = recorders.MachineStateTextRecorder(self.log_name)

    def tearDown(self):
        self.recorder.cleanup_files()

    def test_writeread_data(self):
        # write a line of dummy data
        t0, set0 = 1.0, 7.0
        meas0 = np.asarray([10.1, 1.1, 0.6, 0.2], dtype=np.float32)
        targ0 = np.asarray([10.0, 1.0, 0.5, 0.1], dtype=np.float32)
        a0 = np.concatenate(([t0, set0], meas0, targ0))
        success = self.recorder.write_data(t0, set0, meas0, targ0)
        self.assertTrue(success)
        # write another line of dummy data
        t1, set1 = 1.1, 7.0
        meas1 = np.asarray([10.2, 0.9, 0.4, 0.2], dtype=np.float32)
        targ1 = np.asarray([10.3, 1.1, 0.3, 0.1], dtype=np.float32)
        a1 = np.concatenate(([t1, set1], meas1, targ1))
        success = self.recorder.write_data(t1, set1, meas1, targ1)
        self.assertTrue(success)
        # close and zip the file
        self.recorder.close()
        # read the contents into an array of strings
        content = self.recorder.read_data()
        # parse the content
        line0 = map(float, content[0].split(','))
        line1 = map(float, content[1].split(','))
        # check we read what we wrote
        for a, b in zip(a0, line0):
            self.assertAlmostEqual(a, b, places=6)
        for a, b in zip(a1, line1):
            self.assertAlmostEqual(a, b, places=6)


if __name__ == '__main__':
    unittest.main()
