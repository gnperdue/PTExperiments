'''
Usage:
    python test_machine.py -v
    python test_machine.py
'''
import unittest
import time
import os

import numpy as np
import matplotlib.pyplot as plt

from sim.recorders import MachineStateTextRecorder as Recorder
from sim.data_model import DataGenerator as Generator
from sim.data_model import NoiseModel as Noise
from sim.engines import SimulationMachine
from policies.rule_based import SimpleRuleBased
from utils.common_defs import DEFAULT_COMMANDS

from tests.common_defs import LOG_TEMPLATE
from tests.common_defs import PLT_TEMPLATE
from tests.common_defs import REFERNECE_LOG
from tests.common_defs import REFERENCE_PLT


class TestMachineWithRuleBased(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.test_time = time.time()
        self.machine_log = './' + LOG_TEMPLATE % self.test_time
        self.machine_plt = './' + PLT_TEMPLATE % self.test_time + '.pdf'

        dgen = Generator()
        nosgen = Noise()
        recorder = Recorder(self.machine_log)
        self.machine = SimulationMachine(
            setting=10.0, data_generator=dgen, noise_model=nosgen,
            logger=recorder, maxsteps=2000
        )
        self.policy = SimpleRuleBased(
            time=0.0, amplitude=10.0, period=2.0,
            commands_array=DEFAULT_COMMANDS
        )

    def tearDown(self):
        pass

    def test_settings(self):
        settings = [10.0, 9.5, 9.125, 8.875, 8.75,
                    8.75, 8.875, 9.125, 9.5]
        for i, s in enumerate(settings):
            self.assertAlmostEqual(self.machine.get_setting(), s)
            self.machine.update_machine(i)
        self.assertAlmostEqual(self.machine.get_setting(), 10.0)

    def test_end_to_end_simplerulebased_run(self):
        sequence_size = 20
        m1 = []
        m2 = []
        m3 = []
        m4 = []
        totals = []
        heats = []
        settings = []
        ts = []
        sequence_buffer = []
        for i in range(5000):
            if self.machine.step():
                t = self.machine.get_time()
                sensor_vals = self.machine.get_sensor_values()
                total = sum(sensor_vals)
                setting = self.machine.get_setting()
                heat = self.machine.get_heat()
                state = sensor_vals + [heat, setting, t]
                ts.append(t)
                totals.append(total)
                settings.append(setting)
                heats.append(heat)
                for i, m in enumerate([m1, m2, m3, m4]):
                    m.append(sensor_vals[i])
                if len(sequence_buffer) < sequence_size:
                    sequence_buffer.append(state)
                else:
                    sequence_buffer.pop(0)
                    sequence_buffer.append(state)
                    self.policy.set_state(sequence_buffer)
                    command = self.policy.compute_action()
                    self.machine.update_machine(command)

        self.machine.close_logger()
        reference_log_size = os.stat(REFERNECE_LOG).st_size
        new_log_size = os.stat(self.machine_log + '.csv.gz').st_size
        self.assertEqual(reference_log_size, new_log_size)

        fig = plt.Figure(figsize=(10, 6))
        gs = plt.GridSpec(1, 4)
        ax1 = plt.subplot(gs[0])
        ax1.scatter(ts, m1, c='r')
        ax1.scatter(ts, m2, c='g')
        ax1.scatter(ts, m3, c='b')
        ax1.scatter(ts, m4, c='y')
        ax2 = plt.subplot(gs[1])
        ax2.scatter(ts, totals, c='k')
        ax3 = plt.subplot(gs[2])
        ax3.scatter(ts, heats, c='k')
        ax4 = plt.subplot(gs[3])
        ax4.scatter(ts, settings, c='k')

        fig.tight_layout()
        plt.savefig(self.machine_plt, bbox_inches='tight')
        reference_plot_size = os.stat(REFERENCE_PLT).st_size
        new_plot_size = os.stat(self.machine_plt).st_size
        relative_size = 100.0 * \
            np.abs(reference_plot_size - new_plot_size) / \
            reference_plot_size
        self.assertTrue(relative_size < 5.0)


if __name__ == '__main__':
    unittest.main()
