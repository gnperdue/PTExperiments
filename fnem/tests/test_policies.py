'''
Usage:
    python test_policies.py -v
    python test_policies.py
'''
import unittest

from sim.engines import SimulationMachine
from policies.rule_based import SimpleRuleBased


class TestSimpleRuleBased(unittest.TestCase):

    def setUp(self):
        start = 0.0
        amplitude = 10.0
        period = 2.0
        machine = SimulationMachine(
            setting=None, data_generator=None, noise_model=None,
            logger=None
        )
        self.policy = SimpleRuleBased(
            time=start, setting=10.0, amplitude=amplitude, period=period,
            commands_array=machine.get_commands()
        )

    def tearDown(self):
        pass

    def test_basic_operation(self):
        state = [10.0, 1.0, 0.5, 0.1, 0.1]
        self.policy.set_state(state)
        self.assertEqual(self.policy._state, state[0:4])
        self.assertEqual(self.policy._time, state[-1])
        setting0 = self.policy._setting
        action = self.policy.compute_action()
        self.policy.update_setting(action)
        setting1 = self.policy._setting
        self.assertIsNotNone(setting1 - setting0)


if __name__ == '__main__':
    unittest.main()
