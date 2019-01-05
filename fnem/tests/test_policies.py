'''
Usage:
    python test_policies.py -v
    python test_policies.py
'''
import unittest

import torch

from utils.common_defs import DEFAULT_COMMANDS
from policies.base import BasePolicy
from policies.rule_based import SimpleRuleBased
from policies.simple_mlp import SimpleMLP


class TestBasePolicy(unittest.TestCase):

    def setUp(self):
        self.policy = BasePolicy(commands_array=DEFAULT_COMMANDS)

    def test_api_methods(self):
        with self.assertRaises(NotImplementedError):
            state = [[10.0, 1.0, 0.5, 0.1, 5.0, 5.0, 0.1]]
            self.policy.set_state(state)
        with self.assertRaises(NotImplementedError):
            self.policy.compute_action()
        with self.assertRaises(NotImplementedError):
            self.policy.train()
        with self.assertRaises(NotImplementedError):
            self.policy.build_or_restore_model_and_optimizer()
        with self.assertRaises(NotImplementedError):
            self.policy.loss_fn(preds=None, r=None)

    def test_get_adjustments(self):
        for i, command in enumerate(DEFAULT_COMMANDS):
            self.assertEqual(self.policy.get_adjustment_value(i), command)

    def test_discount_rewards(self):
        rewards = torch.Tensor([1.0, 1.0, 10.0])
        answer = torch.Tensor([-0.5764, -0.5783,  1.1547])
        d_rewards = self.policy.discount_rewards(rewards)
        for i in range(len(rewards)):
            self.assertAlmostEqual(d_rewards[i].item(), answer[i].item(),
                                   places=4)


class TestSimpleMLP(unittest.TestCase):

    def setUp(self):
        self.policy = SimpleMLP(commands_array=DEFAULT_COMMANDS)

    def tearDown(self):
        pass

    def test_set_state(self):
        pass


class TestSimpleRuleBased(unittest.TestCase):

    def setUp(self):
        start = 0.0
        amplitude = 10.0
        period = 2.0
        self.policy = SimpleRuleBased(
            time=start, amplitude=amplitude, period=period,
            commands_array=DEFAULT_COMMANDS
        )

    def tearDown(self):
        pass

    def test_set_state(self):
        # `state` is a batche of size N of lists of sensor and settings data.
        state = [[10.0, 1.0, 0.5, 0.1, 5.0, 10.0, 0.1]]
        self.policy.set_state(state)
        self.assertEqual(self.policy._state, state[0][0:4])
        self.assertEqual(self.policy._setting, state[0][-2])
        self.assertEqual(self.policy._time, state[0][-1])

    def test_compute_action(self):
        pass


if __name__ == '__main__':
    unittest.main()
