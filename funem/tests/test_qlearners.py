'''
Usage:
    python test_qlearners.py -v
    python test_qlearners.py
'''
import unittest
import os

import numpy as np
import torch

import datasources.live as live
import qlearners.qbase as qbase
import qlearners.simple_mlp as simple_mlp
import qlearners.simple_rulebased as simple_rulebased
from utils.common_defs import DEFAULT_COMMANDS

TEST_LOG = 'test_tmplog.csv'
TEST_LOG_GZ = TEST_LOG + '.gz'


class TestQBase(unittest.TestCase):

    def setUp(self):
        self.learner = qbase.BaseQ(DEFAULT_COMMANDS)

    def tearDown(self):
        pass

    def test_get_adjustment_value(self):
        for i in range(len(DEFAULT_COMMANDS)):
            self.assertEqual(self.learner.get_adjustment_value(i),
                             DEFAULT_COMMANDS[i])

    def test_notimplementeds(self):
        with self.assertRaises(NotImplementedError):
            observation = None
            self.learner.compute_qvalues(observation)
        with self.assertRaises(NotImplementedError):
            self.learner.compute_action(None)
        with self.assertRaises(NotImplementedError):
            self.learner.build_trainbatch(None)
        with self.assertRaises(NotImplementedError):
            self.learner.train(None, None)
        with self.assertRaises(NotImplementedError):
            self.learner.build_or_restore_model_and_optimizer()
        with self.assertRaises(NotImplementedError):
            self.learner.anneal_epsilon(None)
        with self.assertRaises(NotImplementedError):
            self.learner.save_model(None, None)


class TestSimpleMLP(unittest.TestCase):

    def setUp(self):
        d = {}
        d['commands_array'] = DEFAULT_COMMANDS
        self.learner = simple_mlp.SimpleMLP(train_pars_dict=d)

    def tearDown(self):
        pass

    def test_compute_qvalues(self):
        self.fail('Finish the test...')


class TestSimpleRuleBased(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.live_data = live.LiveData(setting=10.0, logname=TEST_LOG)

        d = {}
        d['commands_array'] = DEFAULT_COMMANDS
        self.learner = simple_rulebased.SimpleRuleBased(train_pars_dict=d)

    def tearDown(self):
        if os.path.isfile(TEST_LOG_GZ):
            os.remove(TEST_LOG_GZ)

    def test_compute_qvalues_and_action(self):
        observation = torch.Tensor([10.5, 1.0, 1.0, 0.1,
                                    10.0, 0.9, 0.9, 0.09,
                                    9.5, 0.8, 0.8, 0.08])
        qvals = self.learner.compute_qvalues(observation)
        testvals = torch.Tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(qvals.shape, testvals.shape)
        for i in range(testvals.shape[0]):
            self.assertEqual(qvals[i], testvals[i])
        self.assertEqual(self.learner.compute_action(qvals), 0)

    def test_training(self):
        self.assertEqual(self.learner.train(None, None), -1.0)
        self.assertEqual(self.learner.build_trainbatch(None), (None, None))

    def test_do_nothings(self):
        self.learner.build_or_restore_model_and_optimizer()
        self.learner.anneal_epsilon(None)
        self.learner.save_model(None, None)


if __name__ == '__main__':
    unittest.main()
