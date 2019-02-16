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
            self.learner.train(None, None)
        with self.assertRaises(NotImplementedError):
            self.learner.anneal_epsilon(None)
        with self.assertRaises(NotImplementedError):
            self.learner.save_model(None, None)


class TestSimpleMLP(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.live_data = live.LiveData(setting=10.0, logname=TEST_LOG)

        d = {}
        d['commands_array'] = DEFAULT_COMMANDS
        self.learner = simple_mlp.SimpleMLP(train_pars_dict=d)
        self.learner.restore_model_and_optimizer()

    def tearDown(self):
        if os.path.isfile(TEST_LOG_GZ):
            os.remove(TEST_LOG_GZ)

    def test_compute_qvalues_and_action(self):
        it = iter(self.live_data)
        obs, setting, t, heat = next(it)
        qvals = self.learner.compute_qvalues(obs)
        self.assertEqual(qvals.shape, torch.Size([9]))
        self.learner.epsilon = 1.0
        action = self.learner.compute_action(qvals)
        self.assertLess(action, qvals.shape[0])
        self.learner.epsilon = 0.0
        action = self.learner.compute_action(qvals)
        self.assertLess(action, qvals.shape[0])

    def test_make_trainbatch(self):
        replay_buffer = []
        data_iter = iter(self.live_data)
        action_ = 5
        observation, _, _, heat = next(data_iter)
        for step in range(30):
            new_observation, setting, time, heat = next(data_iter)
            replay_buffer.append(
                (observation, action_, heat.item(), new_observation)
            )
            observation = new_observation
        X_train, y_train = self.learner.build_trainbatch(replay_buffer)
        self.assertEqual(X_train.shape, torch.Size([20, 9]))
        self.assertEqual(y_train.shape, torch.Size([20, 9]))

    def test_train(self):
        x = torch.randn(20, 80)
        y = torch.randn(20, 9)
        y_pred = self.learner.model(x)
        loss_val1 = self.learner.train(y_pred, y)
        y_pred = self.learner.model(x)
        loss_val2 = self.learner.train(y_pred, y)
        self.assertLess(loss_val2, loss_val1)

    def test_restore_model_and_optimizer_do_nothing(self):
        '''just, don't crash...'''
        self.learner.restore_model_and_optimizer()

    def test_restore_model_and_optimizer_load_model(self):
        '''just, don't crash...'''
        self.learner._ckpt_path = './reference_files/ckpt_f155a263ea.tar'
        self.learner.restore_model_and_optimizer()

    def test_anneal_epsilon(self):
        self.assertEqual(self.learner.epsilon, 0.99)
        self.learner.anneal_epsilon(1)
        self.assertEqual(self.learner.epsilon, 0.94)
        self.learner.anneal_epsilon(1e9)
        self.assertEqual(self.learner.epsilon, 0.9399999)

    def test_save_model(self):
        self.learner._ckpt_path = './temp_temp.tar'
        self.learner.save_model(314, 278)
        self.learner.restore_model_and_optimizer()
        self.assertEqual(self.learner.start_epoch, 315)
        self.assertEqual(self.learner.start_step, 279)
        os.remove('./temp_temp.tar')


class TestSimpleRuleBased(unittest.TestCase):

    def setUp(self):
        d = {}
        d['commands_array'] = DEFAULT_COMMANDS
        self.learner = simple_rulebased.SimpleRuleBased(train_pars_dict=d)

    def tearDown(self):
        pass

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
        self.fail('Finish the test...')

    def test_do_nothings(self):
        self.learner.restore_model_and_optimizer()
        self.learner.anneal_epsilon(None)
        self.learner.save_model(None, None)


if __name__ == '__main__':
    unittest.main()
