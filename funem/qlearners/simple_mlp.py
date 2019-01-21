import logging
import random

import numpy as np
import torch
from torch.autograd import Variable

from .qbase import BaseQ
import utils.util_funcs as utils
from utils.common_defs import MAX_HEAT


LOGGER = logging.getLogger(__name__)


class SimpleMLP(BaseQ):

    def __init__(self, train_pars_dict):
        super(SimpleMLP, self).__init__(train_pars_dict['commands_array'])
        self.pytorch = True
        self.epsilon = 0.99
        self._min_epsilon = train_pars_dict.get('min_epsilon', 0.05)
        self.gamma = train_pars_dict.get('gamma', 0.99)

        l1 = 4
        l2 = 150
        l3 = self.noutputs
        self.model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.Softmax(dim=0)
        )
        self._ckpt_path = train_pars_dict.get('ckpt_path',
                                              '/tmp/simple_mlp/ckpt.tar')
        self._learning_rate = train_pars_dict.get('learning_rate', 0.0009)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self._learning_rate)
        self.loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')

    def compute_qvalues(self, observation):
        '''return an array of q-values for each action in the commands_array'''
        return self.model(observation)

    def compute_action(self, qvalues):
        '''compute the action index'''
        qvalues_ = qvalues.cpu().data.numpy()
        if np.random.rand() < self.epsilon:
            action_ = np.random.randint(0, self.noutputs)
        else:
            action_ = np.argmax(qvalues_)
        return action_

    def build_trainbatch(self, replay_buffer):
        '''build a training batch and targets from the replay buffer'''
        X_train, y_train = None, None
        minibatch = random.sample(replay_buffer, self._batch_size)
        X_train = Variable(torch.empty(
            self._batch_size, self.noutputs, dtype=torch.float
        ))
        y_train = Variable(torch.empty(
            self._batch_size, self.noutputs, dtype=torch.float
        ))
        # Fill X_train and y_train minibatch tensors by index `h` by
        # looping through memory and computing the Q-values before (X)
        # and after (y) each move
        h = 0
        for memory in minibatch:
            # reward is heat, and we want head to be small, but here we need
            # a large "reward", so use max_heat - heat instead of the raw heat
            # value.
            old_state, action_m, reward_m, new_state_m = memory
            old_qval = self.model(old_state)
            # TODO - use a target model...
            # new_qval = self.target_model(new_state_m).cpu().data.numpy()
            new_qval = self.model(new_state_m).cpu().data.numpy()
            max_qval = np.max(new_qval)
            y = torch.zeros((1, self.noutputs))
            y[:] = old_qval[:]
            update = (MAX_HEAT - reward_m + (self.gamma * max_qval))
            y[0][action_m] = update
            X_train[h] = old_qval
            y_train[h] = Variable(y)
            h += 1
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        return X_train, y_train

    def train(self, X_train, y_train):
        loss = self.loss_fn(X_train, y_train)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def build_or_restore_model_and_optimizer(self):
        '''
        '''
        # TODO - need to track epsilon in the checkpoint
        LOGGER.info('model has {} parameters'.format(
            utils.count_parameters(self.model)
        ))
        try:
            checkpoint = torch.load(self._ckpt_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            LOGGER.info('Loaded checkpoint from {}'.format(self._ckpt_path))
        except FileNotFoundError:
            LOGGER.info('No checkpoint found...')

        LOGGER.debug('Model state dict:')
        for param_tensor in self.model.state_dict():
            LOGGER.debug(str(param_tensor) + '\t'
                         + str(self.model.state_dict()[param_tensor].size()))
        LOGGER.debug('Optimizer state dict:')
        for var_name in self.optimizer.state_dict():
            LOGGER.debug(str(var_name) + '\t'
                         + str(self.optimizer.state_dict()[var_name]))

        self.model.to(self.device)

        # TODO - get the start step out of the checkpoint and return it
        # TODO - return other stuff also to the learning loop controller?
        return 0

    def anneal_epsilon(self, step):
        if self.epsilon > self._min_epsilon:
            self.epsilon -= (1. / (step / 100.0))
