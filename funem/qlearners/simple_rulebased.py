import logging

import numpy as np
import torch

from .qbase import BaseQ


LOGGER = logging.getLogger(__name__)


class SimpleRuleBased(BaseQ):

    def __init__(self, train_pars_dict):
        super(SimpleRuleBased, self).__init__(
            train_pars_dict['commands_array']
        )
        self.pytorch = True
        self.start_step = 0
        self.start_epoch = 0

    def compute_qvalues(self, observation):
        '''return an array of q-values for each action in the commands_array'''
        observation_ = observation.cpu().data.numpy()
        ob1 = observation_[-4]
        ob2 = observation_[-8]
        delta = ob1 - ob2
        diffs = np.abs(self._commands - delta)
        qvalues = torch.zeros(len(self._commands))
        qvalues[np.argmin(diffs)] = 1.0
        return qvalues

    def compute_action(self, qvalues):
        '''compute the action index'''
        qvalues_ = qvalues.cpu().data.numpy()
        action_ = np.argmax(qvalues_)
        return action_

    def build_trainbatch(self, replay_buffer):
        '''build a training batch and targets from the replay buffer'''
        return None, None

    def train(self, X_train, y_train):
        '''
        train should:
        * compute loss
        * zero gradiaents
        * call `backward()`
        * call `optimizer.step()`
        '''
        return -1.0

    def restore_model_and_optimizer(self):
        pass

    def anneal_epsilon(self, step):
        pass

    def save_model(self, epoch, step):
        pass
