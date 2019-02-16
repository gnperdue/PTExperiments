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

        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def compute_qvalues(self, observation, use_target=False):
        '''
        return an array of q-values for each action in the commands_array.
        -> no target network, so `use_target` is ignored.
        '''
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

    def train(self, X_train, y_train):
        '''
        train should:
        * compute loss
        * zero gradiaents
        * call `backward()`
        * call `optimizer.step()`
        '''
        loss = self.loss_fn(X_train, y_train)
        return loss.item()

    def anneal_epsilon(self, step):
        pass

    def save_model(self, epoch, step):
        pass
