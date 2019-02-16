import logging

import numpy as np
import torch

from .qbase import BaseQ
import utils.util_funcs as utils


LOGGER = logging.getLogger(__name__)


class SimpleMLP(BaseQ):

    def __init__(self, train_pars_dict):
        super(SimpleMLP, self).__init__(train_pars_dict['commands_array'])
        self.pytorch = True
        self.epsilon = 0.99
        self.start_step = 0
        self.start_epoch = 0
        self._min_epsilon = train_pars_dict.get('min_epsilon', 0.05)
        self.gamma = train_pars_dict.get('gamma', 0.99)

        l1 = 80  # observation length = 4 * 20 timesteps
        l2 = 150
        l3 = self.noutputs
        self.model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.LeakyReLU()
            # torch.nn.Softmax(dim=0)
        )
        self._ckpt_path = train_pars_dict.get('ckpt_path',
                                              '/tmp/simple_mlp/ckpt.tar')
        self._learning_rate = train_pars_dict.get('learning_rate', 0.0009)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self._learning_rate)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def compute_qvalues(self, observation):
        '''return an array of q-values for each action in the commands_array'''
        return self.model(observation)

    def compute_action(self, qvalues):
        '''compute the action index'''
        if np.random.rand() < self.epsilon:
            action_ = np.random.randint(0, self.noutputs)
        else:
            qvalues_ = qvalues.cpu().data.numpy()
            action_ = np.argmax(qvalues_)
        return action_

    def train(self, X_train, y_train):
        loss = self.loss_fn(X_train, y_train)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def restore_model_and_optimizer(self):
        '''
        '''
        # TODO - need to track epsilon in the checkpoint
        LOGGER.info('model has {} parameters'.format(
            utils.count_parameters(self.model)
        ))
        try:
            checkpoint = torch.load(self._ckpt_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.start_step = checkpoint['step'] + 1
            self.epsilon = checkpoint['epsilon']
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

    def anneal_epsilon(self, step):
        '''
        x = np.arange(10000) + 1
        np.sum(1 / x / 10.)
        -> 0.9787606036044383
        '''
        if self.epsilon > self._min_epsilon:
            update = max(1. / ((step + 1)) / 10.0, 0.0000001)
            self.epsilon -= update

    def save_model(self, epoch, step):
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, self._ckpt_path)
