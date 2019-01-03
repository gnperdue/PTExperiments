import logging
import time
from collections import deque

import torch
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)


class Trainer(object):
    '''base class - defines the API'''

    def __init__(self, policy, data_source, performance_memory_maxlen=5000):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        LOGGER.info('Device = {}'.format(self.device))
        self.data_source = data_source
        self.tstamp = int(time.time())
        self.policy = policy
        self.machine = None
        self.training_data_file = None
        self.performance_memory_maxlen = performance_memory_maxlen
        self.m1 = deque([], maxlen=self.performance_memory_maxlen)
        self.m2 = deque([], maxlen=self.performance_memory_maxlen)
        self.m3 = deque([], maxlen=self.performance_memory_maxlen)
        self.m4 = deque([], maxlen=self.performance_memory_maxlen)
        self.ts = deque([], maxlen=self.performance_memory_maxlen)
        self.totals = deque([], maxlen=self.performance_memory_maxlen)
        self.heats = deque([], maxlen=self.performance_memory_maxlen)
        self.settings = deque([], maxlen=self.performance_memory_maxlen)
        self.setting_diffs = deque([], maxlen=self.performance_memory_maxlen)

    def build_or_restore_model_and_optimizer(self):
        self.policy.build_or_restore_model_and_optimizer()

    def train_or_run_model(self, train):
        raise NotImplementedError

    def save_performance_plots(self):
        raise NotImplementedError


class HistoricalTrainer(Trainer):
    '''data source is a file to loop over'''

    def __init__(self, policy, data_source, arguments_dict):
        super(HistoricalTrainer, self).__init__(policy=policy,
                                                data_source=data_source)
        self.figname = 'historical_trainer_%d.pdf' % self.tstamp
        self.num_epochs = arguments_dict['num_epochs']
        self.num_steps = arguments_dict['num_steps']
        self.sequence_size = arguments_dict['sequence_size']
        self.replay_buffer_size = arguments_dict['replay_buffer_size']

    def train_or_run_model(self, train):
        # loop over epochs, where an epoch is 1 pass over the historical data.
        # here, loss is defined as minimizing the difference between the
        # historical setting and the predicted setting.
        for ep in range(self.num_epochs):
            sequence_buffer = []
            for i, data in enumerate(self.data_source):
                setting = self.data_source.get_setting()
                state = data[0]
                sensor_vals = state[0:4].numpy()
                heat = state[4].numpy()
                target_setting = state[5].numpy()
                t = state[6].numpy()
                for i, m in enumerate([self.m1, self.m2, self.m3, self.m4]):
                    m.append(sensor_vals[i])
                self.totals.append(sum(sensor_vals))
                self.heats.append(heat)
                self.settings.append(setting)
                self.setting_diffs.append(target_setting - setting)
                self.ts.append(t)
                if len(sequence_buffer) < self.sequence_size:
                    sequence_buffer.append(state)
                else:
                    sequence_buffer.pop(0)
                    sequence_buffer.append(state)
                    self.policy.set_state(sequence_buffer)
                    #     ## train should:
                    #     ## * zero gradiaents
                    #     ## * compute loss, add loss to a monitoring log
                    #     ## * call `backward()`
                    #     ## * call `optimizer.step()`
                    if train:
                        self.policy.train()
                    command = self.policy.compute_action()
                    adjustment = self.policy.get_adjustment_value(command)
                    self.data_source.adjust_setting(adjustment)

    def save_performance_plots(self):
        fig = plt.Figure(figsize=(10, 6))
        gs = plt.GridSpec(1, 4)
        ax1 = plt.subplot(gs[0])
        ax1.scatter(self.ts, self.m1, c='r')
        ax1.scatter(self.ts, self.m2, c='g')
        ax1.scatter(self.ts, self.m3, c='b')
        ax1.scatter(self.ts, self.m4, c='y')
        ax2 = plt.subplot(gs[1])
        ax2.scatter(self.ts, self.totals, c='k')
        ax3 = plt.subplot(gs[2])
        ax3.scatter(self.ts, self.settings, c='k')
        ax4 = plt.subplot(gs[3])
        ax4.scatter(self.ts, self.setting_diffs, c='k')

        fig.tight_layout()
        plt.savefig(self.figname, bbox_inches='tight')


class LiveTrainer(Trainer):
    '''data source is a sim machine we interrogate for steps and values'''

    def __init__(self, policy, data_source, arguments_dict):
        super(LiveTrainer, self).__init__(policy=policy,
                                          data_source=data_source)
        self.figname = 'live_trainer_%d.pdf' % self.tstamp
        self.num_epochs = None
        self.num_steps = arguments_dict['num_steps']
        self.sequence_size = arguments_dict['sequence_size']
        self.replay_buffer_size = arguments_dict['replay_buffer_size']

    def train_or_run_model(self, train):
        # no concept of epochs with live data, run over machine steps as long
        # as they are available or until we reach a max step value.
        # here, loss is defined as minimizing the heat from the machine.

        sequence_buffer = []
        for i, state in enumerate(self.data_source):
            sensor_vals = state[0:4].numpy()
            heat = state[4].numpy()
            setting = state[5].numpy()
            t = state[6].numpy()
            for i, m in enumerate([self.m1, self.m2, self.m3, self.m4]):
                m.append(sensor_vals[i])
            self.totals.append(sum(sensor_vals))
            self.heats.append(heat)
            self.settings.append(setting)
            self.ts.append(t)
            if len(sequence_buffer) < self.sequence_size:
                sequence_buffer.append(state)
            else:
                sequence_buffer.pop(0)
                sequence_buffer.append(state)
                self.policy.set_state(sequence_buffer)
                #     ## train should:
                #     ## * zero gradiaents
                #     ## * compute loss, add loss to a monitoring log
                #     ## * call `backward()`
                #     ## * call `optimizer.step()`
                if train:
                    self.policy.train()
                command = self.policy.compute_action()
                self.data_source.update_setting(command)

        self.data_source.close_dataset_logger()

    def save_performance_plots(self):
        fig = plt.Figure(figsize=(10, 6))
        gs = plt.GridSpec(1, 4)
        ax1 = plt.subplot(gs[0])
        ax1.scatter(self.ts, self.m1, c='r')
        ax1.scatter(self.ts, self.m2, c='g')
        ax1.scatter(self.ts, self.m3, c='b')
        ax1.scatter(self.ts, self.m4, c='y')
        ax2 = plt.subplot(gs[1])
        ax2.scatter(self.ts, self.totals, c='k')
        ax3 = plt.subplot(gs[2])
        ax3.scatter(self.ts, self.heats, c='k')
        ax4 = plt.subplot(gs[3])
        ax4.scatter(self.ts, self.settings, c='k')

        fig.tight_layout()
        plt.savefig(self.figname, bbox_inches='tight')
