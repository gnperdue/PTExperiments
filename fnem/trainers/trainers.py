import logging
import time
from collections import deque

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.common_defs import DATASET_HISTORY_PLT_TEMPLATE
from utils.common_defs import DATASET_HISTORY_PLT_LOSS_TEMPLATE
from utils.common_defs import DATASET_MACHINE_PLT_TEMPLATE
from utils.common_defs import DATASET_MACHINE_PLT_LOSS_TEMPLATE


LOGGER = logging.getLogger(__name__)


class Trainer(object):
    '''base class - defines the API'''

    def __init__(self, policy, data_source, performance_memory_maxlen=5000):
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
        self.losses = deque([], maxlen=self.performance_memory_maxlen)
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
        self.figname = (DATASET_HISTORY_PLT_TEMPLATE % self.tstamp) + '.pdf'
        self.loss_figname = (DATASET_HISTORY_PLT_LOSS_TEMPLATE % self.tstamp) \
            + '.pdf'
        self.num_epochs = arguments_dict['num_epochs']
        self.num_steps = arguments_dict['num_steps']
        self.sequence_size = arguments_dict['sequence_size']

    def train_or_run_model(self, train):
        # loop over epochs, where an epoch is 1 pass over the historical data.
        # here, loss is defined as minimizing the difference between the
        # historical setting and the predicted setting.
        for ep in range(self.num_epochs):
            sequence_buffer = []
            heats_buffer = []
            for i, data in enumerate(self.data_source):
                setting = self.data_source.get_setting()
                historical_state = data[0] # true sensor vals in data[2]
                sensor_vals = list(historical_state[0:4].numpy())
                target_setting = historical_state[4].item()
                t = historical_state[5].item()
                heat = data[1]
                for i, m in enumerate([self.m1, self.m2, self.m3, self.m4]):
                    m.append(sensor_vals[i])
                self.totals.append(sum(sensor_vals))
                self.heats.append(heat.item())
                self.settings.append(setting)
                self.setting_diffs.append(target_setting - setting)
                self.ts.append(t)
                state = data[0] # historical_state
                if len(sequence_buffer) < self.sequence_size:
                    sequence_buffer.append(state)
                    heats_buffer.append(heat)
                else:
                    sequence_buffer.pop(0)
                    sequence_buffer.append(state)
                    heats_buffer.pop(0)
                    heats_buffer.append(heat)
                    self.policy.set_state(sequence_buffer, heats_buffer)
                    if train:
                        loss = self.policy.train()
                        self.losses.append(loss)
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

        fig = plt.Figure(figsize=(10, 6))
        gs = plt.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0])
        ll = len(self.losses)
        ax1.scatter(list(self.ts)[-ll:], self.losses, c='r')

        fig.tight_layout()
        plt.savefig(self.loss_figname, bbox_inches='tight')


class LiveTrainer(Trainer):
    '''data source is a sim machine we interrogate for steps and values'''

    def __init__(self, policy, data_source, arguments_dict):
        super(LiveTrainer, self).__init__(policy=policy,
                                          data_source=data_source)
        self.figname = (DATASET_MACHINE_PLT_TEMPLATE % self.tstamp) + '.pdf'
        self.loss_figname = (DATASET_MACHINE_PLT_LOSS_TEMPLATE % self.tstamp) \
            + '.pdf'
        self.num_epochs = None
        self.num_steps = arguments_dict['num_steps']
        self.sequence_size = arguments_dict['sequence_size']

    def train_or_run_model(self, train):
        # no concept of epochs with live data, run over machine steps as long
        # as they are available or until we reach a max step value.
        # here, loss is defined as minimizing the heat from the machine.
        sequence_buffer = []
        heats_buffer = []
        for i, data in enumerate(self.data_source):
            sensor_vals = data[0][0:4].numpy()
            setting = data[0][4].numpy()
            t = data[0][5].numpy()
            heat = data[1]
            for i, m in enumerate([self.m1, self.m2, self.m3, self.m4]):
                m.append(sensor_vals[i])
            self.totals.append(sum(sensor_vals))
            self.heats.append(heat.item())
            self.settings.append(setting)
            self.ts.append(t)
            state = data[0]
            if len(sequence_buffer) < self.sequence_size:
                sequence_buffer.append(state)
                heats_buffer.append(heat)
            else:
                sequence_buffer.pop(0)
                sequence_buffer.append(state)
                heats_buffer.pop(0)
                heats_buffer.append(heat)
                self.policy.set_state(sequence_buffer, heats_buffer)
                if train:
                    loss = self.policy.train()
                    self.losses.append(loss)
                command_idx = self.policy.compute_action()
                self.data_source.update_setting(command_idx)

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

        fig = plt.Figure(figsize=(10, 6))
        gs = plt.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0])
        ll = len(self.losses)
        ax1.scatter(list(self.ts)[-ll:], self.losses, c='r')

        fig.tight_layout()
        plt.savefig(self.loss_figname, bbox_inches='tight')
