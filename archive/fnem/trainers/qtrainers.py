import logging
import time
from collections import deque


LOGGER = logging.getLogger(__name__)


class QTrainer(object):
    '''base class - defines the API'''

    def __init__(self, qlearner, data_source, performance_memory_maxlen=5000):
        self.data_source = data_source
        self.tstamp = int(time.time())
        self.qlearner = qlearner
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
        self.qlearner.build_or_restore_model_and_optimizer()

    def train_or_run_model(self, train):
        raise NotImplementedError

    def save_performance_plots(self):
        raise NotImplementedError


class LiveQTrainer(QTrainer):
    '''data source is a sim machine we interrogate for steps and values'''

    def __init__(self, qlearner, data_source, arguments_dict):
        super(LiveQTrainer, self).__init__(qlearner=qlearner,
                                           data_source=data_source)
        # self.figname = (DATASET_MACHINE_PLT_TEMPLATE % self.tstamp) + '.pdf'
        # self.loss_figname = (DATASET_MACHINE_PLT_LOSS_TEMPLATE % self.tstamp) \
        #     + '.pdf'
        self.num_epochs = None
        self.num_steps = arguments_dict['num_steps']
        self.sequence_size = arguments_dict['sequence_size']

    def train_or_run_model(self, train):
        # train or run model only on full buffers
        sequence_buffer = []
        heats_buffer = []
        # no concept of epochs with live data, run over machine steps as long
        # as they are available or until we reach a max step value.
        #
        # TODO - need to rethink the datasource and not enumerate over it,
        # and instead step manually?
        for i, data in enumerate(self.data_source):
            state = data[0]
            heat = data[1]
            if len(sequence_buffer) < self.sequence_size:
                sequence_buffer.append(state)
                heats_buffer.append(heat)
            else:
                if len(sequence_buffer) > self.sequence_size:
                    sequence_buffer.pop(0)
                    heats_buffer.pop(0)
                sequence_buffer.append(state)
                heats_buffer.append(heat)
                qval = self.qlearner.compute_qvalues(sequence_buffer)
                action, action_ = self.qlearner.compute_action(qval)
                # TODO - data_source should return a new heat when the setting
                # is updated.
                reward = self.data_source.update_setting(action)
                new_state = self.data_source.get_state()
                if train:
                    loss = self.qlearner.train(
                        state, action_, reward, new_state
                    )
                    self.losses.append(loss)
                # self.policy.set_state(sequence_buffer, heats_buffer)
                # if train:
                #     loss = self.policy.train()
                #     self.losses.append(loss)
                # command_idx = self.policy.compute_action()
                # self.data_source.update_setting(command_idx)

        self.data_source.close_dataset_logger()

    def save_performance_plots(self):
        pass
