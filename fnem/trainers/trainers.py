import torch
import logging
from collections import deque


LOGGER = logging.getLogger(__name__)


class Trainer(object):
    '''base class - defines the API'''

    def __init__(self, policy, performance_memory_maxlen=5000):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        LOGGER.info('Device = {}'.format(self.device))
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
        self.heat = deque([], maxlen=self.performance_memory_maxlen)
        self.settings = deque([], maxlen=self.performance_memory_maxlen)

    def build_or_restore_model_and_optimizer(self):
        self.policy.build_or_restore_model_and_optimizer()

    def train_model_with_target_replay(self):
        raise NotImplementedError

    def run_model(self):
        raise NotImplementedError

    def save_performance_plots(self):
        raise NotImplementedError


class HistoricalTrainer(Trainer):
    '''data source is a file to loop over'''

    def __init__(self, policy, training_file, arguments_dict):
        super(HistoricalTrainer, self).__init__(policy=policy)
        self.training_data_file = training_file
        self.num_epochs = arguments_dict['num_epochs']
        self.num_steps = arguments_dict['num_steps']
        self.batch_size = arguments_dict['batch_size']
        self.replay_buffer_size = arguments_dict['replay_buffer_size']

    def train_model_with_target_replay(self):
        # loop over epochs, where an epoch is one pass over the historical data
        pass


class LiveTrainer(Trainer):
    '''data source is a sim machine we interrogate for steps and values'''

    def __init__(self, policy, sim_machine, arguments_dict):
        super(LiveTrainer, self).__init__(policy=policy)
        self.machine = sim_machine
        self.num_epochs = None
        self.num_steps = arguments_dict['num_steps']
        self.batch_size = arguments_dict['batch_size']
        self.replay_buffer_size = arguments_dict['replay_buffer_size']

    def train_model_with_target_replay(self):
        # no concept of epochs with live data, run over machine steps as long
        # as they are available or until we reach a max step value

        # TODO - loop over batches...
        # TODO - build up target replay buffer of batches
        batch_buffer, replay_buffer = [], []
        for i in range(self.num_steps):
            if self.machine.step():
                t = self.machine.get_time()
                sensor_vals = self.machine.get_sensor_values()
                self.ts.append(t)
                for i, m in enumerate([self.m1, self.m2, self.m3, self.m4]):
                    m.append(sensor_vals[i])
                self.totals.append(sum(sensor_vals))
                setting = self.machine.get_setting()
                self.settings.append(setting)
                heat = self.machine.get_heat()
                self.heat.append(heat)

                state = sensor_vals + [heat, t]
                batch_buffer.append(state)
                # if len(replay_buffer) < self.replay_buffer_size:
                #     replay_buffer.append(state)
                # else:
                #     replay_buffer.pop(0)
                #     replay_buffer.append(state)
                #     # TODO - build trainbatch, pass the batch to the policy
                #     # X_train, y_train = self._build_trainbatch(replay_buffer)
                #     # self.policy.set_state_batch(X_train, y_train)
                #     ## compute action should:
                #     ## * zero gradiaents
                #     ## * compute loss, add loss to a monitoring log
                #     ## * call `backward()`
                #     ## * call `optimizer.step()`
                #     # command = self.policy.compute_action()
                self.policy.set_state(state)
                command = self.policy.compute_action()
                self.machine.update_machine(command)
                self.policy.update_setting(command)

        self.machine.close_logger()
