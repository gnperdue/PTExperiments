import torch
import logging
from collections import deque


LOGGER = logging.getLogger(__name__)


def make_perf_memories(n, maxlen):
    ll = []
    for _ in range(n):
        d = deque([], maxlen=maxlen)
        ll.append(d)
    return tuple(ll)


class Trainer(object):
    '''base class - defines the API'''

    def __init__(self, policy):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        LOGGER.info('Device = {}'.format(self.device))
        self.policy = policy
        self.machine = None
        self.training_data_file = None
        self.performance_memory_maxlen = 5000

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
        # self.m1, m2, m3, m4, ts, totals, heat, settings = make_perf_memories(
        #     8, self.performance_memory_maxlen
        # )

    def train_model_with_target_replay(self):
        # no concept of epochs with live data, run over machine steps as long
        # as they are available or until we reach a max step value?

        # TODO - perf counters should be part of self
        m1, m2, m3, m4, ts, totals, heat, settings = make_perf_memories(
            8, self.performance_memory_maxlen
        )
        for i in range(self.num_steps):
            self.machine.step()
            t = self.machine.get_time()
            sensor_vals = self.machine.get_sensor_values()
            ts.append(t)
            for i, m in enumerate([m1, m2, m3, m4]):
                m.append(sensor_vals[i])
            totals.append(sum(sensor_vals))
            settings.append(self.machine.get_setting())
            heat.append(self.machine.get_heat())
            state = sensor_vals + [t]
            self.policy.set_state(state)
            command = self.policy.compute_action()
            self.machine.update_machine(command)
            self.policy.update_setting(command)

        self.machine.close_logger()
