import torch
import logging


LOGGER = logging.getLogger(__name__)


class Trainer(object):
    '''base class - defines the API'''

    def __init__(self, policy):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        LOGGER.info('Device = {}'.format(self.device))
        self.policy = policy
        self.training_sim_machine = None
        self.training_data_file = None

    def build_or_restore_model_and_optimizer(self):
        raise NotImplementedError

    def train_model_with_target_replay(self):
        raise NotImplementedError

    def save_performance_plots(self):
        raise NotImplementedError


class HistoricalTrainer(Trainer):
    '''data source is a file to loop over'''

    def __init__(self, policy, training_file):
        super(HistoricalTrainer, self).__init__(policy=policy)
        self.training_data_file = training_file

    def build_or_restore_model_and_optimizer(self):
        pass

    def train_model_with_target_replay(self):
        pass

    def save_performance_plots(self):
        pass


class LiveTrainer(Trainer):
    '''data source is a sim machine we interrogate for steps and values'''

    def __init__(self, policy, sim_machine):
        super(LiveTrainer, self).__init__(policy=policy)
        self.training_sim_machine = sim_machine
