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

    def save_performance_plots(self):
        pass


class LiveTrainer(Trainer):
    '''data source is a sim machine we interrogate for steps and values'''

    def __init__(self, policy, sim_machine, arguments_dict):
        super(LiveTrainer, self).__init__(policy=policy)
        self.training_sim_machine = sim_machine
        self.num_epochs = None
        self.num_steps = arguments_dict['num_steps']

    def train_model_with_target_replay(self):
        # no concept of epochs with live data, run over machine steps as long
        # as they are available or until we reach a max step value?
        pass
