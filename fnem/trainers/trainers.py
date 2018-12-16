import torch
import logging


LOGGER = logging.getLogger(__name__)


class HistoricalTrainer(object):
    '''data source is a file to loop over'''

    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        LOGGER.info('Device = {}'.format(self.device))

    def build_or_restore_model_and_optimizer(self):
        pass

    def train_model_with_target_replay(self):
        pass

    def save_performance_plots(self):
        pass


class LiveTrainer(object):
    '''data source is a class we interrogate for steps and values'''
    pass
