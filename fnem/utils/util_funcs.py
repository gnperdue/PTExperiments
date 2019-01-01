import logging
import os
from trainers.trainers import HistoricalTrainer, LiveTrainer
import policies.rule_based as rule_based
from utils.common_defs import DEFAULT_COMMANDS

LOGGER = logging.getLogger(__name__)


def create_default_arguments_dict(policy, mode):
    if policy == 'SimpleRuleBased':
        d = {}
        d['start'] = 0.0
        d['amplitude'] = 10.0
        d['period'] = 2.0
        d['commands_array'] = DEFAULT_COMMANDS
        return d
    return None


def create_policy(policy, arguments_dict):
    if policy == 'SimpleRuleBased':
        start = arguments_dict.get('start', 0.0)
        amplitude = arguments_dict.get('amplitude', 10.0)
        period = arguments_dict.get('period', 2.0)
        commands_array = arguments_dict.get('commands_array', DEFAULT_COMMANDS)
        policy_class = rule_based.SimpleRuleBased(
            time=start, amplitude=amplitude, period=period,
            commands_array=commands_array
        )
        return policy_class
    else:
        raise ValueError('Unknown policy ({}).'.format(policy))


def create_trainer(data_source, policy, mode, num_epochs, num_steps,
                   sequence_size=20, replay_buffer_size=100):
    '''
    * data_source: either a file (for historical training) or a
    simulation engine (for live training)
    * policy: what learning algorithm is deployed (may be a static learner)
    * mode: set whether we are running based on a historical policy (no
    learning updates applied), training based on historical data, or
    training based on "live" data
    '''
    arguments_dict = {}
    arguments_dict['num_epochs'] = num_epochs
    arguments_dict['num_steps'] = num_steps
    arguments_dict['sequence_size'] = sequence_size
    arguments_dict['replay_buffer_size'] = replay_buffer_size
    if 'HISTORICAL' in mode:
        trainer = HistoricalTrainer(policy, data_source, arguments_dict)
    elif 'LIVE' in mode:
        trainer = LiveTrainer(policy, data_source, arguments_dict)
    else:
        raise ValueError('Unknown mode ({}).'.format(mode))
    return trainer


def create_data_source(mode, source_path=None, source_file=None):
    data_source = None
    if 'HISTORICAL' in mode:
        if source_path is not None and source_file is not None:
            # TODO - need to make a proper data loader and wrap it to unpack,
            # etc.
            return os.path.join(source_path, source_file)
        else:
            raise ValueError('Source paths required for historical training.')
    elif 'LIVE' in mode:
        pass
    else:
        raise ValueError('Unknown mode ({}).'.format(mode))
    return data_source


def get_logging_level(log_level):
    log_level = log_level.upper()
    logging_level = logging.INFO
    if log_level == 'DEBUG':
        logging_level = logging.DEBUG
    elif log_level == 'INFO':
        logging_level = logging.INFO
    elif log_level == 'WARNING':
        logging_level = logging.WARNING
    elif log_level == 'ERROR':
        logging_level = logging.ERROR
    elif log_level == 'CRITICAL':
        logging_level = logging.CRITICAL
    else:
        print('Unknown or unset logging level. Using INFO')

    return logging_level


def count_parameters(model):
    '''https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
