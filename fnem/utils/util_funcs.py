import logging
import time

from trainers.trainers import HistoricalTrainer, LiveTrainer
from datasources.live import LiveData
from datasources.historical import HistoricalData
import policies.rule_based as rule_based
import policies.simple_mlp as simple_mlp
from utils.common_defs import DEFAULT_COMMANDS
from utils.common_defs import DATASET_MACHINE_LOG_TEMPLATE

LOGGER = logging.getLogger(__name__)


def create_default_arguments_dict(policy, mode):
    d = {}
    if policy == 'SimpleRuleBased':
        d['start'] = 0.0
        d['amplitude'] = 10.0
        d['period'] = 2.0
        d['commands_array'] = DEFAULT_COMMANDS
    elif policy == 'SimpleMLP':
        d['commands_array'] = DEFAULT_COMMANDS
        d['learning_rate'] = 1e-4
        d['ckpt_path'] = '/tmp/simple_mlp/ckpt.tar'
    else:
        raise ValueError('Unknown policy ({}).'.format(policy))
    return d


def create_policy(policy, arguments_dict):
    policy_class = None
    if policy == 'SimpleRuleBased':
        start = arguments_dict.get('start', 0.0)
        amplitude = arguments_dict.get('amplitude', 10.0)
        period = arguments_dict.get('period', 2.0)
        commands_array = arguments_dict.get('commands_array', DEFAULT_COMMANDS)
        policy_class = rule_based.SimpleRuleBased(
            time=start, amplitude=amplitude, period=period,
            commands_array=commands_array
        )
    elif policy == 'SimpleMLP':
        learning_rate = arguments_dict.get('learning_rate', 1e-4)
        ckpt_path = arguments_dict.get('ckpt_path',
                                        '/tmp/simple_mlp/ckpt.tar')
        commands_array = arguments_dict.get('commands_array', DEFAULT_COMMANDS)
        policy_class = simple_mlp.SimpleMLP(
            commands_array=commands_array, learning_rate=learning_rate,
            ckpt_path=ckpt_path
        )
    else:
        raise ValueError('Unknown policy ({}).'.format(policy))
    return policy_class


def create_trainer(data_source, policy, mode, num_epochs, num_steps,
                   sequence_size=20):
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
    if 'HISTORICAL' in mode:
        trainer = HistoricalTrainer(policy, data_source, arguments_dict)
    elif 'LIVE' in mode:
        trainer = LiveTrainer(policy, data_source, arguments_dict)
    else:
        raise ValueError('Unknown mode ({}).'.format(mode))
    return trainer


def create_data_source(
    mode, source_path=None, maxsteps=None, run_time=None
):
    # TODO - need to pass in starting setting
    log_time = run_time or time.time()
    data_source = None
    if 'HISTORICAL' in mode:
        if source_path is not None:
            data_source = HistoricalData(setting=10.0, source_file=source_path)
        else:
            raise ValueError('Source paths required for historical training.')
    elif 'LIVE' in mode:
        logname = './' + DATASET_MACHINE_LOG_TEMPLATE % log_time
        data_source = LiveData(maxsteps=maxsteps, logname=logname)
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
