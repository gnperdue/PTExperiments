import logging
import time

from trainers.qtrainers import LiveQTrainer
from datasources.live import LiveData
import qlearners.simple_mlp as simple_mlp
import qlearners.simple_rulebased as simple_rulebased
import qlearners.simple_random as simple_random
from utils.common_defs import DEFAULT_COMMANDS
from utils.common_defs import DATASOURCE_LIVE_LOG_TEMPLATE

LOGGER = logging.getLogger(__name__)


def create_default_learner_arguments_dict(learner, mode):
    d = {}
    d['commands_array'] = DEFAULT_COMMANDS
    if learner == 'SimpleRuleBased' or learner == 'SimpleRandom':
        return d
    elif learner == 'SimpleMLP':
        d['learning_rate'] = 1e-4
        d['min_epsilon'] = 0.05
        d['gamma'] = 0.99
    else:
        raise ValueError('Unknown learner: ({}).'.format(learner))
    return d


def create_learner(learner, arguments_dict):
    learner_class = None
    if learner == 'SimpleRuleBased':
        learner_class = simple_rulebased.SimpleRuleBased(
            train_pars_dict=arguments_dict
        )
    elif learner == 'SimpleRandom':
        learner_class = simple_random.SimpleRandom(
            train_pars_dict=arguments_dict
        )
    elif learner == 'SimpleMLP':
        learner_class = simple_mlp.SimpleMLP(train_pars_dict=arguments_dict)
    else:
        raise ValueError('Unknown learner ({}).'.format(learner))
    return learner_class


def create_data_source(
    mode, source_path=None, maxsteps=None, run_time=None
):
    # TODO - need to pass in starting setting
    log_time = run_time or int(time.time())
    data_source = None
    if 'HISTORICAL' in mode:
        raise ValueError('Not ready for mode ({}).'.format(mode))
        # if source_path is not None:
        #     data_source = HistoricalData(setting=10.0,
        #                                  source_file=source_path)
        # else:
        #     raise ValueError('Sources required for historical training.')
    elif 'LIVE' in mode:
        logname = './' + DATASOURCE_LIVE_LOG_TEMPLATE % log_time
        # TODO - pass in the setting also
        data_source = LiveData(logname=logname)
    else:
        raise ValueError('Unknown mode ({}).'.format(mode))
    return data_source


def create_trainer(data_source, learner_instance, mode, num_epochs, num_steps,
                   show_progress=False):
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
    arguments_dict['show_progress'] = show_progress
    if 'HISTORICAL' in mode:
        raise ValueError('Not ready for mode ({}).'.format(mode))
        # trainer = HistoricalTrainer(policy, data_source, arguments_dict)
    elif 'LIVE' in mode:
        trainer = LiveQTrainer(learner_instance, data_source, arguments_dict)
    else:
        raise ValueError('Unknown mode ({}).'.format(mode))
    return trainer


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
