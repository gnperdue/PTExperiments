import logging
import policies.rule_based as rule_based

LOGGER = logging.getLogger(__name__)


def create_policy(policy, arguments_dict):
    if policy == 'SimpleRuleBased':
        # time=start, setting=10.0, amplitude=amplitude, period=period,
        # commands_array=machine.get_commands()
        policy_class = rule_based.SimpleRuleBased()
        return policy_class
    else:
        raise ValueError('Unknown policy ({}).'.format(policy))


def create_trainer(data_source, policy, mode):
    if mode == 'RUN-TRAINED':
        pass
    elif mode == 'TRAIN-HISTORICAL':
        pass
    elif mode == 'TRAIN-LIVE':
        pass
    else:
        raise ValueError('Unknown mode ({}).'.format(mode))


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
