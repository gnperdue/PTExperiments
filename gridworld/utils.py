import logging

LOGGER = logging.getLogger(__name__)


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
