import argparse
import logging
import time

from ptlib.utils import get_logging_level


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--ckpt-path', default='ckpt.tar', type=str,
                    help='checkpoint path')
parser.add_argument('--log-level', default='INFO', type=str,
                    help='log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)')
parser.add_argument('--num-epochs', default=100, type=int,
                    help='number of epochs')
parser.add_argument('--show-progress', default=False, action='store_true',
                    help='print tdqm and other output')


def main(
    batch_size, ckpt_path, log_level, num_epochs, show_progress
):
    logfilename = 'log_' + __file__.split('/')[-1].split('.')[0] \
        + str(int(time.time())) + '.txt'
    logging.basicConfig(
        filename=logfilename, level=get_logging_level(log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Starting...")
    LOGGER.info(__file__)

    # set up a data handler
    # set up a model
    # create a trainer with data handler and model
    # run training


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
