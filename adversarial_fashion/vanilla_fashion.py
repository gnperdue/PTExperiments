import argparse
import logging
import time
import torch

from ptlib.dataloaders import FashionDataManager as DataManager
from ptlib.models import SimpleConvNet as Model
from ptlib.utils import get_logging_level
from ptlib.utils import log_function_args


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--ckpt-path', default='ckpt.tar', type=str,
                    help='checkpoint path')
parser.add_argument('--data-dir', default='', type=str, help='data dir')
parser.add_argument('--log-level', default='INFO', type=str,
                    help='log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)')
parser.add_argument('--num-epochs', default=100, type=int,
                    help='number of epochs')
parser.add_argument('--show-progress', default=False, action='store_true',
                    help='print tdqm and other output')


def main(
    batch_size, ckpt_path, data_dir, log_level, num_epochs, show_progress
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
    log_function_args(vars())

    # set up a data manager
    data_manager = DataManager(data_dir=data_dir)

    # set up a model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model()
    model.to(device)

    # create a trainer with data handler and model
    # run training


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
