import argparse
import logging
import time

from gridrl.models import build_basic_model
from gridrl.models import build_conv_model
from gridrl.trainers import RLTrainer as Trainer
from gridrl.utils import get_logging_level

import warnings
# 'error' to stop on warns, 'ignore' to ignore silly matplotlib noise
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=100, type=int, help='batch size')
parser.add_argument('--buffer', default=500, type=int, help='replay buffer')
parser.add_argument('--ckpt-path', default='ckpt.tar', type=str,
                    help='checkpoint path')
parser.add_argument('--conv', default=False, action='store_true',
                    help='use a convolutional net')
parser.add_argument('--gamma', default=0.95, type=float, help='discount')
parser.add_argument('--game-mode', default='random', type=str,
                    help='initial board configuration')
parser.add_argument('--game-size', default=4, type=int, help='game size')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--log-level', default='INFO', type=str,
                    help='log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)')
parser.add_argument('--make-plot', default=False, action='store_true',
                    help='plot win percentages and losses')
parser.add_argument('--num-epochs', default=100, type=int,
                    help='number of epochs')
parser.add_argument('--saved-losses-path', default='losses.npy', type=str,
                    help='saved losses location')
parser.add_argument('--saved-winpct-path', default='winpct.npy', type=str,
                    help='saved win percentages location')
parser.add_argument('--show-progress', default=False, action='store_true',
                    help='print tdqm and other output')
parser.add_argument('--target-network-update', default=500, type=int,
                    help='target network update period')


def main(
    batch_size, buffer, ckpt_path, conv, gamma, game_mode, game_size,
    learning_rate, log_level, make_plot, num_epochs, saved_losses_path,
    saved_winpct_path, show_progress, target_network_update
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

    # * create and configure a RLTrainer
    game_parameters = {}
    game_parameters['mode'] = game_mode
    game_parameters['size'] = game_size
    train_parameters = {}
    train_parameters['batch_size'] = batch_size
    train_parameters['buffer'] = buffer
    train_parameters['ckpt_path'] = ckpt_path
    train_parameters['gamma'] = gamma
    train_parameters['learning_rate'] = learning_rate
    train_parameters['saved_losses_path'] = saved_losses_path
    train_parameters['saved_winpct_path'] = saved_winpct_path
    train_parameters['target_network_update'] = target_network_update
    train_parameters['show_progress'] = show_progress
    trainer = Trainer(game_parameters, train_parameters)

    # * train and validate
    if conv:
        trainer.build_or_restore_model_and_optimizer(build_conv_model, conv)
    else:
        trainer.build_or_restore_model_and_optimizer(build_basic_model, conv)
    trainer.train_model_with_target_replay(num_epochs)
    trainer.save_losses_and_winpct_plots(make_plot)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
