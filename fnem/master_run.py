'''
Running modes (`--mode`):

* RUN-TRAINED: run a trained policy and generate historical data.
* TRAIN-HISTORICAL: run a policy with live training from historical data.
* TRAIN-LIVE: run a policy with live training using an asynchronous running
  simulation.

Polcies (`--policy`):

* SimpleRuleBased
'''
import argparse
import logging
import time
import sys

from utils.util_funcs import create_default_arguments_dict
from utils.util_funcs import create_policy
from utils.util_funcs import create_trainer
from utils.util_funcs import get_logging_level
from utils.common_defs import RUN_MODES

import warnings
# 'error' to stop on warns, 'ignore' to ignore silly matplotlib noise
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=100, type=int, help='batch size')
parser.add_argument('--ckpt-path', default='ckpt.tar', type=str,
                    help='checkpoint path')
parser.add_argument('--exp-replay-buffer', default=500, type=int,
                    help='replay buffer')
parser.add_argument('--log-level', default='INFO', type=str,
                    help='log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)')
parser.add_argument('--mode', default='RUN-TRAINED',
                    type=str, help='run mode')
parser.add_argument('--num-epochs', default=1, type=int,
                    help='number of epochs (train)')
parser.add_argument('--num-steps', default=100, type=int,
                    help='number of time steps (train or run)')
parser.add_argument('--policy', default='SimpleRuleBased', type=str,
                    help='policy class name')


def main(
    batch_size, ckpt_path, exp_replay_buffer, log_level, mode, num_epochs,
    num_steps, policy
):
    mode = mode.upper()
    if mode not in RUN_MODES:
        print(__doc__)
        sys.exit(1)

    logfilename = 'log_' + __file__.split('/')[-1].split('.')[0] \
        + str(int(time.time())) + '.txt'
    logging.basicConfig(
        filename=logfilename, level=get_logging_level(log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Starting...")
    LOGGER.info(__file__)

    arguments_dict = create_default_arguments_dict(policy, mode)
    # TODO - add code to allow arg dict override...
    policy_class = create_policy(policy, arguments_dict)
    data_source = None  # TODO - utils function to get file or make engine
    trainer = create_trainer(data_source, policy_class, mode, num_epochs,
                             num_steps)
    trainer.build_or_restore_model_and_optimizer()
    if 'TRAIN' in mode:
        trainer.train_model_with_target_replay(num_epochs)
    else:
        trainer.run_model(num_epochs)
    # trainer.save_losses_and_winpct_plots(make_plot)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
