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

# from policies.rule_based import SimpleRuleBased
# from sim.engines import SimulationMachine
# from sim.data_model import DataGenerator
# from sim.data_model import NoiseModel
# from sim.recorders import MachineStateTextRecorder
from utils.util_funcs import get_logging_level

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
parser.add_argument('--num-steps', default=100, type=int,
                    help='number of time steps')
parser.add_argument('--policy', default='SimpleRuleBased', type=str,
                    help='policy class name')


RUN_MODES = [
    'RUN-TRAINED', 'TRAIN-HISTORICAL', 'TRAIN-LIVE'
]


def main(
    batch_size, ckpt_path, exp_replay_buffer, log_level, mode, num_steps,
    policy
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

    if policy == 'SimpleRuleBased':
        if mode == 'RUN-TRAINED':
            # trainer.run_historical(policy, batch_size, ckpt_path, num_steps)
            # trained.run(policy, batch_size, ckpt_path, num_steps)
            pass
        elif mode == 'TRAIN-HISTORICAL':
            pass
        elif mode == 'TRAIN-LIVE':
            pass
        else:
            raise ValueError('Inappropriate mode ({}) for policy ({}).'.format(
                mode, policy
            ))
    else:
        raise NotImplementedError


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
