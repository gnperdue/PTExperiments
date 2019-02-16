import logging
import time
from collections import deque

from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.common_defs import DATASET_MACHINE_PLT_HEATS_TEMPLATE
from utils.common_defs import DATASET_MACHINE_PLT_LOSS_TEMPLATE
from utils.common_defs import DATASET_MACHINE_PLT_SENSORS_TEMPLATE


LOGGER = logging.getLogger(__name__)


class QTrainer(object):
    '''base class - defines the API'''

    def __init__(self, qlearner, data_source, performance_memory_maxlen=5000):
        # TODO - pass in time stamp
        self.data_source = data_source
        self.tstamp = int(time.time())
        self.qlearner = qlearner
        self.performance_memory_maxlen = performance_memory_maxlen
        self.steps = deque([], maxlen=self.performance_memory_maxlen)
        self.m1 = deque([], maxlen=self.performance_memory_maxlen)
        self.m2 = deque([], maxlen=self.performance_memory_maxlen)
        self.m3 = deque([], maxlen=self.performance_memory_maxlen)
        self.m4 = deque([], maxlen=self.performance_memory_maxlen)
        self.ts = deque([], maxlen=self.performance_memory_maxlen)
        self.losses = deque([], maxlen=self.performance_memory_maxlen)
        self.totals = deque([], maxlen=self.performance_memory_maxlen)
        self.heats = deque([], maxlen=self.performance_memory_maxlen)
        self.settings = deque([], maxlen=self.performance_memory_maxlen)
        self.setting_diffs = deque([], maxlen=self.performance_memory_maxlen)
        self._replay_buffer_length = 100
        self._target_network_update_freq = 500  # update every 500 steps
        self.heats_figname = 'heats.pdf'
        self.loss_figname = 'loss.pdf'
        self.sensors_figname = 'sensors.pdf'

    def restore_model_and_optimizer(self):
        self.qlearner.restore_model_and_optimizer()

    def train_or_run_model(self, train):
        raise NotImplementedError

    def save_performance_plots(self):
        fig = plt.Figure(figsize=(10, 6))
        gs = plt.GridSpec(1, 2)
        ax1 = plt.subplot(gs[0])
        ax1.scatter(self.ts, self.m1, c='r')
        ax1.scatter(self.ts, self.m2, c='g')
        ax1.scatter(self.ts, self.m3, c='b')
        ax1.scatter(self.ts, self.m4, c='y')
        ax1.set_title('sensors')
        ax2 = plt.subplot(gs[1])
        ax2.scatter(self.ts, self.totals, c='k')
        ax2.set_title('totals')
        fig.tight_layout()
        plt.savefig(self.sensors_figname, bbox_inches='tight')

        fig = plt.Figure(figsize=(10, 6))
        gs = plt.GridSpec(1, 2)
        ax1 = plt.subplot(gs[0])
        ax1.scatter(self.ts, self.heats, c='k')
        ax1.set_title('heats')
        ax2 = plt.subplot(gs[1])
        ax2.scatter(self.ts, self.settings, c='k')
        ax2.set_title('settings')

        fig.tight_layout()
        plt.savefig(self.heats_figname, bbox_inches='tight')

        fig = plt.Figure(figsize=(10, 6))
        gs = plt.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0])
        ll = len(self.losses)
        ax1.scatter(list(self.ts)[-ll:], self.losses, c='r')
        ax1.set_title('losses')

        fig.tight_layout()
        plt.savefig(self.loss_figname, bbox_inches='tight')

    def _add_log_data(self, obs, setting, time, heat, loss_value):
        # obs is 20*4 sensor values
        obs_ = obs.cpu().data.numpy()
        self.heats.append(heat.item())
        self.ts.append(time.item())
        self.settings.append(setting.item())
        self.m4.append(obs_[-1])
        self.m3.append(obs_[-2])
        self.m2.append(obs_[-3])
        self.m1.append(obs_[-4])
        self.totals.append(sum(obs_[-4:]))
        self.losses.append(loss_value)


class LiveQTrainer(QTrainer):
    '''data source is a sim machine we interrogate for steps and values'''

    def __init__(self, qlearner, data_source, arguments_dict):
        super(LiveQTrainer, self).__init__(qlearner=qlearner,
                                           data_source=data_source)
        self.heats_figname = (DATASET_MACHINE_PLT_HEATS_TEMPLATE %
                              self.tstamp) + '.pdf'
        self.loss_figname = (DATASET_MACHINE_PLT_LOSS_TEMPLATE % self.tstamp) \
            + '.pdf'
        self.sensors_figname = (DATASET_MACHINE_PLT_SENSORS_TEMPLATE %
                                self.tstamp) + '.pdf'
        self.num_steps = arguments_dict['num_steps']
        self.show_progress = arguments_dict.get('show_progress', False)
        self._f = tqdm if self.show_progress else (lambda x: x)

    def train_or_run_model(self, train):
        if train:
            LOGGER.info('Running training...')
        else:
            LOGGER.info('Running model...')
        # TODO - specialize log message based on `if train`
        log_msg = ' step={:08d}, epsilon={:04.4f}, loss={:04.8f}'

        # TODO - use a deque for the replay buffer since we imported it anyway
        replay_buffer = []
        data_iter = iter(self.data_source)
        observation, _, _, heat = next(data_iter)
        for step in self._f(
            range(self.qlearner.start_step,
                  self.qlearner.start_step + self.num_steps)
        ):
            if step % self._target_network_update_freq == 0:
                LOGGER.info('Updating target network on step {}'.format(step))
                self.qlearner.update_target_model()
            qvalue = self.qlearner.compute_qvalues(observation)
            action_ = self.qlearner.compute_action(qvalue)
            self.data_source.update_setting(action_)
            new_observation, setting, time, heat = next(data_iter)

            # TODO - test with train==False
            if train:
                buffer_size = len(replay_buffer)
                if buffer_size < (self._replay_buffer_length - 1):
                    replay_buffer.append(
                        (observation, action_, heat.item(), new_observation)
                    )
                else:
                    if buffer_size == self._replay_buffer_length:
                        replay_buffer.pop(0)
                    replay_buffer.append(
                        (observation, action_, heat.item(), new_observation)
                    )
                    X_train, y_train = self.qlearner.build_trainbatch(
                        replay_buffer
                    )
                    loss_value = self.qlearner.train(X_train, y_train)
                    self.qlearner.anneal_epsilon(step)
                    self._add_log_data(
                        new_observation, setting, time, heat, loss_value
                    )
                    if (step + 1) % 100 == 0:
                        LOGGER.info(
                            log_msg.format(step, self.qlearner.epsilon,
                                           loss_value)
                        )
                        self.qlearner.save_model(epoch=0, step=step)

            observation = new_observation

        # final save after training
        if train:
            self.qlearner.save_model(epoch=0, step=step)
        self.data_source.close_dataset_logger()
