import logging
import time
from collections import deque

from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


class QTrainer(object):
    '''base class - defines the API'''

    def __init__(self, qlearner, data_source, performance_memory_maxlen=5000):
        # TODO - pass in time stamp
        self.data_source = data_source
        self.tstamp = int(time.time())
        self.qlearner = qlearner
        self.performance_memory_maxlen = performance_memory_maxlen
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

    def build_or_restore_model_and_optimizer(self):
        self.start_step = self.qlearner.build_or_restore_model_and_optimizer()

    def train_or_run_model(self, train):
        raise NotImplementedError

    def save_performance_plots(self):
        raise NotImplementedError


class LiveQTrainer(QTrainer):
    '''data source is a sim machine we interrogate for steps and values'''

    def __init__(self, qlearner, data_source, arguments_dict):
        super(LiveQTrainer, self).__init__(qlearner=qlearner,
                                           data_source=data_source)
        # self.figname = (DATASET_MACHINE_PLT_TEMPLATE % self.tstamp) + '.pdf'
        # self.loss_figname = (DATASET_MACHINE_PLT_LOSS_TEMPLATE % self.tstamp) \
        #     + '.pdf'
        self.num_steps = arguments_dict['num_steps']
        self.show_progress = arguments_dict.get('show_progress', False)
        self._f = tqdm if self.show_progress else (lambda x: x)

    def train_or_run_model(self, train):
        if train:
            LOGGER.info('Running training...')
        else:
            LOGGER.info('Running model...')

        # TODO - add epsilon greedy strategy parts
        replay_buffer = []
        data_iter = iter(self.data_source)
        observation, _, _, heat = next(data_iter)
        for step in self._f(
            range(self.qlearner.start_step,
                  self.qlearner.start_step + self.num_steps)
        ):
            # TODO - if step > x, do target network update, etc.
            qvalue = self.qlearner.compute_qvalues(observation)
            action_ = self.qlearner.compute_action(qvalue)
            self.data_source.update_setting(action_)
            new_observation, _, _, heat = next(data_iter)

            if train:
                buffer_size = len(replay_buffer)
                if buffer_size < self._replay_buffer_length:
                    replay_buffer.append(
                        (observation, action_, heat.item(), new_observation)
                    )
                else:
                    if buffer_size > self._replay_buffer_length:
                        replay_buffer.pop(0)
                    replay_buffer.append(
                        (observation, action_, heat.item(), new_observation)
                    )
                    X_train, y_train = self.qlearner.build_trainbatch(
                        replay_buffer
                    )
                    loss_value = self.qlearner.train(X_train, y_train)
                    LOGGER.debug('  step={:08d}, loss={:04.8f}'.format(
                        step, loss_value
                    ))

            observation = new_observation
            self.qlearner.anneal_epsilon(step)

    def save_performance_plots(self):
        pass
