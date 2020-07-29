import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

import copy
import random
import logging
import time
import datetime

from tqdm import tqdm
from games.gridworld import Gridworld as Game
from gridrl.utils import count_parameters


LOGGER = logging.getLogger(__name__)


class RLTrainer(object):

    def __init__(self, game_parameters, train_parameters):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        LOGGER.info('Device = {}'.format(self.device))
        self.action_set = Game.action_set
        self.game_size = game_parameters['size']
        self.game_mode = game_parameters['mode']
        self._configure(train_parameters)

    def _configure(self, train_d):
        # set these properties directly
        self.model = None
        self.target_model = None
        self.loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')
        self.optimizer = None
        self.start_epoch = None
        self.start_step = None
        # train configuration
        self.batch_size = train_d.get('batch_size', 100)
        self.buffer = train_d.get('buffer', 500)
        self.ckpt_path = train_d.get('ckpt_path', './ckpt_deepq_targrep.tar')
        self.epsilon = train_d.get('epsilon', 1.0)
        self.gamma = train_d.get('gamma', 0.95)
        self.learning_rate = train_d.get('learning_rate', 1e-3)
        self.max_moves = train_d.get('max_moves', 50)
        self.min_epsilon = train_d.get('min_epsilon', 0.1)
        self.saved_losses_path = train_d.get('saved_losses_path', 'losses.npy')
        self.saved_winpct_path = train_d.get('saved_winpct_path', 'winpct.npy')
        self.target_network_update = train_d.get(
            'target_network_update', 500
        )
        self.show_progress = train_d.get('show_progress', False)
        self._f = tqdm if self.show_progress else (lambda x: x)
        # for visualizing training progress
        self.losses = []
        self.winpct = []
        try:
            losses_file = self.saved_losses_path
            winpct_file = self.saved_winpct_path
            losses_arr = np.load(losses_file)
            winpct_arr = np.load(winpct_file)
            self.losses.extend(list(map(tuple, losses_arr)))
            self.winpct.extend(list(map(tuple, winpct_arr)))
        except FileNotFoundError:
            pass

    def build_or_restore_model_and_optimizer(
        self, build_model_function, conv, epsilon=None
    ):
        '''
        params:
        * build_model_function - fn that returns the model to be used
        * conv - T/F - how to reshape game input
        '''
        self.model = build_model_function()
        LOGGER.info('model has {} parameters'.format(
            count_parameters(self.model)
        ))
        self.conv = conv
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)
        self.start_epoch = 0
        self.start_step = 0

        try:
            checkpoint = torch.load(self.ckpt_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.start_step = checkpoint['step'] + 1
            self.epsilon = checkpoint['epsilon']
            LOGGER.info('Loaded checkpoint from {}'.format(self.ckpt_path))
        except FileNotFoundError:
            LOGGER.info('No checkpoint found...')

        if epsilon is not None:
            self.epsilon = epsilon

        LOGGER.info('Model state dict:')
        for param_tensor in self.model.state_dict():
            LOGGER.info(str(param_tensor) + '\t'
                        + str(self.model.state_dict()[param_tensor].size()))
        LOGGER.info('Optimizer state dict:')
        for var_name in self.optimizer.state_dict():
            LOGGER.info(str(var_name) + '\t'
                        + str(self.optimizer.state_dict()[var_name]))

        self.model.to(self.device)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.to(self.device)

    def _new_game(self):
        game = Game(size=self.game_size, mode=self.game_mode)
        return game

    def _game_move_for_reward(self, game, action):
        game.makeMove(action)
        reward = game.reward()
        return reward

    def _check_continue(self, reward, move_count):
        if reward != Game.default_reward or move_count > self.max_moves:
            return 0, 0
        return move_count, 1

    def _get_torch_state(self, game, noise_factor=100.0):
        game_boardsz = game.board.size
        if self.conv:
            state_ = game.board.render_np() + \
                np.random.rand(4, game_boardsz, game_boardsz) / noise_factor
            state_ = state_.reshape(1, 4, game_boardsz, game_boardsz)
        else:
            full_game_size = 4 * game_boardsz * game_boardsz
            state_ = game.board.render_np().reshape(1, full_game_size) + \
                np.random.rand(1, full_game_size) / noise_factor
        state = Variable(torch.from_numpy(state_).float()).to(self.device)
        return state

    def _compute_action(self, qval):
        qval_ = qval.cpu().data.numpy()
        if np.random.rand() < self.epsilon:
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)
        action_str = self.action_set[action_]
        return action_str, action_

    def _build_trainbatch(self, replay):
        minibatch = random.sample(replay, self.batch_size)
        X_train = Variable(torch.empty(self.batch_size, 4, dtype=torch.float))
        y_train = Variable(torch.empty(self.batch_size, 4, dtype=torch.float))
        # Fill X_train and y_train minibatch tensors by index `h` by
        # looping through memory and computing the Q-values before (X)
        # and after (y) each move
        h = 0
        for memory in minibatch:
            old_state, action_m, reward_m, new_state_m = memory
            old_qval = self.model(old_state)
            new_qval = self.target_model(new_state_m).cpu().data.numpy()
            max_qval = np.max(new_qval)
            y = torch.zeros((1, 4))
            y[:] = old_qval[:]
            if reward_m == -1:
                update = (reward_m + (self.gamma * max_qval))
            else:
                update = reward_m
            y[0][action_m] = update
            X_train[h] = old_qval
            y_train[h] = Variable(y)
            h += 1
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        return X_train, y_train

    def _anneal_epsilon(self, epochs):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= (1. / epochs)

    def _save_model(self, epoch, step):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'epsilon': self.epsilon,
        }, self.ckpt_path)

    def _play_single_game(self):
        LOGGER.debug('Tesing single game:')
        i = 0
        game = self._new_game()
        state = self._get_torch_state(game)
        LOGGER.debug(' Initial state:')
        LOGGER.debug(str(game.display()))
        status = 1
        while status == 1:
            qval = self.model(state)
            action, action_ = self._compute_action(qval)
            LOGGER.debug('  Move #: %s; Taking action: %s' % (i, action))
            reward = self._game_move_for_reward(game, action)
            state = self._get_torch_state(game)
            LOGGER.debug(str(game.display()))
            if reward != Game.default_reward:
                status = 0
                if reward == Game.win_reward:
                    return 1
                elif reward == Game.loss_reward:
                    return 0
                else:
                    raise ValueError('Bad game reward.')
            i += 1
            if (i > self.max_moves):
                return 0

    def play_model(self, max_games=1000, step_count=None):
        wins = 0
        for i in self._f(range(max_games)):
            win = self._play_single_game()
            if win:
                wins += 1
        win_perc = float(wins) / max_games
        fs = '  step={:010d}, games={:04d}/{:04d}, win pct={:1.5f}'
        LOGGER.info(fs.format(step_count, wins, max_games, win_perc))
        if step_count is not None:
            self.winpct.append((step_count, win_perc))

    def train_model_with_target_replay(self, epochs):
        LOGGER.info('Running training...')
        LOGGER.info(' Start epoch = {}; start step = {}; epsilon = {}'.format(
            self.start_epoch, self.start_step, self.epsilon
        ))
        replay, step, c_step = [], self.start_step, 0
        t0 = time.perf_counter()
        for epoch in self._f(
            range(self.start_epoch, self.start_epoch + epochs)
        ):
            game = self._new_game()
            state = self._get_torch_state(game)
            status, move_count = 1, 0
            while status == 1:
                move_count, step, c_step = move_count + 1, step + 1, c_step + 1
                if c_step > self.target_network_update:
                    LOGGER.info(" ...updating target model")
                    self.target_model.load_state_dict(self.model.state_dict())
                    c_step = 0
                # Compute Q-value and use to make move decisions
                qval = self.model(state)
                action, action_ = self._compute_action(qval)
                reward = self._game_move_for_reward(game, action)
                new_state = self._get_torch_state(game)

                # save move information into replay buffer
                if len(replay) < self.buffer:
                    replay.append((state, action_, reward, new_state))
                else:
                    replay.pop(0)
                    replay.append((state, action_, reward, new_state))
                    X_train, y_train = self._build_trainbatch(replay)
                    loss = self.loss_fn(X_train, y_train)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.losses.append((step, loss.item()))
                    self.optimizer.step()
                    LOGGER.debug(
                        'epoch={:06d}, step={:08d}, loss={:04.8f}'.format(
                            epoch, step, loss.item()
                        ))

                state = new_state
                move_count, status = self._check_continue(reward, move_count)

            # after game/epoch, update epsilon, ...
            LOGGER.info('replay buffer length at epoch {} end = {}'.format(
                epoch, len(replay)
            ))
            self._anneal_epsilon(epochs)
            if epoch % 50 == 0:
                LOGGER.info(' Playing model at epoch {}'.format(epoch))
                self.play_model(step_count=step)
                self._save_model(epoch, step)

        # Final save
        self.play_model(step_count=step)
        self._save_model(epoch, step)
        t1 = time.perf_counter()
        LOGGER.info('Total train time for {} epochs = {:04.3f}s = {}'.format(
            epochs, t1 - t0, str(datetime.timedelta(seconds=t1-t0))
        ))

    def save_losses_and_winpct_plots(self, make_plot):

        def running_mean(x, N=500):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)

        wins_arr = np.asarray(self.winpct)
        loss_arr = np.asarray(self.losses)
        np.save(self.saved_losses_path, loss_arr)
        np.save(self.saved_winpct_path, wins_arr)

        if make_plot:

            try:
                wins_stps = wins_arr[:, 0]
                wins_wins = wins_arr[:, 1]
                loss_stps = loss_arr[:, 0]
                loss_loss = loss_arr[:, 1]
                nrunning_mean = 5
                running_loss = running_mean(loss_loss, N=nrunning_mean)

                fig = plt.figure()
                gs = plt.GridSpec(1, 2)

                ax1 = plt.subplot(gs[0])
                ax1.scatter(wins_stps, wins_wins)
                ax1.set_xlabel('Steps')
                ax1.set_ylabel('Win percentage')
                ax1.set_ylim(0., 1.)
                ax2 = plt.subplot(gs[1])
                ax2.scatter(loss_stps[nrunning_mean - 1:], running_loss)
                ax2.set_xlabel('Steps')
                ax2.set_ylabel('Running Loss')
                ax2.set_ylim(0., 3.)

                fig.tight_layout()
                figname = 'deepq_targrep_%d.pdf' % (time.time())
                plt.savefig(figname, bbox_inches='tight')
            except IndexError as e:
                LOGGER.error(e)
                LOGGER.error('Empty loss/winpct arrays...')
