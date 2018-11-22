'''
Usage:
    python test_trainers.py -v
    python test_trainers.py
'''
import unittest

from gridrl.models import build_model as build_model_function
import gridrl.trainers as trainers


class TestTrainers(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTrainers, self).__init__(*args, **kwargs)
        self.batch_size = 100
        self.buffer = 500
        self.ckpt_path = 'test_ckpt.tar'
        self.gamma = 0.95
        self.learning_rate = 1e-3
        self.saved_losses_path = 'test_losses.npy'
        self.saved_winpct_path = 'test_winpct.npy'
        self.target_network_update = 500

    def setUp(self):
        game_parameters = {}
        game_parameters['mode'] = 'random'
        game_parameters['size'] = 4
        train_parameters = {}
        train_parameters['batch_size'] = self.batch_size
        train_parameters['buffer'] = self.buffer
        train_parameters['ckpt_path'] = self.ckpt_path
        train_parameters['gamma'] = self.gamma
        train_parameters['learning_rate'] = self.learning_rate
        train_parameters['saved_losses_path'] = self.saved_losses_path
        train_parameters['saved_winpct_path'] = self.saved_winpct_path
        train_parameters['target_network_update'] = self.target_network_update
        self.trainer = trainers.RLTrainer(game_parameters, train_parameters)
        self.trainer.build_or_restore_model_and_optimizer(build_model_function)

    def test_parameters(self):
        self.assertEqual(self.trainer.batch_size, self.batch_size)
        self.assertEqual(self.trainer.buffer, self.buffer)
        self.assertEqual(self.trainer.ckpt_path, self.ckpt_path)
        self.assertEqual(self.trainer.gamma, self.gamma)
        self.assertEqual(self.trainer.learning_rate, self.learning_rate)
        self.assertEqual(self.trainer.saved_losses_path,
                         self.saved_losses_path)
        self.assertEqual(self.trainer.saved_winpct_path,
                         self.saved_winpct_path)
        self.assertEqual(self.trainer.target_network_update,
                         self.target_network_update)
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.target_model)


if __name__ == '__main__':
    unittest.main()
