'''
Usage:
    python test_trainers.py -v
    python test_trainers.py
'''
import unittest

import ptlib.trainers as trainers
import tests.utils as utils


class TestVanillaTrainer(unittest.TestCase):

    def setUp(self):
        dm = utils.configure_and_get_fash_data_manager()
        model, _ = utils.configure_and_get_SimpleConvNet()
        ckpt_path = './test_vanilla_trainer.tar'
        tnsrboard_out = '/tmp/fashion/test_trainers/'
        self.trainer = trainers.VanillaTrainer(
            dm, model, ckpt_path, tnsrboard_out)

    def test_private_functions(self):
        batch_idx = self.trainer.log_freq - 1
        self.assertEqual(True, self.trainer._write_to_log(batch_idx))
        self.assertEqual(self.trainer.start_epoch, 0)
        self.trainer._save_state(10)
        self.trainer.restore_model_and_optimizer()
        self.assertEqual(self.trainer.start_epoch, 11)

    def test_stuff(self):
        pass


if __name__ == '__main__':
    unittest.main()
