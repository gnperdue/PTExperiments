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
        dm = utils.configure_and_get_testing_data_manager()
        model, _ = utils.configure_and_get_SimpleConvNet()
        self.trainer = trainers.VanillaTrainer(dm, model)

    def test_stuff(self):
        pass


if __name__ == '__main__':
    unittest.main()
