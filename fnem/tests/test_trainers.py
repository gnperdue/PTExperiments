import unittest
import trainers.trainers as trainers

from tests.common_defs import REFERNECE_LOG


class TestHistoricalTrainers(unittest.TestCase):

    def setUp(self):
        self.trainer = trainers.HistoricalTrainer(
            training_file=REFERNECE_LOG
        )

    def tearDown(self):
        pass

    def test_configuration(self):
        self.assertIsNotNone(self.trainer.device)
        self.assertIsNotNone(self.trainer.training_data_file)

    def test_build_model_and_optimizer(self):
        self.assertIsNotNone(self.trainer)


class TestLiveTrainers(unittest.TestCase):

    def setUp(self):
        self.trainer = trainers.LiveTrainer()

    def tearDown(self):
        pass

    def test_configuration(self):
        pass

    def test_build_model_and_optimizer(self):
        self.assertIsNotNone(self.trainer)


if __name__ == '__main__':
    unittest.main()
