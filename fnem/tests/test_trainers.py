import unittest
import trainers.trainers as trainers

from sim.data_model import DataGenerator as Generator
from sim.data_model import NoiseModel as Noise
from sim.engines import SimulationMachine
from policies.rule_based import SimpleRuleBased
from utils.common_defs import DEFAULT_COMMANDS
from tests.common_defs import REFERNECE_LOG


class TestBaseTrainers(unittest.TestCase):

    def setUp(self):
        policy = SimpleRuleBased(
            time=0.0, setting=10.0, amplitude=10.0, period=2.0,
            commands_array=DEFAULT_COMMANDS
        )
        self.trainer = trainers.Trainer(policy)

    def test_configuration(self):
        self.assertIsNotNone(self.trainer.device)
        self.assertIsNone(self.trainer.training_sim_machine)
        self.assertIsNone(self.trainer.training_data_file)

    def test_basic_methods(self):
        # with self.assertRaises(NotImplementedError):
        #     self.trainer.build_or_restore_model_and_optimizer()
        with self.assertRaises(NotImplementedError):
            self.trainer.train_model_with_target_replay()
        with self.assertRaises(NotImplementedError):
            self.trainer.save_performance_plots()


class TestHistoricalTrainers(unittest.TestCase):

    def setUp(self):
        policy = SimpleRuleBased(
            time=0.0, setting=10.0, amplitude=10.0, period=2.0,
            commands_array=DEFAULT_COMMANDS
        )
        self.trainer = trainers.HistoricalTrainer(
            policy=policy, training_file=REFERNECE_LOG
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
        policy = SimpleRuleBased(
            time=0.0, setting=10.0, amplitude=10.0, period=2.0,
            commands_array=DEFAULT_COMMANDS
        )

        dgen = Generator()
        nosgen = Noise()
        machine = SimulationMachine(
            setting=10.0, data_generator=dgen, noise_model=nosgen, logger=None
        )

        self.trainer = trainers.LiveTrainer(
            policy=policy, sim_machine=machine
        )

    def tearDown(self):
        pass

    def test_configuration(self):
        self.assertIsNotNone(self.trainer.device)
        self.assertIsNotNone(self.trainer.training_sim_machine)

    def test_build_model_and_optimizer(self):
        pass

    def test_train_model_with_target_replay(self):
        pass

    def test_save_performance_plots(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.save_performance_plots()


if __name__ == '__main__':
    unittest.main()
