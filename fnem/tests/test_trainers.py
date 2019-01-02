import unittest
import trainers.trainers as trainers

from policies.rule_based import SimpleRuleBased
from utils.util_funcs import create_data_source
from utils.util_funcs import create_default_arguments_dict
from utils.util_funcs import create_policy
from utils.util_funcs import create_trainer
from utils.common_defs import DEFAULT_COMMANDS
from utils.common_defs import MACHINE_WITH_RULE_REFERNECE_LOG

TEST_TRAIN_ARGS_DICT = {
    'num_epochs': 1, 'num_steps': 100, 'replay_buffer_size': 100,
    'sequence_size': 20
}


class TestBaseTrainers(unittest.TestCase):

    def setUp(self):
        policy = SimpleRuleBased(
            time=0.0, amplitude=10.0, period=2.0,
            commands_array=DEFAULT_COMMANDS
        )
        self.trainer = trainers.Trainer(policy, data_source=None)

    def test_configuration(self):
        self.assertIsNotNone(self.trainer.device)
        self.assertIsNone(self.trainer.machine)
        self.assertIsNone(self.trainer.training_data_file)

    def test_basic_methods(self):
        self.trainer.build_or_restore_model_and_optimizer()
        with self.assertRaises(NotImplementedError):
            self.trainer.train_or_run_model(True)
        with self.assertRaises(NotImplementedError):
            self.trainer.save_performance_plots()


class TestHistoricalTrainers(unittest.TestCase):

    def setUp(self):
        policy = SimpleRuleBased(
            time=0.0, amplitude=10.0, period=2.0,
            commands_array=DEFAULT_COMMANDS
        )
        # TODO - need to make a DataLoader to hold the csv data accessor
        self.trainer = trainers.HistoricalTrainer(
            policy=policy, data_source=MACHINE_WITH_RULE_REFERNECE_LOG,
            arguments_dict=TEST_TRAIN_ARGS_DICT
        )

    def tearDown(self):
        pass

    def test_configuration(self):
        self.assertIsNotNone(self.trainer.device)
        self.assertIsNotNone(self.trainer.data_source)
        self.assertIsNotNone(self.trainer.num_steps)
        self.assertIsNotNone(self.trainer.num_epochs)
        self.assertIsNotNone(self.trainer.sequence_size)
        self.assertIsNotNone(self.trainer.replay_buffer_size)

    def test_build_model_and_optimizer(self):
        self.assertIsNotNone(self.trainer.policy)


class TestLiveTrainers(unittest.TestCase):

    def setUp(self):
        arguments_dict = create_default_arguments_dict('SimpleRuleBased',
                                                       'TRAIN-LIVE')
        policy_class = create_policy('SimpleRuleBased', arguments_dict)
        data_source = create_data_source('TRAIN-LIVE', 10, maxsteps=1000)
        self.trainer = create_trainer(data_source, policy_class, 'TRAIN-LIVE',
                                      num_epochs=None, num_steps=1000,
                                      sequence_size=1)

    def tearDown(self):
        pass

    def test_configuration(self):
        self.assertIsNotNone(self.trainer.device)
        self.assertIsNotNone(self.trainer.data_source)
        self.assertIsNotNone(self.trainer.num_steps)
        self.assertIsNotNone(self.trainer.sequence_size)
        self.assertIsNotNone(self.trainer.replay_buffer_size)
        self.assertIsNone(self.trainer.num_epochs)

    def test_build_model_and_optimizer(self):
        self.assertIsNotNone(self.trainer.policy)

    def test_train_or_run_model(self):
        # self.trainer.train_or_run_model(True)
        pass

    def test_save_performance_plots(self):
        pass


if __name__ == '__main__':
    unittest.main()
