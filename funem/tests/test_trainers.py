'''
Usage:
    python test_trainers.py -v
    python test_trainers.py
'''
import unittest

import trainers.qtrainers as qtrainers


class TestQTrainer(unittest.TestCase):

    def setUp(self):
        learner_instance = None
        data_source = None
        performance_memory_maxlen = 1000
        self.trainer = qtrainers.QTrainer(
            learner_instance, data_source, performance_memory_maxlen
        )

    def tearDown(self):
        pass

    def test_train_or_run_model(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.train_or_run_model(False)


class TestLiveQTrainer(unittest.TestCase):

    def setUp(self):
        arguments_dict = {}
        arguments_dict['num_steps'] = 500
        learner_instance = None
        data_source = None
        self.trainer = qtrainers.LiveQTrainer(
            learner_instance, data_source, arguments_dict
        )

    def tearDown(self):
        pass

    def test_train_or_run_model(self):
        self.fail('Finish the test...')


if __name__ == '__main__':
    unittest.main()
