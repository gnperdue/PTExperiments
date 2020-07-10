import logging
from tqdm import tqdm
import torch.optim as optim
from tensorboardX import SummaryWriter

from ptlib.model_handlers import ModelHandlerBase

LOGGER = logging.getLogger(__name__)


class VanillaTrainer(ModelHandlerBase):
    def __init__(self, data_manager, model, ckpt_path, tnsrbrd_out_dir,
                 log_freq=100, show_progress=False):
        super(VanillaTrainer, self).__init__(
            data_manager, model, ckpt_path, log_freq)
        self._f = tqdm if show_progress else (lambda x: x)
        self.start_epoch = 0
        self.writer = SummaryWriter(tnsrbrd_out_dir)
        # TODO - add method to configure
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9)

    def train(self, num_epochs, batch_size, short_test=False):
        LOGGER.info('Starting training for {} for {} epochs'.format(
            self.__class__.__name__, num_epochs))
        train_dl, valid_dl, _ = self.dm.get_data_loaders(
            batch_size=batch_size)
        for epoch in self._f(range(
                self.start_epoch, self.start_epoch + num_epochs)):
            LOGGER.info('training epoch {}'.format(epoch))

            running_loss = 0.0
            for iter_num, (inputs, labels) in enumerate(train_dl, 0):
                if short_test and iter_num >= 20:
                    break
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                # TODO - configure a logging frequency...
                if self._write_to_log(iter_num):
                    running_loss = running_loss / self.log_freq
                    self.writer.add_scalar(
                        'train_loss', running_loss,
                        iter_num + epoch * len(train_dl))
                    LOGGER.info('[%d, %5d] loss: %.3f' % (
                        epoch + 1, iter_num + 1, running_loss
                    ))
                    running_loss = 0.0

            # save the model after each epoch
            self._save_state(epoch)

            # validation after each epoch
            LOGGER.info('validating epoch {}'.format(epoch))
            valid_loss, correct, total = self.run_inference_on_dataloader(
                valid_dl, short_test=short_test)

            self.writer.add_scalar(
                'valid_loss', valid_loss / len(valid_dl), epoch)
            self.writer.add_scalar(
                'valid_accuracy', 100 * correct / total, epoch)
            LOGGER.info('accuracy of net on {} test images: {} %%'.format(
                total, 100 * correct / total
            ))

        self.writer.close()
        LOGGER.info('finished training')

    def test(self):
        # TODO - add this...
        pass
