import logging
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from ptlib.model_handlers import ModelHandlerBase

LOGGER = logging.getLogger(__name__)


class VanillaTrainer(ModelHandlerBase):
    def __init__(self, data_manager, model, ckpt_path, tnsrbrd_out_dir,
                 log_freq=100, show_progress=False):
        super(VanillaTrainer, self).__init__(model, ckpt_path)
        self.dm = data_manager
        self.log_freq = log_freq
        self._f = tqdm if show_progress else (lambda x: x)
        self.start_epoch = 0
        self.writer = SummaryWriter(tnsrbrd_out_dir)
        # TODO - add method to configure these...
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9)

    def _write_to_log(self, batch_idx):
        return True if (batch_idx + 1) % self.log_freq == 0 else False

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
            correct = 0
            total = 0
            valid_loss = 0
            with torch.no_grad():
                for i, (images, labels) in enumerate(valid_dl, 0):
                    if short_test and i >= 20:
                        break
                    images, labels = \
                        images.to(self.device), labels.to(self.device)
                    # self.model.eval()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    valid_loss += loss.item()
                    _, preds = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()
                    if self._write_to_log(i):
                        LOGGER.info(
                            'batch {} truth: '.format(i)
                            + ' '.join('%5s' % self.dm.label_names[labels[j]]
                                       for j in range(4)))
                        LOGGER.info(
                            '         preds: '
                            + ' '.join('%5s' % self.dm.label_names[preds[j]]
                                       for j in range(4)))

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
