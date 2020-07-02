import logging
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

LOGGER = logging.getLogger(__name__)

# TODO - log train and valid progress to tensorboardX


class VanillaTrainer(object):
    def __init__(self, data_manager, model, ckpt_path, log_freq=100,
                 show_progress=False):
        self.dm = data_manager
        self.model = model
        self.ckpt_path = ckpt_path
        self.log_freq = log_freq
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self._f = tqdm if show_progress else (lambda x: x)
        # TODO - add method to configure these...
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9)
        # TODO - look to see if checkpoints exist and load them if they do.
        # TODO - also apply loaded checkpoints to optimizer state.

    def _write_to_log(self, batch_idx):
        return True if (batch_idx + 1) % self.log_freq == 0 else False

    def train(self, num_epochs, batch_size, short_test=False):
        LOGGER.info('Starting training for {} for {} epochs'.format(
            self.__class__.__name__, num_epochs))
        train_dl, valid_dl, _ = self.dm.get_data_loaders(
            batch_size=batch_size)
        for epoch in self._f(range(num_epochs)):
            LOGGER.info('training epoch {}'.format(epoch))

            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_dl, 0):
                if short_test and i >= 10:
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
                if self._write_to_log(i):
                    LOGGER.info('[%d, %5d] loss: %.3f' % (
                        epoch + 1, i + 1, running_loss / 10
                    ))
                    running_loss = 0.0

            # save the model after each epoch
            torch.save(self.model.state_dict(),
                       './short_test.tar' if short_test else self.ckpt_path)

            # validation after each epoch
            LOGGER.info('validating epoch {}'.format(epoch))
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (images, labels) in enumerate(valid_dl, 0):
                    if short_test and i >= 10:
                        break
                    images, labels = \
                        images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
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

            LOGGER.info('accuracy of net on 10,000 test images: %d %%' % (
                100 * correct / total
            ))

        LOGGER.info('finished training')

    def test(self):
        '''no test set for Fashion MNIST here...'''
        pass
