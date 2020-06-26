import logging
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim

LOGGER = logging.getLogger(__name__)


class VanillaTrainer(object):
    def __init__(self, data_manager, model, show_progress=False):
        self.data_manager = data_manager
        self.model = model
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self._f = tqdm if show_progress else (lambda x: x)
        # TODO - add method to configure these...
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9)

    def train(self, num_epochs, batch_size, short_test=False):
        train_dataloader, _ = self.data_manager.get_data_loaders(
            batch_size=batch_size)
        for epoch in range(num_epochs):
            LOGGER.info('epch = {}'.format(epoch))

            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_dataloader, 0):
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
                if i % 10 == 9:
                    LOGGER.info('[%d, %5d] loss: %.3f' % (
                        epoch + 1, i + 1, running_loss / 10
                    ))
                    running_loss = 0.0

                # TODO - include validation

        # TODO - properly use the ckpt_path here...
        torch.save(self.model.state_dict(),
                   './myfashionmodel_short.pth'
                   if short_test else './myfashionmodel.pth')

    def test(self):
        '''no test set for Fashion MNIST here...'''
        pass
