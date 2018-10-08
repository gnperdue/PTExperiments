import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import h5py

import os
import argparse

from torch_fashion_data import make_data_loaders
from torch_fashion_data import FashionMNISTDataset as fdset

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num-epochs', default=1, type=int, help='num. epochs')
parser.add_argument('--data-dir', default='', type=str, help='data dir')
parser.add_argument('--model-dir', default='fashion', type=str,
                    help='model dir')


class Net(nn.Module):
    '''sizes for 28x28 image'''

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.pool(x))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class WrapDataLoader(object):

    def __init__(self, dl):
        self.dl = dl

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for data in batches:
            yield data['image'], data['label']


def count_parameters(model):
    '''https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(
    batch_size, num_epochs, data_dir, model_dir
):
    train_dataloader, test_dataloader = make_data_loaders(
        data_dir, batch_size=64
    )

    net = Net()
    print(count_parameters(net))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # return 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train
    for epoch in range(num_epochs):
        print('epch = {}'.format(epoch))

        running_loss = 0.0
        trndl = WrapDataLoader(train_dataloader)
        for i, (inputs, labels) in enumerate(trndl, 0):
            if i > 10:
                break
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (
                    epoch + 1, i + 1, running_loss / 10
                ))
                running_loss = 0.0

    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            if i > 40:
                break
            print('testing batch {}'.format(i))
            images, labels = data['image'], data['label']
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            if i % 10 == 9:
                print('batch {} truth: '.format(i)
                      + ' '.join('%5s' % fdset.label_names[labels[j]]
                                 for j in range(4)))
                print('         preds: '
                      + ' '.join('%5s' % fdset.label_names[preds[j]]
                                 for j in range(4)))

    print('accuracy of net on 10,000 test images: %d %%' % (
        100 * correct / total
    ))
    print('finished training')


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
