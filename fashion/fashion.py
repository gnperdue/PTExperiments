import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import h5py
import matplotlib.pyplot as plt

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num-epochs', default=1, type=int, help='num. epochs')

TESTFILE = os.path.join(
    os.environ['HOME'], 'Dropbox/Data/RandomData/hdf5/fashion_test.hdf5'
)
TRAINFILE = os.path.join(
    os.environ['HOME'], 'Dropbox/Data/RandomData/hdf5/fashion_train.hdf5'
)
MEANFILE = os.path.join(
    os.environ['HOME'], 'Dropbox/Data/RandomData/hdf5/fashion_mean.npy'
)
STDFILE = os.path.join(
    os.environ['HOME'], 'Dropbox/Data/RandomData/hdf5/fashion_stddev.npy'
)


def make_means():
    f = h5py.File(TRAINFILE, 'r')
    m = np.mean(f['fashion/images'], axis=0)
    s = np.std(f['fashion/images'], axis=0)
    np.save(MEANFILE, m)
    np.save(STDFILE, s)
    f.close()


class FashionMNISTDataset(Dataset):

    label_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    def __init__(self, hdf5_file, transform=None):
        super(FashionMNISTDataset, self).__init__()
        self._nlabels = 10
        self._file = hdf5_file
        self._f = h5py.File(self._file, 'r')
        self.transform = transform

    def __len__(self):
        return len(self._f['fashion/labels'])

    def __getitem__(self, idx):
        image = self._f['fashion/images'][idx]
        label = self._f['fashion/labels'][idx].reshape([-1])
        oh_label = np.zeros((1, self._nlabels), dtype=np.uint8)
        oh_label[0, label] = 1
        oh_label = oh_label.reshape(self._nlabels,)
        sample = {'image': image, 'label': oh_label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Standardize(object):
    '''transform to shift images to unit variance and zero mean'''

    def __init__(self, mean_file, std_file):
        self.mean = np.load(mean_file)
        self.std = np.load(mean_file)
        assert (self.std == 0).any() == False, '0-values in std. dev.'

    def __call__(self, sample):
        img = (sample['image'] - self.mean) / self.std
        return {'image': img, 'label': sample['label']}


class ToTensor(object):
    '''transform for moving fashion data to tensors'''

    def __call__(self, sample):
        '''CrossEntropyLoss does not expect a one-hot encoded vector'''
        image, label = sample['image'], sample['label']
        # torch.max(torch.from_numpy(label).type(torch.LongTensor))
        return {
            'image': torch.from_numpy(image).float(),
            'label': torch.argmax(
                torch.from_numpy(label).type(torch.LongTensor)
            )
        }


class Net(nn.Module):
    '''sizes for 28x28 image'''

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main(
    batch_size, num_epochs
):
    standardizer = Standardize(mean_file=MEANFILE, std_file=STDFILE)
    trnsfrms = transforms.Compose([
        standardizer, ToTensor()
    ])

    fashion_trainset = FashionMNISTDataset(TRAINFILE, trnsfrms)
    fashion_testset = FashionMNISTDataset(TESTFILE, trnsfrms)

    train_dataloader = DataLoader(
        fashion_trainset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    test_dataloader = DataLoader(
        fashion_testset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    net = Net()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train
    for epoch in range(num_epochs):
        print('epch = {}'.format(epoch))

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # if i > 10:
            #     break
            inputs, labels = data['image'], data['label']
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
            # if i > 40:
            #     break
            print('testing batch {}'.format(i))
            images, labels = data['image'], data['label']
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            if i % 10 == 9:
                print('batch {} truth: '.format(i)
                      + ' '.join('%5s' % FashionMNISTDataset.label_names[
                          labels[j]
                      ] for j in range(4)))
                print('         preds: '
                      + ' '.join('%5s' % FashionMNISTDataset.label_names[
                          preds[j]
                      ] for j in range(4)))

    print('accuracy of net on 10,000 test images: %d %%' % (
        100 * correct / total
    ))
    print('finished training')


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
