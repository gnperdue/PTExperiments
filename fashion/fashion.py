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

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num-epochs', default=1, type=int, help='num. epochs')
parser.add_argument('--data-dir', default='', type=str, help='data dir')
parser.add_argument('--model-dir', default='fashion', type=str,
                    help='model dir')


def make_file_paths(data_dir):
    testfile = os.path.join(data_dir, 'fashion_test.hdf5')
    trainfile = os.path.join(data_dir, 'fashion_train.hdf5')
    meanfile = os.path.join(data_dir, 'fashion_mean.npy')
    stdfile = os.path.join(data_dir, 'fashion_stddev.npy')
    return testfile, trainfile, meanfile, stdfile


def make_means(trainfile, meanfile, stdfile):
    f = h5py.File(trainfile, 'r')
    m = np.mean(f['fashion/images'], axis=0)
    s = np.std(f['fashion/images'], axis=0)
    np.save(meanfile, m)
    np.save(stdfile, s)
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


def count_parameters(model):
    '''https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(
    batch_size, num_epochs, data_dir, model_dir
):
    testfile, trainfile, meanfile, stdfile = make_file_paths(data_dir)

    standardizer = Standardize(mean_file=meanfile, std_file=stdfile)
    trnsfrms = transforms.Compose([
        standardizer, ToTensor()
    ])

    fashion_trainset = FashionMNISTDataset(trainfile, trnsfrms)
    fashion_testset = FashionMNISTDataset(testfile, trnsfrms)

    train_dataloader = DataLoader(
        fashion_trainset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    test_dataloader = DataLoader(
        fashion_testset, batch_size=batch_size, shuffle=True, num_workers=1
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
        for i, data in enumerate(train_dataloader, 0):
            if i > 10:
                break
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
