import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import h5py

import os


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


class WrapDataLoader(object):
    '''class to manage pulling returned dict apart'''

    def __init__(self, dl):
        self.dl = dl

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for data in batches:
            yield data['image'], data['label']


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


def make_data_loaders(data_dir, batch_size):
    testfile, trainfile, meanfile, stdfile = make_file_paths(data_dir)

    standardizer = Standardize(mean_file=meanfile, std_file=stdfile)
    trnsfrms = transforms.Compose([
        standardizer, ToTensor()
    ])

    fashion_trainset = FashionMNISTDataset(trainfile, trnsfrms)
    fashion_testset = FashionMNISTDataset(testfile, trnsfrms)

    train_dataloader = WrapDataLoader(DataLoader(
        fashion_trainset, batch_size=batch_size, shuffle=True, num_workers=1
    ))
    test_dataloader = WrapDataLoader(DataLoader(
        fashion_testset, batch_size=batch_size, shuffle=True, num_workers=1
    ))

    return train_dataloader, test_dataloader
