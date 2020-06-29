import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import h5py

import os

from ptlib.datasets import FashionMNISTDataset
from ptlib.transforms import Standardize
from ptlib.transforms import ToTensor


class WrapFashionDataLoader(object):
    '''class to manage pulling returned dict apart'''

    def __init__(self, dl):
        self.dl = dl

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for data in batches:
            yield data['image'], data['label']


class FashionDataManager(object):
    '''main data access class'''
    def __init__(self, data_dir):
        self.testfile = None
        self.trainfile = os.path.join(data_dir, 'fashion_train.hdf5')
        self.validfile = os.path.join(data_dir, 'fashion_test.hdf5')
        self.meanfile = os.path.join(data_dir, 'fashion_mean.npy')
        self.stdfile = os.path.join(data_dir, 'fashion_stddev.npy')
        self.label_names = FashionMNISTDataset.label_names

    def make_means(self):
        if os.path.isfile(self.meanfile) and os.path.isfile(self.stdfile):
            return
        f = h5py.File(self.trainfile, 'r')
        m = np.mean(f['fashion/images'], axis=0)
        s = np.std(f['fashion/images'], axis=0)
        np.save(self.meanfile, m)
        np.save(self.stdfile, s)
        f.close()

    def get_data_loaders(self, batch_size):
        standardizer = Standardize(
            mean_file=self.meanfile, std_file=self.stdfile)
        trnsfrms = transforms.Compose([
            standardizer, ToTensor()
        ])

        train_dataloader = None
        if self.trainfile is not None:
            fashion_trainset = FashionMNISTDataset(self.trainfile, trnsfrms)
            train_dataloader = WrapFashionDataLoader(DataLoader(
                fashion_trainset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1
            ))

        valid_dataloader = None
        if self.validfile is not None:
            fashion_validset = FashionMNISTDataset(self.validfile, trnsfrms)
            valid_dataloader = WrapFashionDataLoader(DataLoader(
                fashion_validset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1
            ))

        test_dataloader = None
        if self.testfile is not None:
            fashion_testset = FashionMNISTDataset(self.testfile, trnsfrms)
            test_dataloader = WrapFashionDataLoader(DataLoader(
                fashion_testset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1
            ))

        return train_dataloader, valid_dataloader, test_dataloader
