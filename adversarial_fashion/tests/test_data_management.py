'''
Usage:
    python test_transforms.py -v
    python test_transforms.py
'''
import unittest
import numpy as np
import h5py
import os
import torch
import math

import ptlib.dataloaders as dataloaders

TRAINH5 = 'synth_train.h5'
TESTH5 = 'synth_test.h5'
MEANFILE = 'synth_mean.npy'
STDFILE = 'synth_std.npy'


def fill_hdf5(file_name, images, labels):
    f = h5py.File(file_name, 'w')
    grp = f.create_group('fashion')
    images_set = grp.create_dataset(
        'images', np.shape(images), dtype='uint8', compression='gzip'
    )
    labels_set = grp.create_dataset(
        'labels', np.shape(labels), dtype='uint8', compression='gzip'
    )
    images_set[...] = images
    labels_set[...] = labels
    f.close()


def cleanup_synth():
    for filename in [TRAINH5, TESTH5, MEANFILE, STDFILE]:
        if os.path.isfile(filename):
            os.remove(filename)


def make_synth_h5():
    cleanup_synth()
    synth_train_images = np.random.randint(
        0, high=256, size=(1000, 1, 28, 28)).astype('uint8')
    synth_train_labels = np.random.randint(
        0, high=10, size=(1000, 1)).astype('uint8')
    synth_test_images = np.random.randint(
        0, high=256, size=(1000, 1, 28, 28)).astype('uint8')
    synth_test_labels = np.random.randint(
        0, high=10, size=(1000, 1)).astype('uint8')
    fill_hdf5(TRAINH5, synth_train_images, synth_train_labels)
    fill_hdf5(TESTH5, synth_test_images, synth_test_labels)


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.dm = dataloaders.FashionDataManager(data_dir=".")
        make_synth_h5()
        self.dm.testfile = TESTH5
        self.dm.trainfile = TRAINH5
        self.dm.meanfile = MEANFILE
        self.dm.stdfile = STDFILE
        self.dm.make_means()

    def test_dataloaders(self):
        batch_size = 100
        train_loader, test_loader = self.dm.get_data_loaders(
            batch_size=batch_size)
        for i, (inputs, labels) in enumerate(train_loader):
            inputs_l = list(inputs.shape)
            labels_l = list(labels.shape)
            for idx, j in enumerate([batch_size, 1, 28, 28]):
                self.assertEqual(inputs_l[idx], j)
            self.assertEqual(labels_l[0], batch_size)
            self.assertTrue(math.isclose(
                torch.mean(inputs).item(), 0, rel_tol=0.05, abs_tol=0.05))
            self.assertTrue(math.isclose(
                torch.std(inputs).item(), 1.0, rel_tol=0.05, abs_tol=0.05))


if __name__ == '__main__':
    unittest.main()
