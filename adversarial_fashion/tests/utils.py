import os
import h5py
import numpy as np
import torch

import ptlib.dataloaders as dataloaders
import ptlib.models as models

TRAINH5 = 'synth_train.h5'
TESTH5 = 'synth_test.h5'
MEANFILE = 'synth_mean.npy'
STDFILE = 'synth_std.npy'


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


def configure_and_get_testing_data_manager():
    dm = dataloaders.FashionDataManager(data_dir=".")
    make_synth_h5()
    dm.testfile = TESTH5
    dm.trainfile = TRAINH5
    dm.meanfile = MEANFILE
    dm.stdfile = STDFILE
    dm.make_means()
    return dm


def configure_and_get_SimpleConvNet():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.SimpleConvNet()
    model.to(device)
    return model, device