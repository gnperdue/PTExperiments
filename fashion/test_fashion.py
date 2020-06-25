import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch_fashion_data import FashionMNISTDataset
from torch_fashion_data import Standardize, ToTensor
from torch_fashion_data import make_file_paths

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='', type=str, help='data dir')
parser.add_argument('--show', default=False, action='store_true',
                    help='show images')


def show_imgs(dset, figname='test_imgs.pdf'):
    fig = plt.figure()
    for i in range(len(dset)):
        sample = dset[i]
        label = np.argmax(sample['label'])
        label_name = FashionMNISTDataset.label_names[label]
        ax = plt.subplot(2, 2, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{} = {}'.format(i, label_name))
        ax.axis('off')
        plt.imshow(sample['image'][0, :, :])
        print(' image {} mean = {}, stddev = {}'.format(
            i, np.mean(sample['image']), np.std(sample['image'])
        ))
        plt.pause(0.001)
        if i == 3:
            plt.savefig(figname, bbox_inches='tight')
            plt.close()
            break


def main(data_dir, show):

    testfile, trainfile, meanfile, stdfile = make_file_paths(data_dir)

    fashion_testset = FashionMNISTDataset(testfile)
    if show:
        show_imgs(fashion_testset)

    standardizer = Standardize(mean_file=meanfile, std_file=stdfile)
    standardized_testset = FashionMNISTDataset(testfile, standardizer)
    if show:
        show_imgs(standardized_testset, figname='std_test_imgs.pdf')

    trnsfrms = transforms.Compose([
        standardizer, ToTensor()
    ])

    transformed_testset = FashionMNISTDataset(testfile, trnsfrms)
    for i in range(len(transformed_testset)):
        sample = transformed_testset[i]
        print(i, sample['image'].size(), sample['label'].size())
        if i == 3:
            break

    # must use `num_workers=1` here - parallel access is not correctly
    # configured by the reader class.
    test_dataloader = DataLoader(
        transformed_testset, batch_size=64, shuffle=True, num_workers=1
    )
    for i_batch, sample_batched in enumerate(test_dataloader):
        if i_batch > 20:
            break
        print(
            i_batch,
            sample_batched['image'].size(),
            sample_batched['label'].size()
        )
        if i_batch % 10 == 9:
            print(sample_batched['label'])


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
