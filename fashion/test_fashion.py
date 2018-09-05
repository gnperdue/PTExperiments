import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from fashion import FashionMNISTDataset
from fashion import Standardize, ToTensor
from fashion import TESTFILE
from fashion import MEANFILE, STDFILE


def show_imgs(dset, show=False, figname='test_imgs.pdf'):
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
            if show:
                plt.show()
            else:
                plt.savefig(figname, bbox_inches='tight')
                plt.close()
            break


if __name__ == '__main__':
    fashion_testset = FashionMNISTDataset(TESTFILE)
    show_imgs(fashion_testset)

    standardizer = Standardize(mean_file=MEANFILE, std_file=STDFILE)
    standardized_testset = FashionMNISTDataset(TESTFILE, standardizer)
    show_imgs(standardized_testset, figname='std_test_imgs.pdf')

    trnsfrms = transforms.Compose([
        standardizer, ToTensor()
    ])
    # trnsfrms = transforms.Compose([
    #     ToTensor()
    # ])

    transformed_testset = FashionMNISTDataset(TESTFILE, trnsfrms)
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
