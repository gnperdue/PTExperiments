import torch
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

from torch_fashion_data import make_data_loaders
from torch_fashion_data import FashionMNISTDataset as fdset


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num-epochs', default=1, type=int, help='num. epochs')
parser.add_argument('--data-dir',
                    default='/Users/perdue/Dropbox/Data/RandomData/hdf5',
                    type=str, help='data dir')
parser.add_argument('--model-dir', default='fashion', type=str,
                    help='model dir')


class MLPEncoder(nn.Module):
    '''sizes for 28x28 image'''

    def __init__(self, indim=784, latentdim=50):
        super(MLPEncoder, self).__init__()
        self.indim = indim
        self.latentdim = latentdim
        self.encoder = nn.Parameter(torch.rand(indim, latentdim))

    def forward(self, input):
        x = input.view(-1, self.indim)
        x = torch.sigmoid(torch.mm(x, self.encoder))
        x = torch.sigmoid(torch.mm(x, torch.transpose(self.encoder, 0, 1)))
        return x.view_as(input)


def count_parameters(model):
    '''https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(
    batch_size, num_epochs, data_dir, model_dir
):
    train_dataloader, test_dataloader = make_data_loaders(
        data_dir, batch_size=batch_size, flatten=True
    )

    net = MLPEncoder()
    print(count_parameters(net))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # return 0

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3,
                                 weight_decay=1e-5)

    # train
    for epoch in range(num_epochs):
        print('epch = {}'.format(epoch))

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (
                    epoch + 1, i + 1, running_loss / 10
                ))
                running_loss = 0.0

    torch.save(net.state_dict(), './myfashionmodel.pth')

    # test
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_dataloader, 0):
            if i > 40:
                break
            print('testing batch {}'.format(i))
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            fig = plt.figure()
            gs = plt.GridSpec(1, 2)
            ax1 = plt.subplot(gs[0])
            ax1.imshow(outputs[0].reshape(28, 28))
            ax2 = plt.subplot(gs[1])
            ax2.imshow(images[0].reshape(28, 28))
            figname = 'image_batch_{:04d}.pdf'.format(i)
            plt.savefig(figname, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
