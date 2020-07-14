'''
plot star-galaxy images
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ptlib.dataloaders import StarGalaxyDataManager as DataManager

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=6, type=int, help='batch size')
parser.add_argument('--data-dir', default='', type=str, help='data dir')
parser.add_argument('--num-batches', default=1, type=int,
                    help='number of batches')
parser.add_argument('--pdf-name', default='evt_all.pdf', type=str,
                    help='output pdf name')


def main(batch_size, data_dir, num_batches, pdf_name):
    data_manager = DataManager(data_dir=data_dir)
    # TODO - fix hack for file assignment to look at attacked images
    # data_manager.testfile = './fgsm_0_000.hdf5'
    data_manager.testfile = './fgsm_0_050.hdf5'
    print(data_manager.testfile)
    _, _, test_dl = data_manager.get_data_loaders(
        batch_size=batch_size, standardize=False)
    with PdfPages(pdf_name) as pdf:
        for iter_num, (inputs, labels) in enumerate(test_dl, 0):
            if iter_num >= num_batches:
                break
            n_cols = int(np.sqrt(batch_size))
            n_rows = int(np.ceil(batch_size / n_cols))
            # make grid, plots
            gs = plt.GridSpec(n_rows, n_cols)
            for evt, img_tnsr in enumerate(inputs):
                ax = plt.subplot(gs[evt])
                ax.axis('on')
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())
                _ = ax.imshow(
                    np.moveaxis(img_tnsr.numpy().astype('uint8'), 0, -1))
                plt.title(data_manager.label_names[labels[evt]])
            pdf.savefig()


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
