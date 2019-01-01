import gzip
import torch
from torch.utils.data import Dataset, DataLoader


class ToTensor(object):
    '''transform for moving historical data to tensors'''

    def __call__(self, sample):
        '''todo - this is legacy image code...'''
        image, label = sample['image'], sample['label']
        return {
            'image': torch.from_numpy(image).float(),
            'label': torch.argmax(
                torch.from_numpy(label).type(torch.LongTensor)
            )
        }


class HistoricalDataset(Dataset):

    def __init__(self, source_file, transform=None):
        super(HistoricalDataset, self).__init__()
        self._file = source_file
        self.is_zipped = self._file[-3:] == '.gz'
        self.transform = transform
        if self.is_zipped:
            self.open_fn = gzip.open
        else:
            self.open_fn = open

    def _parse_line(self, line):
        if self.transform:
            pass
        else:
            pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        with self.open_fn(self._file, 'r') as fp:
            for line in fp:
                yield self._parse_line(line)
