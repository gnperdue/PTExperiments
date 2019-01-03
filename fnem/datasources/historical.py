import gzip
import torch


class HistoricalData(object):

    def __init__(self, source_file, pytorch=True):
        self._file = source_file
        self._is_zipped = str(self._file)[-3:] == '.gz'
        self._pytorch = pytorch
        if self._is_zipped:
            self._open_fn = gzip.open
        else:
            self._open_fn = open

    def _parse_line(self, line):
        fields = line.decode('utf8').strip().split(',')
        t = float(fields[0])
        setting = float(fields[1])
        measured_sensors = [float(x) for x in fields[2:6]]
        true_sensors = [float(x) for x in fields[6:]]
        true_total = sum(true_sensors)
        heat = true_total - setting
        state = measured_sensors + [heat, setting, t]
        if self._pytorch:
            state = torch.Tensor(state)
            true_sensors = torch.Tensor(true_sensors)
        return (state, true_sensors)

    def __iter__(self):
        # for i in range(self.nepochs):?
        with self._open_fn(self._file, 'rb') as fp:
            for line in fp:
                yield self._parse_line(line)
            return
