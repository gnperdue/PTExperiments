import gzip
import shutil
import os


class MachineStateTextRecorder(object):
    '''
    record observation values to a .csv file and gzip after calling `close()`.
    '''

    def __init__(self, log_base_name):
        self.log_name = log_base_name
        self.gzfile = self.log_name + '.gz'
        self._cleanup_files()

    def _cleanup_files(self):
        for f in [self.log_name, self.gzfile]:
            if os.path.isfile(f):
                os.remove(f)

    def write_data(self, data):
        with open(self.log_name, 'ab+') as f:
            msg = ','.join([str(i) for i in data]) + '\n'
            f.write(bytes(msg, 'utf8'))
        return True

    def read_data(self):
        '''
        do not call this on large files (we just read it all).
        NOTE: we are assuming gzip compression has occurred!
        '''
        with gzip.open(self.log_name + '.gz', 'rb') as f:
            content = f.readlines()
            content = [x.decode('utf8').strip() for x in content]
        return content

    def close(self):
        '''zip the log file'''
        with open(self.log_name, 'rb') as f_in:
            with gzip.open(self.gzfile, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        if os.path.isfile(self.gzfile) and (os.stat(self.gzfile).st_size > 0):
            os.remove(self.log_name)
        else:
            raise IOError('Compressed file not produced!')
