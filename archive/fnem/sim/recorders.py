import gzip
import shutil
import os


class MachineStateTextRecorder(object):
    '''
    record time, machine setting, sensor values, true values - log values to
    a .csv file and gzip the file after calling `close()`.
    '''

    def __init__(self, log_base_name):
        self.log_name = log_base_name + '.csv'
        self.gzfile = self.log_name + '.gz'
        self.cleanup_files()

    def cleanup_files(self):
        for f in [self.log_name, self.gzfile]:
            if os.path.isfile(f):
                os.remove(f)

    def write_data(self, t, setting, measurements, targets):
        '''
        write t, the machine setting, the measurements and the targets (true
        settings). measurements are affected by noise, targets are true values.
        'heat' can be inferred as the difference between the sum of the true
        values and the setting.
        '''
        with open(self.log_name, 'ab+') as f:
            meas_string = ','.join([str(i) for i in measurements])
            targ_string = ','.join([str(i) for i in targets])
            msg = str(t) + ',' + str(setting) + ',' + meas_string + \
                ',' + targ_string + '\n'
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
        '''
        zip the log file
        '''
        with open(self.log_name, 'rb') as f_in:
            with gzip.open(self.gzfile, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        if os.path.isfile(self.gzfile) and (os.stat(self.gzfile).st_size > 0):
            os.remove(self.log_name)
        else:
            raise IOError('Compressed file not produced!')
