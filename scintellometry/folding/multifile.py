from __future__ import division

import numpy as np
from astropy.table import Table


class multifile(object):
    def __init__(self, sequence_file, raw_voltage_files, recsize=2**25):
        self.sequence_file = sequence_file
        self.sequence = Table(np.loadtxt(sequence_file, np.int32),
                              names=['seq', 'raw'])
        self.sequence.sort('seq')
        self.fh_raw = [open(raw, 'rb') for raw in raw_voltage_files]
        self.recsize = recsize
        self.index = 0
        self.seq0 = self.sequence['seq'][0]

    def seek(self, offset):
        assert offset % self.recsize == 0
        self.index = offset // self.recsize

    def close(self):
        for fh in self.fh_raw:
            fh.close()

    def read(self, size):
        assert size == self.recsize
        if self.index == len(self.sequence):
            raise EOFError
        self.index += 1
        # so far, only works for continuous data, so ensure we're not missing
        # any sequence numbers
        if self.sequence['seq'][self.index-1] != self.index-1 + self.seq0:
            raise IOError("multifile sequence numbers have to be consecutive")
        i = self.sequence['raw'][self.index-1]
        return self.fh_raw[i].read(size)

    def __repr__(self):
        return ("<open multifile raw_voltage_files {} "
                "using sequence file '{}' at index {}>"
                .format(self.fh_raw, self.sequence_file, self.index))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
