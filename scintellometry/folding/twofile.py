class twofile(object):
    def __init__(self, files, recsize=2**22):
        self.fh_raw = [open(raw, 'rb') for raw in files]
        self.recsize = recsize
        self.index = 0

    def seek(self, offset):
        assert offset % self.recsize == 0
        self.index = offset // self.recsize
        half = self.index // 2
        self.fh_raw[0].seek((half + self.index % 2) * self.recsize)
        self.fh_raw[1].seek(half * self.recsize)

    def close(self):
        for fh in self.fh_raw:
            fh.close()

    def read(self, size):
        assert size == self.recsize
        i = self.index % 2
        self.index += 1
        return self.fh_raw[i].read(size)

    def __repr__(self):
        return ("<open two raw_voltage_files {} at index {}>"
                .format(self.fh_raw, self.index))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
