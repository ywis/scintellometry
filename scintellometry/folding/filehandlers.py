""" classes to handle ARO, LOFAR, and GMRT data in a consistent way """

from __future__ import division

import numpy as np
import os
import re

from astropy import units as u
from astropy.time import Time, TimeDelta
from fromfile import fromfile
from h5py import File as HDF5File
try:
    from mpi4py import MPI
except ImportError:
    pass
from psrfits_tools import psrFITS


# size in bytes of records read from file (simple for ARO: 1 byte/sample)
# double since we need to get ntint samples after FFT
def bytes_per_sample(dtype):
    bps = {'ci1': 2, '4bit': 0.5}.get(dtype, None)
    if bps is None:
        bps = np.dtype(dtype).itemsize
    return bps

# default properties for various telescopes
_defaults = {}

# hdf5 dtype conversion
_lofar_dtypes = {'float':'>f4', 'int8':'>i1'}


class MultiFile(psrFITS):

    def __init__(self, files=None, comm=None):
        super(MultiFile, self).__init__(hdus=['SUBINT'])
        self.set_hdu_defaults(_defaults[self.telescope])
        if comm is None:
            self.comm = MPI.COMM_SELF
        else:
            self.comm = comm
        if files is not None:
            self.open(files)

    def set_hdu_defaults(self, dictionary):
        for hdu in dictionary:
            self[hdu].header.update(dictionary[hdu])

    def open(self, files):
        # MPI.File.Open doesn't handle files with ":"
        self.fh_raw = []
        self.fh_links = []
        for raw in files:
            fname, islnk = good_name(os.path.abspath(raw))
            self.fh_raw.append(MPI.File.Open(self.comm, fname,
                                             amode=MPI.MODE_RDONLY))
            if islnk:
                self.fh_links.append(fname)

    def close(self):
        for fh in self.fh_raw:
            fh.Close()
        for fh in self.fh_links:
            if os.path.exists(fh):
                os.unlink(fh)

    def seek(self, offset):
        assert offset % self.setsize == 0
        self.index = offset // self.setsize
        for i, fh in enumerate(self.fh_raw):
            fh.Seek(np.count_nonzero(self.indices[:self.index] == i) *
                    self.recsize)

    def read(self, size):
        assert size == self.recsize
        if self.index == len(self.indices):
            raise EOFError
        self.index += 1
        z = np.zeros(size, dtype='i1')
        self.fh_raw[self.indices[self.index-1]].Iread(z)
        return z

    # ARO and GMRT (LOFAR_Pcombined overwrites this)
    def seek_record_read(self, offset, count):
        """Read count samples starting from offset (also in samples)"""
        self.seek(offset)
        return self.record_read(count)

    def record_read(self, count):
        return fromfile(self, self.dtype,
                        self.recsize).reshape(-1, self.nchan).squeeze()

    def nskip(self, date, time0=None):
        """
        return the number of records needed to skip from start of
        file to iso timestamp 'date'.

        Optionally:
        time0 : use this start time instead of self.time0
                either a astropy.time.Time object or string in 'utc'
        """
        time0 = self.time0 if time0 is None else Time(time0, scale='utc')
        dt = Time(date, scale='utc') - time0
        nskip = int(round((dt / self.dtsample / self.setsize)
                          .to(u.dimensionless_unscaled)))
        return nskip

    def ntimebins(self, t0, t1):
        """
        determine the number of timebins between UTC start time 't0'
        and end time 't1'
        """
        t0 = Time(t0, scale='utc')
        t1 = Time(t1, scale='utc')
        nt = ((t1-t0) / self.dtsample /
              (self.setsize)).to(u.dimensionless_unscaled).value
        return np.ceil(nt).astype(int)

    # for use in context manager ("with <MultiFile> as fh:")
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


#    ____  ____   ___
#   /    ||    \ /   \
#  |  o  ||  D  )     |
#  |     ||    /|  O  |
#  |  _  ||    \|     |
#  |  |  ||  .  \     |
#  |__|__||__|\_|\___/
#
class AROdata(MultiFile):

    telescope = 'aro'

    def __init__(self, sequence_file, raw_voltage_files, setsize=2**26,
                 dtype='4bit', samplerate=200.*u.MHz, comm=None):

        super(AROdata, self).__init__(raw_voltage_files, comm=comm)

        self.sequence_file = sequence_file
        seq, indices = np.loadtxt(sequence_file, np.int32, unpack=True)

        # seq starts counting at 2 for some reason
        seq -= 2
        self.indices = -np.ones(seq.max() + 1, dtype=np.int64)
        self.indices[seq] = indices

        # get start date from sequence filename
        arodate = re.search('\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                            os.path.basename(sequence_file))
        if arodate:
            isot = arodate.group()
            # convert time to UTC; dates given in EDT
            self.time0 = Time(isot, scale='utc') + 4*u.hr
            self['PRIMARY'].header['DATE-OBS'] = self.time0.iso
            # ARO time is off by two 32MiB record or 128Misamples
            self.time0 -= (2.**27/samplerate).to(u.s)
        else:
            self.time0 = None

        self.index = 0

        # defaults expected by fold.py
        self.dtype = dtype
        self.itemsize = bytes_per_sample(dtype)
        self.setsize = setsize  # number of samples
        assert (setsize * self.itemsize) % 1 == 0
        self.recsize = int(setsize * self.itemsize)  # number of bytes
        self.fedge = 200. * u.MHz
        self.fedge_at_top = True
        self.nchan = 1
        self.samplerate = samplerate
        self.dtsample = (1./samplerate).to(u.s)
        # update headers for fun
        self[0].header.update('TBIN', (1./samplerate).to('s').value),

    def ntint(self, nchan):
        """
        number of samples in a frequency bin
        this is baseband data so need to know number of channels we're making
        """
        return self.setsize // (2*nchan)

    def __repr__(self):
        return ("<open MultiFile raw_voltage_files {} "
                "using sequence file '{}' at index {}>"
                .format(self.fh_raw, self.sequence_file, self.index))


_defaults['aro'] = {
    'PRIMARY': {'TELESCOP': 'Algonquin',
                'IBEAM': 1,
                'FD_POLN': 'LIN',
                'OBS_MODE': 'SEARCH',
                'ANT_X': 0, 'ANT_Y': 0, 'ANT_Z': 0,
                'NRCVR': 1,
                'FD_HAND': 1, 'FD_SANG': 0, 'FD_XYPH': 0,
                'BE_PHASE': 0, 'BE_DCC': 0, 'BE_DELAY': 0,
                'TRK_MODE': 'TRACK',
                'TCYCLE': 0, 'OBSFREQ': 300, 'OBSBW': 100,
                'OBSNCHAN': 20, 'CHAN_DM': 0,
                'EQUINOX': 2000.0,
                'BMAJ': 1, 'BMIN': 1, 'BPA': 0,
                'SCANLEN': 1, 'FA_REQ': 0,
                'CAL_FREQ': 0, 'CAL_DCYC': 0,
                'CAL_PHS': 0, 'CAL_NPHS': 0,
                'STT_IMJD': 54000, 'STT_SMJD': 0, 'STT_OFFS': 0},
    'SUBINT': {'INT_TYPE': 'TIME',
               'SCALE': 'FluxDen',
               'POL_TYPE': 'AABB',
               'NPOL': 1,
               'NBIN': 1,
               'NBIN_PRD': 1,
               'PHS_OFFS': 0,
               'NBITS': 1,
               'ZERO_OFF': 0,
               'SIGNINT': 0,
               'NSUBOFFS': 0,
               'NCHAN': 1,
               'CHAN_BW': 1,
               'DM': 0, 'RM': 0,
               'NCHNOFFS': 0,
               'NSBLK': 1}}


#   _       ___   _____   ____  ____
#  | |     /   \ |     | /    ||    \
#  | |    |     ||   __||  o  ||  D  )
#  | |___ |  O  ||  |_  |     ||    /
#  |     ||     ||   _] |  _  ||    \
#  |     ||     ||  |   |  |  ||  .  \
#  |_____| \___/ |__|   |__|__||__|\_|
#
class LOFARdata(MultiFile):

    telescope = 'lofar'

    def __init__(self, fname1, fname2, comm=None, setsize=2**16):
        """
        Initialize a lofar observation, tracking/joining the two polarizations.
        We also parse the corresponding HDF5 files to initialize:
        nchan, samplerate, fwidth
        """
        super(LOFARdata, self).__init__([fname1, fname2], comm=comm)

        # read the HDF5 file and get useful data
        h0 = HDF5File(fname1.replace('.raw', '.h5'), 'r')
        saps = sorted([i for i in h0.keys() if 'SUB_ARRAY_POINTING' in i])
        s0 = h0[saps[0]]
        time0 = Time(s0.attrs['EXPTIME_START_UTC'].replace('Z',''),
                     scale='utc')

        beams = sorted([i for i in s0.keys() if 'BEAM' in i])
        b0 = s0[beams[0]]
        freqs = (b0['COORDINATES']['COORDINATE_1']
                 .attrs['AXIS_VALUES_WORLD'] * u.Hz).to(u.MHz)
        fbottom = freqs[0]

        stokes = sorted([i for i in b0.keys()
                         if 'STOKES' in i and 'i2f' not in i])
        st0 = b0[stokes[0]]
        dtype = st0.attrs['DATATYPE']

        nchan = len(freqs)  # = st0.attrs['NOF_SUBBANDS']

        # can also get from np.diff(freqs.diff).mean()
        fwidth = (b0.attrs['SUBBAND_WIDTH'] *
                  u.__dict__[b0.attrs['CHANNEL_WIDTH_UNIT']]).to(u.MHz)

        samplerate = (b0.attrs['SAMPLING_RATE'] *
                      u.__dict__[b0.attrs['SAMPLING_RATE_UNIT']]).to(u.MHz)
        h0.close()

        # defaults expected by fold.py
        self.time0 = time0
        self.dtype = _lofar_dtypes[dtype]
        self.itemsize = bytes_per_sample(self.dtype)
        if self.itemsize is None:
            self.itemsize = np.dtype(self.dtype).itemsize

        self.samplerate = samplerate
        self.nchan = nchan
        self.fwidth = fwidth  # = samplerate = np.diff(freqs).mean()
        self.setsize = setsize
        self.recsize = self.itemsize * nchan * setsize
        self.freqs = freqs
        self.fedge = fbottom
        self.fedge_at_top = False
        self.dtsample = (1./self.fwidth).to(u.s)
        # update some of the hdu data
        self['PRIMARY'].header['DATE-OBS'] = self.time0.isot
        self[0].header.update('TBIN', (1./samplerate).to('s').value)

    def ntint(self, nchan):
        """
        number of samples in an integration
        Lofar data is already channelized so we assert
        nchan is the same
        """
        assert(nchan == self.nchan)
        return self.setsize

    def read(self, size):
        """
        read 'size' bytes of the LOFAR data
        returns a tuple from the two polarizations
        """
        z1 = np.zeros(size, dtype='i1')
        self.fh_raw[0].Iread([z1, MPI.BYTE])
        z2 = np.zeros(size, dtype='i1')
        self.fh_raw[1].Iread([z2, MPI.BYTE])
        return z1, z2

    def record_read(self, count):
        """
        read 'count' records of data,
        returned as a complex number
        """
        raw = [np.fromstring(r, dtype=self.dtype).reshape(-1, self.nchan)
               for r in self.read(count * self.nchan * self.itemsize)]
        return raw[0] + 1j*raw[1]

    def seek(self, offset):
        for fh in self.fh_raw:
            fh.Seek(offset * self.itemsize * self.nchan)

    def __repr__(self):
        return ("<open lofar polarization pair {}>"
                .format(self.fh_raw))


class LOFARdata_Pcombined(MultiFile):
    """
    convenience class to combine multiple subbands, making them act
    as a single file.

    """
    telescope = 'lofar'
    samplerate = 200.*u.MHz

    def __init__(self, filelist, comm=None):
        """
        A list of tuples, to be 'concatenated' together
        (as returned by observations.obsdata[telescope].file_list(obskey) )
        """
        super(LOFARdata_Pcombined, self).__init__(comm=comm)
        self.filelist = filelist

        self.fh_subbands = []
        for filetuple in filelist:
            self.fh_subbands.append(LOFARdata(*filetuple, comm=self.comm))

        # make sure basic properties of the files are the same
        for prop in ['dtype', 'setsize', 'time0', 'samplerate',
                     'fwidth', 'dtsample', 'nchan']:
            props = [fh.__dict__[prop] for fh in self.fh_subbands]
            if prop == 'time0':
                props = [p.isot for p in props]
            assert len(set(props)) == 1
            self.__setattr__(prop, self.fh_subbands[0].__dict__[prop])

        self.recsize = sum([fh.recsize for fh in self.fh_subbands])

        self.itemsize = bytes_per_sample(self.dtype)

        freqs = np.concatenate([fh.freqs.to(u.MHz).value
                                for fh in self.fh_subbands])*u.MHz
        self.freqs = freqs
        self.nchans = [fh.nchan for fh in self.fh_subbands]
        self.nchan = freqs.value.size
        self.fbottom = freqs[0]
        self.fedge = freqs[0]
        self.fedge_at_top = False
        # update some of the hdu data
        self['PRIMARY'].header['DATE-OBS'] = self.time0.isot
        self['PRIMARY'].header.update('TBIN',
                                      (1./self.samplerate).to('s').value)
        self['PRIMARY'].header.update('NCHAN', self.nchan)

    def close(self):
        for fh in self.fh_subbands:
            fh.close()

    def ntint(self, *args):
        """
        number of samples in an integration
        Lofar data is already channelized so we assert
        nchan is the same
        """
        # LOFAR is already channelized, we accept args for generalization
        return self.setsize

    def record_read(self, size):
        return np.hstack([fh.record_read(size) for fh in self.fh_subbands])

    def seek(self, offset):
        for fh in self.fh_subbands:
            fh.seek(offset)

    def seek_record_read(self, offset, size):
        """
        LOFARdata_Pcombined class opens a lot of filehandles.
        This routine tries to minimize file seeks
        """
        return np.hstack([fh.seek_record_read(offset, size)
                          for fh in self.fh_subbands])

    def __repr__(self):
        return ("<open (concatenated) lofar polarization pair from {} to {}>"
                .format(self.fh_subbands[0].fh_raw[0],
                        self.fh_subbands[-1].fh_raw[-1]))

# LOFAR defaults for psrfits HDUs
_defaults['lofar'] = {
    'PRIMARY': {'TELESCOP':'LOFAR',
                'IBEAM':1, 'FD_POLN':'LIN',
                'OBS_MODE':'SEARCH',
                'ANT_X':0, 'ANT_Y':0, 'ANT_Z':0, 'NRCVR':1,
                'FD_HAND':1, 'FD_SANG':0, 'FD_XYPH':0,
                'BE_PHASE':0, 'BE_DCC':0, 'BE_DELAY':0,
                'TRK_MODE':'TRACK',
                'TCYCLE':0, 'OBSFREQ':300, 'OBSBW':100,
                'OBSNCHAN':20, 'CHAN_DM':0,
                'EQUINOX':2000.0, 'BMAJ':1, 'BMIN':1, 'BPA':0,
                'SCANLEN':1, 'FA_REQ':0,
                'CAL_FREQ':0, 'CAL_DCYC':0,
                'CAL_PHS':0, 'CAL_NPHS':0,
                'STT_IMJD':54000, 'STT_SMJD':0, 'STT_OFFS':0},
    'SUBINT': {'INT_TYPE': 'TIME',
               'SCALE': 'FluxDen',
               'POL_TYPE': 'AABB',
               'NPOL':1,
               'NBIN':1, 'NBIN_PRD':1,
               'PHS_OFFS':0,
               'NBITS':1,
               'ZERO_OFF':0, 'SIGNINT':0,
               'NSUBOFFS':0,
               'NCHAN':1,
               'CHAN_BW':1,
               'DM':0, 'RM':0, 'NCHNOFFS':0,
               'NSBLK':1}}


#    ____  ___ ___  ____  ______
#   /    ||   |   ||    \|      |
#  |   __|| _   _ ||  D  )      |
#  |  |  ||  \_/  ||    /|_|  |_|
#  |  |_ ||   |   ||    \  |  |
#  |     ||   |   ||  .  \ |  |
#  |___,_||___|___||__|\_| |__|
#
class GMRTdata(MultiFile):

    telescope = 'gmrt'

    def __init__(self, timestamp_file, raw_files, setsize=2**12, dtype='ci1',
                 utc_offset=5.5*u.hr,
                 samplerate=100./3.*u.MHz, fedge=156.*u.MHz,
                 fedge_at_top=True, nchan=512, comm=None):
        # GMRT phased data stored in 4 MiB blocks with 2Mi complex samples
        # split in 512 channels
        super(GMRTdata, self).__init__(raw_files, comm=comm)

        # parameters for fold:
        self.setsize = setsize
        # ci1 is special complex type, made of two signed int8s.
        self.dtype = dtype
        self.samplerate = samplerate
        self.fedge = fedge
        self.fedge_at_top = fedge_at_top
        self.nchan = nchan

        self.timestamp_file = timestamp_file
        self.index = 0

        self.indices, self.timestamps, self.gsb_start = read_timestamp_file(
            timestamp_file, utc_offset)
        self.time0 = self.timestamps[0]
        # GMRT time is off by one 32MB record
        self.time0 -= (2.**25/samplerate).to(u.s)

        self.itemsize = bytes_per_sample(self.dtype)
        self.recsize = setsize * self.itemsize * self.nchan
        self.dtsample = (nchan * 2 / samplerate).to(u.s)

    def ntint(self, nchan):
        return self.setsize

    @property
    def time(self):
        return self.timestamps[self.index]

    def __repr__(self):
        return ("<open two raw_voltage_files {} "
                "using timestamp file '{}' at index {} (time {})>"
                .format(self.fh_raw, self.timestamp_file, self.index,
                        self.timestamps[self.index]))

# GMRT defaults for psrfits HDUs
# Note: these are largely made-up at this point
_defaults['gmrt'] = {
    'PRIMARY': {'TELESCOP':'GMRT',
                'IBEAM':1, 'FD_POLN':'LIN',
                'OBS_MODE':'SEARCH',
                'ANT_X':0, 'ANT_Y':0, 'ANT_Z':0, 'NRCVR':1,
                'FD_HAND':1, 'FD_SANG':0, 'FD_XYPH':0,
                'BE_PHASE':0, 'BE_DCC':0, 'BE_DELAY':0,
                'TRK_MODE':'TRACK',
                'TCYCLE':0, 'OBSFREQ':300, 'OBSBW':100,
                'OBSNCHAN':0, 'CHAN_DM':0,
                'EQUINOX':2000.0, 'BMAJ':1, 'BMIN':1, 'BPA':0,
                'SCANLEN':1, 'FA_REQ':0,
                'CAL_FREQ':0, 'CAL_DCYC':0, 'CAL_PHS':0, 'CAL_NPHS':0,
                'STT_IMJD':54000, 'STT_SMJD':0, 'STT_OFFS':0},
    'SUBINT': {'INT_TYPE': 'TIME',
               'SCALE': 'FluxDen',
               'POL_TYPE': 'AABB',
               'NPOL':1,
               'NBIN':1, 'NBIN_PRD':1,
               'PHS_OFFS':0,
               'NBITS':1,
               'ZERO_OFF':0, 'SIGNINT':0,
               'NSUBOFFS':0,
               'NCHAN':1,
               'CHAN_BW':1,
               'DM':0, 'RM':0, 'NCHNOFFS':0,
               'NSBLK':1}}


def read_timestamp_file(filename, utc_offset):
    pc_times = []
    gps_times = []
    ist_utc = TimeDelta(utc_offset)
    prevseq = prevsub = -1
    with open(filename) as fh:
        line = fh.readline()
        while line != '':
            strings = ('{}-{}-{}T{}:{}:{} {} {}-{}-{}T{}:{}:{} {} {} {}'
                       .format(*line.split())).split()
            seq = int(strings[4])
            sub = int(strings[5])
            if prevseq > 0:
                assert seq == prevseq+1
                assert sub == (prevsub+1) % 8
            prevseq, prevsub = seq, sub

            time = (Time([strings[0], strings[2]], scale='utc') +
                    TimeDelta([float(strings[1]), float(strings[3])],
                              format='sec')) - ist_utc
            pc_times += [time[0]]
            gps_times += [time[1]]  # add half a step below

            line = fh.readline()

    indices = np.array(len(gps_times)*[0, 1], dtype=np.int8)

    pc_times = Time(pc_times)
    gps_times = Time(gps_times)
    gps_pc = gps_times - pc_times
    assert np.allclose(gps_pc.sec, gps_pc[0].sec, atol=1.e-3)
    dt = gps_times[1:] - gps_times[:-1]
    assert np.allclose(dt.sec, dt[0].sec, atol=1.e-5)
    gsb_start = gps_times[-1] - seq * dt[0]  # should be whole minute
    assert '00.000' in gsb_start.isot

    timestamps = Time([t + i * dt[0] / 2. for t in gps_times for i in (0,1)])

    return indices, timestamps, gsb_start


#   __ __  ______  ____  _     _____
#  |  |  ||      ||    || |   / ___/
#  |  |  ||      | |  | | |  (   \_
#  |  |  ||_|  |_| |  | | |___\__  |
#  |  :  |  |  |   |  | |     /  \ |
#  |     |  |  |   |  | |     \    |
#   \__,_|  |__|  |____||_____|\___|
#
def good_name(f):
    """
    MPI.File.Open can't process files with colons.
    This routine checks for such cases and creates a well-named
    link to the file.

    Returns (good_name, islink)
    """
    if f is None:
        return f

    fl = f
    newlink = False
    if ':' in f:
        #fl = tempfile.mktemp(prefix=os.path.basename(f)
        # .replace(':','_'), dir='/tmp')
        fl = os.path.join('/tmp', os.path.dirname(f).replace('/','_') +
                          '__' + os.path.basename(f).replace(':','_'))
        if not os.path.exists(fl):
            try:
                os.symlink(f, fl)
            except(OSError):
                pass
            newlink = True
    return fl, newlink
