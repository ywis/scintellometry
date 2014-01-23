""" classes to handle ARO, LOFAR, and GMRT data in a consistent way """

from __future__ import division

import numpy as np
import os
import re
import warnings

from scipy.fftpack import fftfreq, fftshift
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
def dtype_itemsize(dtype):
    bps = {'ci1': 2, '4bit': 0.5}.get(dtype, None)
    if bps is None:
        bps = np.dtype(dtype).itemsize
    return bps

# default properties for various telescopes
header_defaults = {}

# hdf5 dtype conversion
_lofar_dtypes = {'float': '>c8', 'int8': 'ci1'}


class MultiFile(psrFITS):

    def __init__(self, files=None, blocksize=None, dtype=None, nchan=None,
                 comm=None):
        if comm is None:
            self.comm = MPI.COMM_SELF
        else:
            self.comm = comm
        if files is not None:
            self.open(files)
        # parameters for fold:
        if blocksize is not None:
            self.blocksize = blocksize
        if dtype is not None:
            self.dtype = dtype
        if nchan is not None:
            self.nchan = nchan
        self.itemsize = dtype_itemsize(self.dtype)
        self.recordsize = self.itemsize * self.nchan
        assert self.blocksize % self.recordsize == 0
        self.setsize = int(self.blocksize / self.recordsize)

        super(MultiFile, self).__init__(hdus=['SUBINT'])
        self.set_hdu_defaults(header_defaults[self.telescope])

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
        self.offset = 0

    def close(self):
        for fh in self.fh_raw:
            fh.Close()
        for fh in self.fh_links:
            if os.path.exists(fh):
                os.unlink(fh)

    def read(self, size):
        """Read at most size bytes, returning an ndarray with np.int8 dtype.
        Incorporate information from multiple underlying files if necessary"""
        if size % self.recordsize != 0:
            raise ValueError("Cannot read a non-integer number of records")

        # ensure we do not read beyond end
        size = min(size, len(self.indices) * self.blocksize - self.offset)
        if size <= 0:
            raise EOFError('At end of file in MultiFile.read')

        # allocate buffer for MPI read
        z = np.empty(size, dtype=np.int8)
        # read one or more pieces
        iz = 0
        while(iz < size):
            block, already_read = divmod(self.offset, self.blocksize)
            fh_size = min(size - iz, self.blocksize - already_read)
            fh_index = self.indices[block]
            if fh_index >= 0:
                self.fh_raw[fh_index].Iread(z[iz:iz+fh_size])
            else:
                z[iz:iz+fh_size] = 0
            self.offset += fh_size
            iz += fh_size

        return z

    def seek(self, offset):
        """Move filepointers to given offset

        Parameters
        ----------
        offset : float, Quantity, TimeDelta, Time, or str (iso-t)
            If float, in units of bytes
            If Quantity in time units or TimeDelta, interpreted as offset from
                start time, and converted to nearest record
            If Time, calculate offset from start time and convert
        """
        if isinstance(offset, Time):
            offset = offset-self.time0
        elif isinstance(offset, str):
            offset = Time(offset, scale='utc') - self.time0

        try:
            offset = offset.to(self.dtsample.unit)
        except AttributeError:
            pass
        except u.UnitsError:
            offset = offset.to(u.byte).value
        else:
            offset = (offset/self.dtsample).to(u.dimensionless_unscaled)
            offset = int(round(offset)) * self.recordsize
        self._seek(offset)

    def _seek(self, offset):
        if offset % self.recordsize != 0:
            raise ValueError("Cannot offset to non-integer number of records")
        # determine index in units of the blocksize
        block, extra = divmod(offset, self.blocksize)
        indices = self.indices[:block]
        fh_offsets = np.bincount(indices[indices >= 0],
                                 minlength=len(self.fh_raw)) * self.blocksize
        if block > len(self.indices):
            raise EOFError('At end of file in MultiFile.read')
        if self.indices[block] >= 0:
            fh_offsets[self.indices[block]] += extra
        for fh, fh_offset in zip(self.fh_raw, fh_offsets):
            fh.Seek(fh_offset)
        self.offset = offset

    def tell(self, offset=None, unit=None):
        if offset is None:
            offset = self.offset

        if unit is None:
            return offset

        if isinstance(unit, str) and unit == 'time':
            return self.time()

        return (offset * u.byte).to(
            unit, equivalencies=[(u.Unit(self.recordsize * u.byte),
                                  u.Unit(self.dtsample))])

    def time(self, offset=None):
        """Get time corresponding to the current (or given) offset"""
        if offset is None:
            offset = self.offset
        if offset % self.recordsize != 0:
            warnings.warn("Offset for which time is requested is not "
                          "integer multiple of record size.")
        return self.time0 + self.tell(offset, u.day)

    # ARO and GMRT (LOFAR_Pcombined overwrites this)
    def seek_record_read(self, offset, count):
        """Read count samples starting from offset (also in samples)"""
        self.seek(offset)
        return self.record_read(count)

    def record_read(self, count):
        return fromfile(self, self.dtype,
                        self.blocksize).reshape(-1, self.nchan).squeeze()

    def nskip(self, date, time0=None):
        """
        Return the number of records needed to skip from start of
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

    def __init__(self, sequence_file, raw_voltage_files, blocksize=2**25,
                 dtype='4bit', samplerate=200.*u.MHz, comm=None):

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
            # ARO time is off by two 32MiB record or 128Misamples
            self.time0 -= (2.**27/samplerate).to(u.s)
        else:
            self.time0 = None

        self.fedge = 200. * u.MHz
        self.fedge_at_top = True
        self.samplerate = samplerate
        self.dtsample = (1./samplerate).to(u.s)

        super(AROdata, self).__init__(raw_voltage_files, blocksize, dtype, 1,
                                      comm=comm)
        # update headers for fun
        self['PRIMARY'].header['DATE-OBS'] = self.time0.iso
        self[0].header.update('TBIN', (1./samplerate).to('s').value),

    def ntint(self, nchan):
        """
        number of samples in a frequency bin
        this is baseband data so need to know number of channels we're making
        """
        return self.setsize // (2*nchan)

    def __repr__(self):
        return ("<open MultiFile raw_voltage_files {} "
                "using sequence file '{}' at offset {}>"
                .format(self.fh_raw, self.sequence_file, self.offset))


header_defaults['aro'] = {
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

    def __init__(self, raw_files, comm=None, blocksize=2**16*20*2*4):
        """
        Initialize a lofar observation, tracking/joining the two polarizations.
        We also parse the corresponding HDF5 files to initialize:
        nchan, samplerate, fwidth
        """
        # read the HDF5 file and get useful data
        h0 = HDF5File(raw_files[0].replace('.raw', '.h5'), 'r')
        saps = sorted([i for i in h0.keys() if 'SUB_ARRAY_POINTING' in i])
        s0 = h0[saps[0]]
        time0 = Time(s0.attrs['EXPTIME_START_UTC'].replace('Z',''),
                     scale='utc')

        beams = sorted([i for i in s0.keys() if 'BEAM' in i])
        b0 = s0[beams[0]]
        frequencies = (b0['COORDINATES']['COORDINATE_1']
                       .attrs['AXIS_VALUES_WORLD'] * u.Hz).to(u.MHz)
        fbottom = frequencies[0]

        stokes = sorted([i for i in b0.keys()
                         if 'STOKES' in i and 'i2f' not in i])
        st0 = b0[stokes[0]]
        dtype = _lofar_dtypes[st0.attrs['DATATYPE']]

        nchan = len(frequencies)  # = st0.attrs['NOF_SUBBANDS']

        # can also get from np.diff(frequencies.diff).mean()
        fwidth = (b0.attrs['SUBBAND_WIDTH'] *
                  u.__dict__[b0.attrs['CHANNEL_WIDTH_UNIT']]).to(u.MHz)

        samplerate = (b0.attrs['SAMPLING_RATE'] *
                      u.__dict__[b0.attrs['SAMPLING_RATE_UNIT']]).to(u.MHz)
        h0.close()

        self.time0 = time0
        self.samplerate = samplerate
        self.fwidth = fwidth
        self.frequencies = frequencies
        self.fedge = fbottom
        self.fedge_at_top = False
        self.dtsample = (1./self.fwidth).to(u.s)

        super(LOFARdata, self).__init__(raw_files, blocksize, dtype, nchan,
                                        comm=comm)
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
        read 'size' bytes of the LOFAR data; returns the two streams
        interleaved, such that one has complex numbers (either complex64
        or ci1, i.e., using two signed one-byte integers).
        """
        z = np.empty(size, dtype='i1').reshape(2, -1, self.itemsize//2)
        for fh, buf in zip(self.fh_raw, z):
            fh.Iread([buf, MPI.BYTE])
        return z.transpose(1, 0, 2).ravel()
        self.offset += size

    def _seek(self, offset):
        """Offset by the given number of bytes"""
        if offset % self.recordsize != 0:
            raise ValueError("Cannot offset to non-integer number of records")
        # half the total number of corresponding bytes in each file
        assert offset % 2 == 0
        for fh in self.fh_raw:
            fh.Seek(offset // 2)
        self.offset = offset

    def __repr__(self):
        return ("<open lofar polarization pair {}>"
                .format(self.fh_raw))


class LOFARdata_Pcombined(MultiFile):
    """
    convenience class to combine multiple subbands, making them act
    as a single file.
    """
    telescope = 'lofar'

    def __init__(self, raw_files_list, comm=None):
        """
        A list of tuples, to be 'concatenated' together
        (as returned by observations.obsdata[telescope].file_list(obskey) )
        """
        super(LOFARdata_Pcombined, self).__init__(raw_files_list, comm=comm)
        self.fbottom = self.frequencies[0]
        self.fedge = self.frequencies[0]
        self.fedge_at_top = False
        # update some of the hdu data
        self['PRIMARY'].header['DATE-OBS'] = self.time0.isot
        self['PRIMARY'].header.update('TBIN',
                                      (1./self.samplerate).to('s').value)
        self['PRIMARY'].header.update('NCHAN', self.nchan)

    def open(self, raw_files_list):
        self.fh_raw = [LOFARdata(raw_files, comm=self.comm)
                       for raw_files in raw_files_list]
        self.fh_links = []
        # make sure basic properties of the files are the same
        for prop in ['dtype', 'itemsize', 'recordsize', 'time0', 'samplerate',
                     'fwidth', 'dtsample']:
            props = [fh.__dict__[prop] for fh in self.fh_raw]
            if prop == 'time0':
                props = [p.isot for p in props]
            assert len(set(props)) == 1
            self.__setattr__(prop, self.fh_raw[0].__dict__[prop])

        self.blocksize = sum([fh.blocksize for fh in self.fh_raw])
        self.recordsize = sum([fh.recordsize for fh in self.fh_raw])
        self.frequencies = u.Quantity([fh.frequencies
                                       for fh in self.fh_raw]).ravel()
        self.nchan = len(self.frequencies)
        self.offset = 0

    def close(self):
        for fh in self.fh_raw:
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
        assert size % len(self.fh_raw) == 0
        raw = np.hstack([fh.record_read(size // len(self.fh_raw))
                         for fh in self.fh_raw])
        self.offset += size
        return raw

    def _seek(self, offset):
        assert offset % len(self.fh_raw) == 0
        for fh in self.fh_raw:
            fh._seek(offset // len(self.fh_raw))
        self.offset = offset

    def seek_record_read(self, offset, size):
        """
        LOFARdata_Pcombined class opens a lot of filehandles.
        This routine tries to minimize file seeks
        """
        nfh = len(self.fh_raw)
        assert offset % nfh == 0 and size % nfh == 0
        raw = np.hstack([fh.seek_record_read(offset // nfh, size // nfh)
                         for fh in self.fh_raw])
        self.offset = offset + size
        return raw

    def __repr__(self):
        return ("<open (concatenated) lofar polarization pair from {} to {}>"
                .format(self.fh_raw[0].fh_raw[0], self.fh_raw[-1].fh_raw[-1]))

# LOFAR defaults for psrfits HDUs
header_defaults['lofar'] = {
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

    def __init__(self, timestamp_file,
                 raw_files, blocksize=2**22, dtype='ci1', nchan=512,
                 utc_offset=5.5*u.hr,
                 samplerate=100./3.*u.MHz, fedge=156.*u.MHz,
                 fedge_at_top=True, comm=None):
        # GMRT phased data stored in 4 MiB blocks with 2Mi complex samples
        # split in 512 channels
        # ci1 is special complex type, made of two signed int8s.
        self.samplerate = samplerate
        self.fedge = fedge
        self.fedge_at_top = fedge_at_top
        f = fftshift(fftfreq(nchan, (2./samplerate).to(u.s).value)) * u.Hz
        if fedge_at_top:
            self.frequencies = fedge - (f-f[0])
        else:
            self.frequencies = fedge + (f-f[0])

        self.timestamp_file = timestamp_file

        self.indices, self.timestamps, self.gsb_start = read_timestamp_file(
            timestamp_file, utc_offset)
        self.time0 = self.timestamps[0]
        # GMRT time is off by one 32MB record
        self.time0 -= (2.**25/samplerate).to(u.s)

        self.dtsample = (nchan * 2 / samplerate).to(u.s)

        super(GMRTdata, self).__init__(raw_files, blocksize, dtype, nchan,
                                       comm=comm)

    def ntint(self, nchan):
        return self.setsize

    def __repr__(self):
        return ("<open two raw_voltage_files {} "
                "using timestamp file '{}' at index {} (time {})>"
                .format(self.fh_raw, self.timestamp_file, self.offset,
                        self.time()))

# GMRT defaults for psrfits HDUs
# Note: these are largely made-up at this point
header_defaults['gmrt'] = {
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
