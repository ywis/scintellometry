""" classes to handle ARO, LOFAR, and GMRT data in a consistent way """

from __future__ import division

import numpy as np
import os
import re

from astropy.table import Table
from astropy import units as u
from astropy.time import Time, TimeDelta
from fromfile import fromfile
from h5py import File as HDF5File
from mpi4py import MPI
from psrfits_tools import psrFITS

from fromfile import fromfile

# size in bytes of records read from file (simple for ARO: 1 byte/sample)
# double since we need to get ntint samples after FFT
_bytes_per_sample = {np.int8: 2, '4bit': 1}

# hdf5 dtype conversion
_lofar_dtypes = {'float':'>f4', 'int8':'>i1'}

class multifile(psrFITS):

    def set_hdu_defaults(self, dictionary):
        for hdu, defs in dictionary.iteritems():
            for card, val in defs.iteritems():
                self[hdu].header.update(card, val)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __enter__(self):
        return self

#    ____  ____   ___  
#   /    ||    \ /   \ 
#  |  o  ||  D  )     |
#  |     ||    /|  O  |
#  |  _  ||    \|     |
#  |  |  ||  .  \     |
#  |__|__||__|\_|\___/ 
#      

class AROdata(multifile):
    def __init__(self, sequence_file, raw_voltage_files, setsize=2**25,
                 dtype='4bit', samplerate=200.*u.MHz, comm=None):
        self.telescope = 'aro'
        super(AROdata, self).__init__(hdus=['SUBINT'])
        self.set_hdu_defaults(_ARO_defs)

        if comm is None:
            self.comm = MPI.COMM_SELF
        else:
            self.comm = comm
        self.sequence_file = sequence_file
        self.sequence = Table(np.loadtxt(sequence_file, np.int32),
                              names=['seq', 'raw'])
        self.sequence.sort('seq')

        # get start date from sequence filename
        arodate = re.search('\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                            os.path.basename(sequence_file))
        if arodate:
            isot = arodate.group()
            # convert time to UTC; dates given in EDT
            self.time0 = Time(isot, scale='utc') + 4*u.hr
            self['PRIMARY'].header['DATE-OBS'] = self.time0.iso
        else:
            self.time0 = None
        # MPI.File.Open doesn't handle files with ":"
        self.fh_raw = []
        self.fh_links = []
        for raw in raw_voltage_files:
            fname, islnk = good_name(os.path.abspath(raw))
            self.fh_raw.append(MPI.File.Open(self.comm, fname,
                                             amode=MPI.MODE_RDONLY))
            if islnk:
                self.fh_links.append(fname)
        self.index = 0
        self.seq0 = self.sequence['seq'][0]

        # defaults expected by fold.py 
        self.dtype = dtype
        self.itemsize = _bytes_per_sample.get(dtype, None)
        if self.itemsize is None:
            self.itemsize = np.dtype(dtype).itemsize
        self.setsize = setsize
        self.recsize = setsize
        self.fedge = 200. * u.MHz
        self.fedge_at_top = True
        self.samplerate = samplerate
        # update headers for fun
        self[0].header.update('TBIN', (1./samplerate).to('s').value),

    def close(self):
        for fh in self.fh_raw:
            fh.Close()
        for fh in self.fh_links:
            if os.path.exists(fh):
                os.unlink(fh)

    def nskip(self, date, time0=None):
        """
        return the number of records needed to skip from start of
        file to iso timestamp 'date'.

        Optionally:
        time0 : use this start time instead of self.time0
                either a astropy.time.Time object or string in 'utc'

        """
        if time0 is None:
            time0 = self.time0
        elif isinstance(time0, str):
            time0 = Time(time0, scale='utc')

        dt = (Time(date, scale='utc')-time0)
        nskip = int(round( (dt/(self.recsize*2 / self.samplerate))
                    .to(u.dimensionless_unscaled)))
        return nskip

    def ntimebins(self, t0, t1):
        """
        determine the number of timebins between UTC start time 't0'
        and end time 't1'.
        """
        if isinstance(t0, str):
            t0 = Time(t0, scale='utc')
        if isinstance(t1, str):
            t1 = Time(t1, scale='utc')
        nt = ((t1-t0)*self.fedge/(2*self.setsize)).to(u.dimensionless_unscaled).value
        return np.ceil(nt).astype(int)

    def ntint(self, nchan):
        """
        number of samples in a frequency bin
        this is baseband data so need to know number of channels we're making

        """
        return 2*self.recsize // (2*nchan) 

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
        # MPI needs a buffer to read into
        z = np.zeros(size, dtype='i1')
        self.fh_raw[i].Iread(z)
        return z

    def record_read(self, count):
        return fromfile(self, self.dtype, count*self.itemsize)
        
    def seek(self, offset):
        assert offset % self.recsize == 0
        self.index = offset // self.recsize
        for i, fh in enumerate(self.fh_raw):
            fh.Seek(np.count_nonzero(self.sequence['raw'][:self.index] == i) *
                    self.recsize)

    def __repr__(self):
        return ("<open multifile raw_voltage_files {} "
                "using sequence file '{}' at index {}>"
                .format(self.fh_raw, self.sequence_file, self.index))

_ARO_defs = {}
_ARO_defs['PRIMARY'] = {'TELESCOP':'Algonquin',
                        'IBEAM':1, 'FD_POLN':'LIN',
                        'OBS_MODE':'SEARCH',
                        'ANT_X':0, 'ANT_Y':0, 'ANT_Z':0, 'NRCVR':1,
                        'FD_HAND':1, 'FD_SANG':0, 'FD_XYPH':0,
                        'BE_PHASE':0, 'BE_DCC':0, 'BE_DELAY':0,
                        'TCYCLE':0, 'OBSFREQ':300, 'OBSBW':100,
                        'OBSNCHAN':20, 'CHAN_DM':0,
                        'EQUINOX':2000.0, 'BMAJ':1, 'BMIN':1, 'BPA':0,
                        'SCANLEN':1, 'FA_REQ':0,
                        'CAL_FREQ':0, 'CAL_DCYC':0, 'CAL_PHS':0, 'CAL_NPHS':0,
                        'STT_IMJD':54000, 'STT_SMJD':0, 'STT_OFFS':0}
samplerate = 200. * u.MHz
_ARO_defs['SUBINT']  = {'INT_TYPE': 'TIME',
                        'SCALE': 'FluxDen',
                        'POL_TYPE': 'AABB',
                        'NPOL':1,
                        'TBIN':(1./samplerate).to('s').value,
                        'NBIN':1, 'NBIN_PRD':1,
                        'PHS_OFFS':0,
                        'NBITS':1,
                        'ZERO_OFF':0, 'SIGNINT':0,
                        'NSUBOFFS':0,
                        'NCHAN':1,
                        'CHAN_BW':1,
                        'DM':0, 'RM':0, 'NCHNOFFS':0,
                        'NSBLK':1}
#   _       ___   _____   ____  ____  
#  | |     /   \ |     | /    ||    \ 
#  | |    |     ||   __||  o  ||  D  )
#  | |___ |  O  ||  |_  |     ||    / 
#  |     ||     ||   _] |  _  ||    \ 
#  |     ||     ||  |   |  |  ||  .  \
#  |_____| \___/ |__|   |__|__||__|\_|
#                                     
class LOFARdata(multifile):
    def __init__(self, fname1, fname2, comm=None, setsize=2**16):
        """
        Initialize a lofar observation. We track/join the two polarizations file,
        We also parse the corresponding HDF5 files to initialize:
        nchan, samplerate, fwidth
          
        """
        self.telescope = 'lofar'
        super(LOFARdata, self).__init__(hdus=['SUBINT'])
        self.set_hdu_defaults(_LOFAR_defs)

        self.fname1 = fname1
        self.fname2 = fname2
        if comm is None:
            self.comm = MPI.COMM_SELF
        else:
            self.comm = comm
        self.fh1 = MPI.File.Open(self.comm, fname1, amode=MPI.MODE_RDONLY)
        self.fh2 = MPI.File.Open(self.comm, fname2, amode=MPI.MODE_RDONLY) 
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
 
        nchan = len(freqs) # = st0.attrs['NOF_SUBBANDS']

        # can also get from np.diff(freqs.diff).mean()
        fwidth = (b0.attrs['SUBBAND_WIDTH'] * 
                      u.__dict__[b0.attrs['CHANNEL_WIDTH_UNIT']]).to(u.MHz)

        samplerate = (b0.attrs['SAMPLING_RATE'] * 
                          u.__dict__[b0.attrs['SAMPLING_RATE_UNIT']]).to(u.MHz)
        h0.close()

        # defaults expected by fold.py 
        self.time0 = time0
        self.dtype = _lofar_dtypes[dtype]
        self.itemsize = _bytes_per_sample.get(self.dtype, None)
        if self.itemsize is None:
            self.itemsize = np.dtype(self.dtype).itemsize

        self.samplerate = samplerate
        self.nchan = nchan
        self.fwidth = fwidth # = samplerate = np.diff(freqs).mean() 
        self.setsize = setsize
        self.recsize = 4 * nchan * setsize
        self.freqs = freqs
        self.fedge = fbottom
        self.fedge_at_top = False

        # update some of the hdu data
        self['PRIMARY'].header['DATE-OBS'] = self.time0.iso
        self[0].header.update('TBIN', (1./samplerate).to('s').value)

    def close(self):
        self.fh1.Close()
        self.fh2.Close()

    def nskip(self, date, time0=None):
        """
        return the number of records needed to skip from start of
        file to iso timestamp 'date'.

        Optionally:
        time0 : use this start time instead of self.time0
                either a astropy.time.Time object or string in 'utc'

        """
        if time0 is None:
            time0 = self.time0
        elif isinstance(time0, str):
            time0 = Time(time0, scale='utc')

        dt = (Time(date, scale='utc')-time0)
        nskip = int(round( (dt/(self.ntint(self.nchan) / self.fwidth)).to(u.dimensionless_unscaled)))
        return nskip

    def ntimebins(self, t0, t1):
        """
        determine the number of timebins between UTC start time 't0'
        and end time 't1'
        """
        if isinstance(t0, str):
            t0 = Time(t0, scale='utc')
        if isinstance(t1, str):
            t1 = Time(t1, scale='utc')
        nt = ((t1-t0)*self.fwidth/(self.setsize)).to(u.dimensionless_unscaled).value
        return np.ceil(nt).astype(int)

    def ntint(self, nchan):
        """
        number of samples in an integration
        Lofar data is already channelized so we assert 
        nchan is the same
        """
        assert(nchan == self.nchan)
        return self.recsize // (4 * self.nchan)

    def read(self, size):
        """
        read 'size' bytes of the LOFAR data
        returns a tuple from the two polarizations
        """
        z1 = np.zeros(size, dtype='i1')
        self.fh1.Iread([z1, MPI.BYTE])
        z2 = np.zeros(size, dtype='i1')
        self.fh2.Iread([z2, MPI.BYTE])
        return z1, z2
 
    def record_read(self, count):
        """
        read 'count' records of data,
        returned as a complex number
        """ 
        raw = [np.fromstring(r, dtype=self.dtype, count=count).reshape(-1, self.nchan)
                          for r in self.read(count * self.itemsize)]
        return raw[0] + 1j*raw[1]

    def seek(self, offset):
        self.fh1.Seek(offset)
        self.fh2.Seek(offset)

    def __repr__(self):
        return ("<open lofar polarization pair {} and {}>"
                .format(os.path.basename(self.fname1), os.path.basename(self.fname2)))

# LOFAR defaults for psrfits HDUs
_LOFAR_defs = {}
_LOFAR_defs['PRIMARY'] = {'TELESCOP':'LOFAR',
                        'IBEAM':1, 'FD_POLN':'LIN',
                        'OBS_MODE':'PSR',
                        'ANT_X':0, 'ANT_Y':0, 'ANT_Z':0, 'NRCVR':1,
                        'FD_HAND':1, 'FD_SANG':0, 'FD_XYPH':0,
                        'BE_PHASE':0, 'BE_DCC':0, 'BE_DELAY':0,
                        'TCYCLE':0, 'OBSFREQ':300, 'OBSBW':100,
                        'OBSNCHAN':20, 'CHAN_DM':0,
                        'EQUINOX':2000.0, 'BMAJ':1, 'BMIN':1, 'BPA':0,
                        'SCANLEN':1, 'FA_REQ':0,
                        'CAL_FREQ':0, 'CAL_DCYC':0, 'CAL_PHS':0, 'CAL_NPHS':0,
                        'STT_IMJD':54000, 'STT_SMJD':0, 'STT_OFFS':0}
samplerate = 200. * u.MHz
_LOFAR_defs['SUBINT']  = {'INT_TYPE': 'TIME',
                        'SCALE': 'FluxDen',
                        'POL_TYPE': 'AABB',
                        'NPOL':1,
                        'TBIN':(1./samplerate).to('s').value,
                        'NBIN':1, 'NBIN_PRD':1,
                        'PHS_OFFS':0,
                        'NBITS':1,
                        'ZERO_OFF':0, 'SIGNINT':0,
                        'NSUBOFFS':0,
                        'NCHAN':1,
                        'CHAN_BW':1,
                        'DM':0, 'RM':0, 'NCHNOFFS':0,
                        'NSBLK':1}

#    ____  ___ ___  ____  ______ 
#   /    ||   |   ||    \|      |
#  |   __|| _   _ ||  D  )      |
#  |  |  ||  \_/  ||    /|_|  |_|
#  |  |_ ||   |   ||    \  |  |  
#  |     ||   |   ||  .  \ |  |  
#  |___,_||___|___||__|\_| |__|  
#                            
class GMRTdata(multifile):
    def __init__(self, timestamp_file, files, setsize=2**22,
                 utc_offset=TimeDelta(5.5*3600, format='sec'), comm=None):
        super(GMRTdata, self).__init__(hdus=['SUBINT'])
        self.set_hdu_defaults(_GMRT_defs)

        self.telescope = 'gmrt'
        if comm is None:
            self.comm = MPI.COMM_SELF
        else:
            self.comm = comm

        self.timestamp_file = timestamp_file
        self.indices, self.timestamps, self.gsb_start = read_timestamp_file(
            timestamp_file, utc_offset)
        self.fh_raw = [MPI.File.Open(self.comm, raw, amode=MPI.MODE_RDONLY) for raw in files]
        self.setsize = setsize
        self.recsize = setsize
        self.index = 0

        # parameters for fold:
        self.samplerate = 100.*u.MHz / 3.
        self.fedge = 156. * u.MHz
        self.fedge_at_top = True
        self.time0 = self.timestamps[0]
        # GMRT time is off by 1 second
        self.time0 -= (2.**24/(100*u.MHz/6.)).to(u.s)
        self.dtype = np.int8
        self.itemsize = {np.int8: 2}[self.dtype]

    def close(self):
        for fh in self.fh_raw:
            fh.Close()

    def nskip(self, date, time0=None):
        """
        return the number of records needed to skip from start of
        file to iso timestamp 'date'.

        Optionally:
        time0 : use this start time instead of self.time0
                either a astropy.time.Time object or string in 'utc'

        """
        if time0 is None:
            time0 = self.time0
        elif isinstance(time0, str):
            time0 = Time(time0, scale='utc')

        dt = (Time(date, scale='utc')-time0)
        nskip = int(round( (dt/(self.recsize / self.samplerate))
                    .to(u.dimensionless_unscaled)))
        return nskip

    def ntimebins(self, t0, t1):
        """
        determine the number of timebins between UTC start time 't0'
        and end time 't1'
        """
        if isinstance(t0, str):
            t0 = Time(t0, scale='utc')
        if isinstance(t1, str):
            t1 = Time(t1, scale='utc')
        nt = ((t1-t0)*self.samplerate/(2*self.setsize)).to(u.dimensionless_unscaled).value
        return np.ceil(nt).astype(int)

    def ntint(self, nchan):
        return self.setsize // (2*nchan)

    def record_read(self, count):
        return fromfile(self, self.dtype, count*self.itemsize)

    def read(self, size):
        assert size == self.recsize
        if self.index == len(self.indices):
            raise EOFError
        self.index += 1
        # print('reading from {}, t={}'.format(
        #     self.fh_raw[self.indices[self.index-1]],
        #     self.timestamps[self.index-1]))
        z = np.zeros(size, dtype='i1')
        self.fh_raw[self.indices[self.index-1]].Iread(z)
        return z

    def seek(self, offset):
        assert offset % self.recsize == 0
        self.index = offset // self.recsize
        for i, fh in enumerate(self.fh_raw):
            fh.Seek(np.count_nonzero(self.indices[:self.index] == i) *
                    self.recsize)

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
_GMRT_defs = {}
_GMRT_defs['PRIMARY'] = {'TELESCOP':'GMRT',
                        'IBEAM':1, 'FD_POLN':'LIN',
                        'OBS_MODE':'PSR',
                        'ANT_X':0, 'ANT_Y':0, 'ANT_Z':0, 'NRCVR':1,
                        'FD_HAND':1, 'FD_SANG':0, 'FD_XYPH':0,
                        'BE_PHASE':0, 'BE_DCC':0, 'BE_DELAY':0,
                        'TCYCLE':0, 'OBSFREQ':300, 'OBSBW':100,
                        'OBSNCHAN':20, 'CHAN_DM':0,
                        'EQUINOX':2000.0, 'BMAJ':1, 'BMIN':1, 'BPA':0,
                        'SCANLEN':1, 'FA_REQ':0,
                        'CAL_FREQ':0, 'CAL_DCYC':0, 'CAL_PHS':0, 'CAL_NPHS':0,
                        'STT_IMJD':54000, 'STT_SMJD':0, 'STT_OFFS':0}
samplerate = 100. * u.MHz / 3.
_GMRT_defs['SUBINT']  = {'INT_TYPE': 'TIME',
                        'SCALE': 'FluxDen',
                        'POL_TYPE': 'AABB',
                        'NPOL':1,
                        'TBIN':(1./samplerate).to('s').value,
                        'NBIN':1, 'NBIN_PRD':1,
                        'PHS_OFFS':0,
                        'NBITS':1,
                        'ZERO_OFF':0, 'SIGNINT':0,
                        'NSUBOFFS':0,
                        'NCHAN':1,
                        'CHAN_BW':1,
                        'DM':0, 'RM':0, 'NCHNOFFS':0,
                        'NSBLK':1}

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
    This routine checks for such cases and creates a well-named link to the file.

    Returns (good_name, islink)
    """
    if f is None: return f

    fl = f
    newlink = False
    if ':' in f:
        #fl = tempfile.mktemp(prefix=os.path.basename(f).replace(':','_'), dir='/tmp')
        fl = os.path.join('/tmp', os.path.dirname(f).replace('/','_') + '__' + os.path.basename(f).replace(':','_'))
        if not os.path.exists(fl):
            try:
                os.symlink(f, fl)
            except(OSError):
                pass
            newlink = True
    return fl, newlink

