""" classes to handle ARO, LOFAR, and GMRT data in a consistent way """

from __future__ import division

import numpy as np
import os
import re

from astropy.table import Table
from astropy import units as u
from astropy.time import Time
from h5py import File as HDF5File
from mpi4py import MPI
from psrfits_tools import psrFITS

class multifile(psrFITS):

    def set_hdu_defaults(self, dictionary):
        for hdu, defs in dictionary.iteritems():
            for card, val in defs.iteritems():
                self[hdu].header.update(card, val)

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
        nskip = int(round(
            (dt/(self.recsize*2 / self.samplerate))
            .to(u.dimensionless_unscaled)))
        return nskip

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
    def __init__(self, sequence_file, raw_voltage_files, recsize=2**25,
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
        self.recsize = recsize
        self.index = 0
        self.seq0 = self.sequence['seq'][0]

        # defaults expected by fold.py 
        self.dtype = dtype
        self.fedge = 200. * u.MHz
        self.fedge_at_top = True
        self.samplerate = samplerate
        # use rfft 
        self.real_data = True
        # update headers for fun
        self[0].header.update('TBIN', (1./samplerate).to('s').value),

    def seek(self, offset):
        assert offset % self.recsize == 0
        self.index = offset // self.recsize
        for i, fh in enumerate(self.fh_raw):
            fh.Seek(np.count_nonzero(self.sequence['raw'][:self.index] == i) *
                    self.recsize)

    def close(self):
        for fh in self.fh_raw:
            fh.Close()
        for fh in self.fh_links:
            if os.path.exists(fh):
                os.unlink(fh)

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
                        'NPOL':2,
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
    def __init__(self, fname1, fname2, comm=None, recsize=2**25, samplerate=200.*u.MHz):
        self.telescope = 'lofar'
        super(LOFARdata, self).__init__(hdus=['SUBINT'])
        self.set_hdu_defaults(_LOFAR_defs)


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
        fbottom = (b0['COORDINATES']['COORDINATE_1']
                       .attrs['AXIS_VALUES_WORLD'][0] * u.Hz).to(u.MHz)

        stokes = sorted([i for i in b0.keys()
                           if 'STOKES' in i and 'i2f' not in i])
        st0 = b0[stokes[0]]
        dtype = st0.attrs['DATATYPE']
 
        self.recsize = recsize # 2**25 = 32 Mb samples
        nchan = st0.attrs['NOF_SUBBANDS']
        self.dtype = _lofar_dtypes[dtype]
        h0.close()

        # defaults expected by fold.py 
        self.time0 = time0
        self.dtype = _lofar_dtypes[dtype]
        self.samplerate = samplerate
        self.fedge = fbottom
        self.fedge_at_top = False
        # use fft  (not rfft)
        self.real_data = False 

        # update some of the hdu data
        self['PRIMARY'].header['DATE-OBS'] = self.time0.iso
        self[0].header.update('TBIN', (1./samplerate).to('s').value)

    def seek(self, offset):
        self.fh1.Seek(offset)
        self.fh2.Seek(offset)
 
    def read(self, size):
        z1 = np.zeros(size, dtype='i1')
        self.fh1.Iread([z1, MPI.BYTE])
        z2 = np.zeros(size, dtype='i1')
        self.fh2.Iread([z2, MPI.BYTE])
        return z1, z2
 
    def close(self):
        self.fh1.Close()
        self.fh2.Close()

    def __repr__(self):
        return ("<open multifile lofar files {} and {}"
                .format(self.fh1, self.fh2))

_lofar_dtypes = {'float':'>f4', 'int8':'>i1'}

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
                        'NPOL':2,
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

