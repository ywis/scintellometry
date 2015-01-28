#           _____   ____   _____ _    _ _____ __  __ ______
#     /\   |  __ \ / __ \ / ____| |  | |_   _|  \/  |  ____|
#    /  \  | |__) | |  | | |    | |__| | | | | \  / | |__
#   / /\ \ |  _  /| |  | | |    |  __  | | | | |\/| |  __|
#  / ____ \| | \ \| |__| | |____| |  | |_| |_| |  | | |____
# /_/    \_\_|  \_\\____/ \_____|_|  |_|_____|_|  |_|______|

from __future__ import division
import os

import numpy as np
from scipy.fftpack import fftfreq, fftshift
from astropy.time import Time
import astropy.units as u

from . import MultiFile, header_defaults


class AROCHIMEData(MultiFile):

    telescope = 'arochime'

    def __init__(self, raw_files, blocksize,
                 samplerate, fedge, fedge_at_top, dtype='cu4bit,cu4bit',
                 comm=None):
        """ARO data aqcuired with a CHIME correlator containts 1024 channels
        over the 400MHz BW, 2 polarizations, and 2 unsigned 8-byte ints for
        real and imaginary for each timestamp.
        """
        self.meta = eval(open(raw_files[0] + '.meta').read())
#        meta_last = eval(open(raw_files[-1] + '.meta').read())
#        duration_last = (os.path.getsize(raw_files[-1]) // self.recordsize *
#                         self.dtsample)
#        self.time1 = Time(end_meta['stime'], format='unit') + duration_last
        nchan = self.meta['nfreq']
        self.time0 = Time(self.meta['stime'], format='unix')
        self.npol = self.meta['ninput']
        self.samplerate = samplerate
        self.fedge = fedge
        self.fedge_at_top = fedge_at_top
        f = fftshift(fftfreq(nchan, (2./samplerate).to(u.s).value)) * u.Hz
        if fedge_at_top :
            self.frequencies = fedge - (f-f[0])
            if self.time0.value > 1406764800  and self.time0.value < 1406937600:
            ### reset these for just july run when we swapped a cable
            	initfreqar = fedge - (f-f[0])
            	newind = np.zeros(1024,dtype=int)
            	iar = [4,5,6,7,0,1,2,3]
            	for i in range(0,128):
                    for j in range(0,8):
                        newind[i*8 + j] = int(i*8 + iar[j])
            	self.frequencies = initfreqar[newind]
            else:
		pass
	else:
            self.frequencies = fedge + (f-f[0])

        self.dtsample = (nchan * 2 / samplerate).to(u.s)
        if comm.rank == 0:
            print("In AROCHIMEData, calling super")
            print("Start time: ", self.time0.iso)

        self.files = raw_files
        self.filesize = os.path.getsize(self.files[0])
        self.current_file_number = None
        super(AROCHIMEData, self).__init__(raw_files, blocksize, dtype, nchan,
                                           comm=comm)

    def open(self, files, file_number=0):
        if file_number == self.current_file_number:
            return
        if self.current_file_number is not None:
            self.fh_raw.close()
        self.fh_raw = open(files[file_number], mode='rb')
        self.current_file_number = file_number

    def close(self):
        if self.current_file_number is not None:
            self.fh_raw.close()

    def read(self, size):
        """Read size bytes, returning an ndarray with np.int8 dtype.

        Incorporate information from multiple underlying files if necessary.
        The current file pointer are assumed to be pointing at the right
        locations, i.e., just before the first bit of data that will be read.
        """

        if size % self.recordsize != 0:
            raise ValueError("Cannot read a non-integer number of records")

        # ensure we do not read beyond end
        size = min(size, len(self.files) * self.filesize - self.offset)
        if size <= 0:
            raise EOFError('At end of file in AROCHIME.read')

        # allocate buffer for MPI read
        z = np.empty(size, dtype=np.int8)

        # read one or more pieces
        iz = 0
        while(iz < size):
            block, already_read = divmod(self.offset, self.filesize)
            fh_size = min(size - iz, self.filesize - already_read)
            z[iz:iz+fh_size] = np.fromstring(self.fh_raw.read(fh_size),
                                             dtype=z.dtype)
            self._seek(self.offset + fh_size)
            iz += fh_size

        return z

    def _seek(self, offset):
        assert offset % self.recordsize == 0
        file_number = offset // self.filesize
        file_offset = offset % self.filesize
        self.open(self.files, file_number)
        self.fh_raw.seek(file_offset)
        self.offset = offset

    def ntint(self, nchan):
        return self.setsize

# GMRT defaults for psrfits HDUs
# Note: these are largely made-up at this point
header_defaults['arochime'] = {
    'PRIMARY': {'TELESCOP':'AROCHIME',
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


def read_start_time(filename):
    """
    Reads in arochime .meta file as a dictionary and gets start time.

    Parameters
    ----------
    filename: str
         full path to .meta file

    Returns
    -------
    start_time: Time object
         Unix time of observation start
    """
    f = eval(open(filename).read())
    return Time(f['stime'], format='unix')
