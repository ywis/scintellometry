from __future__ import division, print_function

from fractions import Fraction
import numpy as np
import os
import astropy.units as u
from astropy.time import Time

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.scipy_fftpack import (rfft, rfftfreq, irfft,
                                                 fft, ifft, fftfreq)
    _fftargs = {'threads': os.environ.get('OMP_NUM_THREADS', 2),
                'planner_effort': 'FFTW_ESTIMATE'}
except(ImportError):
    print("Consider installing pyfftw: https://github.com/hgomersall/pyFFTW")
    # use FFT from scipy, since unlike numpy it does not cast up to complex128
    from scipy.fftpack import rfft, rfftfreq, irfft, fft, ifft, fftfreq
    _fftargs = {}

dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc
_fref = 150. * u.MHz  # ref. freq. for dispersion measure


def correlate(fh1, fh2, dm, nchan, ngate, ntbin, nt, ntint, ntw,
              t0=None, t1=None, comm=None):
    """
    fh1 : file handle of first data stream
    fh2 : file handle of second data stream
    dm :
    nchan :
    t0 : start time (isot) x-corr
         [None] start at common beginning of (fh1, fh2)
    t1 : end time of (isot) x-corr
         [None] end at common ending of (fh1, fh2)
    comm : MPI communicator or None

    """

    if comm is None:
        rank = 0
        size = 1
    else:
        rank = comm.rank
        size = comm.size

    # initialize the folded spectrum and waterfall
    foldspec = np.zeros((nchan, ngate, ntbin))
    icount = np.zeros((nchan, ngate, ntbin), dtype=np.int64)
    waterfall = np.zeros((nchan, ngate, ntbin))
    nwsize = nt*ntint//ntw

    # find nearest starttime landing on same sample
    if t0 is None:
        t0 = max(fh1.time0, fh2.time0)
        print("Starting at %s" % t0)
    t0 = Time(t0, scale='utc')
    print("TTT",t0, type(t0))
    if t1 is None:
        pass
    t1 = Time(t1, scale='utc')

    # configure the fh's for xcorr stream
    for i, fh in enumerate([fh1, fh2]):
        fh.seek(t0)
        fh.dt1 = (1. / fh.samplerate).to(u.s)
        fh.this_nskip = fh.nskip(t0)
        # set up FFT functions: real vs complex fft's
        if fh.nchan > 1:
            fh.thisfft = fft
            fh.thisifft = ifft
            fh.thisfftfreq = fftfreq
        else:
            fh.thisfft = rfft
            fh.thisifft = irfft
            fh.thisfftfreq = rfftfreq

        # pre-calculate time delay due to dispersion in coarse channels
        # LOFAR data is already channelized
        if fh.nchan > 1:
            fh.freq = fh.frequencies
        else:
            if fh.fedge_at_top:
                fh.freq = fh.fedge\
                    - fh.thisfftfreq(nchan * 2, fh.dt1.value) * u.Hz
            else:
                fh.freq = fh.fedge\
                    + fh.thisfftfreq(nchan * 2, fh.dt1.value) * u.Hz
            # sort lowest to highest freq
            # freq.sort()
            # [::2] sets frequency channels to numerical recipes ordering
            # or, rfft has an unusual ordering
            fh.freq = fh.freq[::2]

        fh.dt = (dispersion_delay_constant * dm *
                 ( 1./fh.freq**2 - 1./_fref**2) ).to(u.s).value

    # find the nearest sample that starts at same time
    sr = Fraction(fh1.dtsample.to(u.s).value * fh1.blocksize / fh1.recordsize)
    sr /= Fraction(fh2.dtsample.to(u.s).value  * fh2.blocksize / fh2.recordsize)
    sr = sr.limit_denominator(5000)

    print("0", fh1.time(), fh2.time())
    print(sr)
    n = sr.denominator
    d = sr.numerator
    print(fh1.blocksize,fh1.dtsample, fh1.recordsize, fh2.blocksize,fh2.dtsample, fh2.recordsize)
    raw1 = fh1.seek_record_read((fh1.this_nskip + rank) * fh1.blocksize,
                                (fh1.blocksize*n) )
    raw2 = fh2.seek_record_read((fh2.this_nskip + rank) * fh2.blocksize,
                                (fh2.blocksize*d))
    endread = False
    idx = 1
    while (raw1.size > 0) and (raw2.size > 0):
        print("idx", idx, fh1.time(), fh2.time(), raw1.shape, raw2.shape)
        raw1 = fh1.seek_record_read((fh1.this_nskip + rank + n*idx)
                                    * fh1.blocksize, fh1.blocksize * n)
        raw2 = fh2.seek_record_read((fh2.this_nskip + rank + d*idx)
                                    * fh2.blocksize, fh2.blocksize * d)
        print("idx",idx, fh1.time(), fh2.time())
        idx += 1
        if idx > 4: break

    return foldspec, icount, waterfall
