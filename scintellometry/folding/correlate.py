from __future__ import division, print_function

import numpy as np
import os
import astropy.units as u

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

    if t0 is None:
        t0 = max(fh1.time0, fh2.time0)
        print("Starting at %s" % t0)
    print("TTT",t0)
    if t1 is None:
        pass
    # configure the fh's for xcorr stream

    for fh in [fh1, fh2]:
        fh.seek(t0)
        fh.nskip = fh.tell()/fh.blocksize

        fh.dt1 = (1./fh.samplerate).to(u.s)
        if fh.telescope != 'lofar':
            fh.dtsample = nchan * 2 * fh.dt1
        # else: lofar already channelized/dtsample-d
        fh.this_nskip = fh.nskip(t0)
        fh.tstart = fh.dtsample * fh.ntint(nchan) * fh.this_nskip

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

    # prep. reshape so we have similar freqs
    r1 = np.diff(fh1.freq).mean() / np.diff(fh2e.freq).mean()
    print("r1",r1)
    print("fh1", fh1.freq.min(), fh1.freq.max(), np.diff(fh1.freq).mean(), fh1.nskip)
    print("fh2", fh2.freq.min(), fh2.freq.max(), np.diff(fh2.freq).mean(), fh2.nskip)
    return foldspec, icount, waterfall
