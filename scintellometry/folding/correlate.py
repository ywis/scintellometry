from __future__ import division, print_function

from fractions import Fraction
import numpy as np
import os
import astropy.units as u
from astropy.time import Time
import h5py

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
              dedisperse=None,rfi_filter_raw=None,
              savefile='test_hdf5',
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
        # TODO, make mpi-capable
        rank = comm.rank
        size = comm.size

    # output dataset
    fcorr = h5py.File('savefile.hdf5', 'w') #, driver='mpio', comm=comm)

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

    # read files in same-size time chunks
    sr = Fraction(fh1.dtsample.to(u.s).value * fh1.blocksize / fh1.recordsize)
    sr /= Fraction(fh2.dtsample.to(u.s).value * fh2.blocksize / fh2.recordsize)
    sr = sr.limit_denominator(5000)
    n = sr.denominator
    d = sr.numerator
    raw1 = fh1.seek_record_read((fh1.this_nskip + rank) * fh1.blocksize,
                                fh1.blocksize * n)
    raw2 = fh2.seek_record_read((fh2.this_nskip + rank) * fh2.blocksize,
                                fh2.blocksize * d)

    # time averaging
    Tavg = Fraction(raw1.shape[0], raw2.shape[0]).limit_denominator(1000)
    Tden = Tavg.denominator
    Tnum = Tavg.numerator

    # channel averaging
    f1 = (fh1.freq.max(), fh1.freq.min(), len(fh1.freq),
          np.sign(np.diff(fh1.freq).mean()))
    f2 = (fh2.freq.max(), fh2.freq.min(), len(fh2.freq),
          np.sign(np.diff(fh2.freq).mean()))
    f1keep = (fh1.freq > max(f1[1], f2[1])) & (fh1.freq < min(f1[0], f2[0]))
    f2keep = (fh2.freq > max(f1[1], f2[1])) & (fh2.freq < min(f1[0], f2[0]))
    np.save('f1freq.npy', fh1.freq.value)
    np.save('f2freq.npy', fh2.freq.value)
    Favg = abs(Fraction(np.diff(fh1.freq.value).mean() / np.diff(fh2.freq.value).mean()))
    Favg = Favg.limit_denominator(5000)
    Fden = Favg.denominator
    Fnum = Favg.numerator
    # print the common freq. grid
    f1s = fh1.freq[f1keep]
    f1s = f1s.reshape(f1s.size / Fden, Fden).mean(axis=-1)
    f2s = fh2.freq[f2keep]
    f2s = f2s.reshape(f2s.size / Fnum, Fnum).mean(axis=-1)
    # sort low freq to high freq
    if f1[3] < 0:
        f1s = f1s[::-1]
    if f2[3] < 0:
        f2s = f2s[::-1]

    # output dataset
    # shape is timestep, channel
    # TODO: save the frequency grids (to figure out future interpolation)
    fcorr.create_dataset('freq1', data=f1s.value)
    fcorr.create_dataset('freq2', data=f2s.value)

    # shape is timestep, channel
    dset = fcorr.create_dataset('corr', (0, f1s.size), dtype='complex64',
                                maxshape=(None, nchan))
    # tsteps = fcorr.create_dataset('tsteps', (0), dtype='f', maxshape=None)

    idx = 0
    endread = False
    while (raw1.size > 0) and (raw2.size > 0):
        if rfi_filter_raw is not None:
            raw1, ok = rfi_filter_raw(raw1, nchan)
            raw2, ok = rfi_filter_raw(raw2, nchan)

        if fh1.telescope == 'aro':
            vals1 = raw1.astype(np.float32)
        else:
            vals1 = raw1
        if fh2.telescope == 'aro':
            vals2 = raw2.astype(np.float32)
        else:
            vals2 = raw2

        if fh1.nchan == 1:
            # ARO data should fall here
            chan1 = fh1.thisfft(vals1.reshape(-1, nchan * 2), axis=-1,
                                overwrite_x=True, **_fftargs)
        else:  # lofar and gmrt-phased are already channelised
            chan1 = vals1
        if fh2.nchan == 1:
            chan2 = fh2.thisfft(vals2.reshape(-1, nchan * 2), axis=-1,
                                overwrite_x=True, **_fftargs)
        else:
            chan2 = vals2

        # average onto same time grid
        # this_tsteps = t0 +
        chan1 = chan1.reshape(Tnum, chan1.shape[0] / Tnum, -1).mean(axis=0)
        chan2 = chan2.reshape(Tden, chan2.shape[0] / Tden, -1).mean(axis=0)

        # average onto same freq grid
        chan1 = chan1[..., f1keep]
        chan2 = chan2[..., f2keep]
        chan1 = chan1.reshape(-1, chan1.shape[1] / Fden, Fden).mean(axis=-1)
        chan2 = chan2.reshape(-1, chan2.shape[1] / Fnum, Fnum).mean(axis=-1)

        print("idx", idx, fh1.time(), fh2.time(), dset.shape)
        # x-correlate
        xpower = chan1 * chan2.conjugate()[:, 0:100]
        curshape = dset.shape
        dset.resize((curshape[0] + xpower.shape[0], dset.shape[1]))
        dset[curshape[0]:dset.shape[0]] = xpower

        # read in next dataset if we haven't hit t1 yet
        for fh in [fh1, fh2]:
            if (fh.time() - t1).sec > 0.:
                endread = True
        if endread:
            break
        else:
            raw1 = fh1.seek_record_read((fh1.this_nskip + rank + (n * idx))
                                        * fh1.blocksize, fh1.blocksize * n)
            raw2 = fh2.seek_record_read((fh2.this_nskip + rank + (d * idx))
                                        * fh2.blocksize, fh2.blocksize * d)
        print("idx",idx, fh1.time(), fh2.time())
        idx += 1

    fcorr.close()
    return foldspec, icount, waterfall
