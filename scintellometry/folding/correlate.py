from __future__ import division, print_function

from fractions import gcd as GCD
import numpy as np
import os
import astropy.units as u
from astropy.time import Time
import h5py

_fref = 150. * u.MHz  # ref. freq. for dispersion measure

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.scipy_fftpack import (rfft, rfftfreq, irfft,
                                                 fft, ifft, fftfreq, fftshift)
    _fftargs = {'threads': os.environ.get('OMP_NUM_THREADS', 2),
                'planner_effort': 'FFTW_ESTIMATE'}
except(ImportError):
    print("Consider installing pyfftw: https://github.com/hgomersall/pyFFTW")
    # use FFT from scipy, since unlike numpy it does not cast up to complex128
    from scipy.fftpack import (rfft, rfftfreq, irfft, fft,
                               ifft, fftfreq, fftshift)
    _fftargs = {}

dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc


def correlate(fh1, fh2, nchan, dm=0., t0=None, t1=None, rfi_filter_raw=None,
              comm=None, verbose=0):
    """
    correlate data from two filehandles from time t0 to time t1

    Args:
    fh1 fh2: MultiFile filehandles
    t0 : time to start the correlation
        (default: None, start of file)
    t1 : time to end the correlation
        (default: None, end of file)
    rfi_filter_raw : (default: None)
    comm : MPI communicator
    verbose : (int) level of verbosity
            default 0

    """
    if comm is None:
        rank = 0
        size = 1
    else:
        rank = comm.rank
        size = comm.size

    t0 = Time(t0, scale='utc')
    t1 = Time(t1, scale='utc')
    # configure necessary quantities
    for fh in [fh1, fh2]:
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
                 (1./fh.freq**2 - 1./_fref**2)).to(u.s).value

    fcorr = h5py.File('parallel_test.hdf5', 'w')#, driver='mpio', comm=comm)
    dset = fcorr.create_dataset('corr', (10000000, 85), dtype='complex64',
                                maxshape=(None, nchan))

    # get frequencies on same grid
    # number of bins to sum over to have roughly same BW/channel
    df1df2 = int(np.abs(np.round(np.diff(fh1.freq).mean()/np.diff(fh2.freq).mean())))
    # resultant centre frequencies
    fh2freqs = []
    for i in range(int(fh2.freq.size/df1df2)):
        fh2freqs.append(fh2.freq[i*df1df2:(i+1)*df1df2].mean())

    # start reading the raw data
    idx = 0
    raw1 = fh1.seek_record_read((fh1.this_nskip + idx + rank) * fh1.blocksize,
                                fh1.blocksize)
    raw2 = fh2.seek_record_read((fh2.this_nskip + idx + rank) * fh2.blocksize,
                                fh2.blocksize)
    print("TT",fh1.dt1, fh2.dt1)
    # lofar dt = 5.12e-6 s, gmrt dt = 3e-8 s
    #np.save('lofarfreq.npy', fh1.freq.value)
    #np.save('gmrtfreq.npy', fh2.freq.value)
    #return
    endread = False
    while raw1.size > 0 and raw2.size > 0:
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
            chan1 = fh1.thisfft(vals1.reshape(-1, nchan*2), axis=-1,
                                overwrite_x=True, **_fftargs)
        else:  # lofar and gmrt-phased are already channelised
            chan1 = vals1
        if fh2.nchan == 1:
            chan2 = fh2.thisfft(vals2.reshape(-1, nchan*2), axis=-1,
                                overwrite_x=True, **_fftargs)
        else:
            chan2 = vals2

        # we now have both, put on same grid
        # sum fh2 to have roughly same freq. bw
        chan2 = np.hstack([chan2[:,i*df1df2:(i+1)*df1df2].mean(axis=-1, keepdims=1)
                          for i in range(int(chan2.shape[1]/df1df2))])
        # sum fh1 to have roughly same sample times as fh2
        # and chop off non-overlapping freq. bins (TODO: interp onto same grid)
        dt1dt2 = int(chan1.shape[0]/chan2.shape[0])
        chan1 = np.array([chan1[i*dt1dt2:(i+1)*dt1dt2, 11:-4].mean(axis=0)
                          for i in range(int(chan1.shape[0]/dt1dt2))])
        #chan1 = chan1[:,11:-4]
        nsamp = chan1.shape[0]
        xpower = chan1 * chan2.conjugate()
        for n, v in enumerate(xpower):
            dset[idx+rank+n] = v
        #dset[(idx+rank)*nsamp:(idx+rank+1)*nsamp,:] = chan1 * chan2.conjugate()
        #dset[idx] = np.random.random(85)
        idx += 1
        print("DONE",idx)
        # get ready for next read
        for fh in [fh1, fh2]:
            tstream = t0 + (idx + rank)*fh.ntint(nchan)*fh.dtsample
            print("T",tstream)
            if (tstream - t1).sec > 0.:
                endread = True
        if endread:
            break
        else:
            raw1 = fh1.seek_record_read((fh1.this_nskip + idx + rank)
                                    * fh1.blocksize, fh1.blocksize)
            raw2 = fh2.seek_record_read((fh2.this_nskip + idx + rank)
                                    * fh2.blocksize, fh2.blocksize)
    fcorr.close()
