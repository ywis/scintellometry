# -*- coding: utf-8 -*-
"""Fold LOFAR data.  Use sub-bands; sub-channeling not yet implemented"""
from __future__ import division, print_function

import numpy as np
from scipy.fftpack import rfftfreq  # rfft, irfft
import astropy.units as u

from fromfile import fromfile

dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc


def fold(file1, file2, dtype, fbottom, fwidth, nchan,
         nt, ntint, nskip, ngate, ntbin, ntw, dm, fref, phasepol,
         coherent=False, do_waterfall=True, do_foldspec=True, verbose=True,
         progress_interval=100):
    """Fold pre-channelized LOFAR data, possibly dedispersing it

    Parameters
    ----------
    file1, file2 : string
        names of the files holding real and imaginary subchannel timeseries
    dtype : numpy dtype
        way the data are stored in the file (normally '>f4')
    fbottom : float
        frequency of the lowest channel (frequency units)
    fwidth : float
        channel width (frequency units, normally 200*u.MHz/1024.)
    nchan : int
        number of frequency channels
    nt, ntint : int
        number nt of sets to use, each containing ntint samples;
        hence, total # of samples used is nt*(2*ntint).
    nskip : int
        number of bytes to skip before reading
    ngate, ntbin : int
        number of phase and time bins to use for folded spectrum
        ntbin should be an integer fraction of nt
    ntw : int
        number of time samples to combine for waterfall (does not have to be
        integer fraction of nt)
    dm : float
        dispersion measure of pulsar, used to correct for ism delay
        (column number density)
    fref: float
        reference frequency for dispersion measure
    phasepol : callable
        function that returns the pulsar phase for time in seconds relative to
        the start of the file that is read (i.e., including nskip)
    do_waterfall, do_foldspec : bool
        whether to construct waterfall, folded spectrum (default: True)
    verbose : bool
        whether to give some progress information (default: True)
    progress_interval : int
        Ping every progress_interval sets
    """

    # initialize folded spectrum and waterfall
    foldspec2 = np.zeros((nchan, ngate, ntbin))
    nwsize = nt*ntint//ntw
    waterfall = np.zeros((nchan, nwsize))

    # # of items to read from file.
    itemsize = np.dtype(dtype).itemsize
    count = nchan*ntint
    if verbose:
        print('Reading from {}\n         and {}'.format(file1, file2))

    with open(file1, 'rb', count*itemsize) as fh1, \
         open(file2, 'rb', count*itemsize) as fh2:

        if nskip > 0:
            if verbose:
                print('Skipping {0} bytes'.format(nskip))
            fh1.seek(nskip)
            fh2.seek(nskip)

        foldspec = np.zeros((nchan, ngate))
        icount = np.zeros((nchan, ngate))

        dtsample = (1./fwidth).to(u.s)
        tstart = dtsample * nskip // itemsize // nchan

        # pre-calculate time delay due to dispersion in course channels
        freq = fbottom + fwidth * np.arange(nchan)
        dt = (dispersion_delay_constant * dm *
              (1./freq**2 - 1./fref**2)).to(u.s).value

        for j in xrange(nt):
            if verbose and j % progress_interval == 0:
                print('Doing {:6d}/{:6d}; time={:18.12f}'.format(
                    j+1, nt, (tstart+dtsample*j*ntint).value))
                # time since start of file

            # just in case numbers were set wrong -- break if file ends
            # better keep at least the work done
            try:
                # data stored as series of floats in two files,
                # one for real and one for imaginary
                raw1 = fromfile(fh1, dtype, count).reshape(-1,nchan)
                raw2 = fromfile(fh2, dtype, count).reshape(-1,nchan)
            except(EOFError):
                break

            power = raw1**2+raw2**2
            # power[#int, #block]

            # current sample positions in stream
            isr = j*ntint + np.arange(ntint)

            if do_waterfall:
                # loop over corresponding positions in waterfall
                for iw in xrange(isr[0]//ntw, isr[-1]//ntw + 1):
                    if iw < nwsize:  # add sum of corresponding samples
                        waterfall[:,iw] += np.sum(power[isr//ntw == iw],
                                                  axis=0)

            if do_foldspec:
                tsample = (tstart + dtsample * isr).value  # times since start

                for k in xrange(nchan):
                    t = tsample - dt[k]  # dedispersed times
                    phase = phasepol(t)  # corresponding PSR phases
                    iphase = np.remainder(phase*ngate,
                                          ngate).astype(np.int)
                    # sum and count samples by phase bin
                    foldspec[k] += np.bincount(iphase, power[:,k], ngate)
                    icount[k] += np.bincount(iphase, None, ngate)

                ibin = j*ntbin//nt  # bin in the time series: 0..ntbin-1
                if (j+1)*ntbin//nt > ibin:  # last addition to bin?
                    # get normalised flux in each bin (where any were added)
                    nonzero = icount > 0
                    nfoldspec = np.where(nonzero, foldspec/icount, 0.)
                    # subtract phase average and store
                    nfoldspec -= np.where(nonzero,
                                          np.sum(nfoldspec, 1, keepdims=True) /
                                          np.sum(nonzero, 1, keepdims=True), 0)
                    foldspec2[:,:,ibin] = nfoldspec
                    # reset for next iteration
                    foldspec *= 0
                    icount *= 0

    if verbose:
        print('read {0:6d} out of {1:6d}'.format(j+1, nt))

    if do_waterfall:
        nonzero = waterfall == 0.
        waterfall -= np.where(nonzero,
                              np.sum(waterfall, 1, keepdims=True) /
                              np.sum(nonzero, 1, keepdims=True), 0.)

    return foldspec2, waterfall
