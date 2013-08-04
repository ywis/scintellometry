"""FFT and fold ARO data"""
from __future__ import division, print_function

import numpy as np
# use FFT from scipy, since unlike numpy it does not cast up to complex128
from scipy.fftpack import rfft, rfftfreq, irfft
import astropy.units as u

from fromfile import fromfile

dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc


def fold(fh1, dtype, samplerate, fedge, fedge_at_top, nchan,
         nt, ntint, nhead, ngate, ntbin, ntw, dm, fref, phasepol,
         dedisperse='incoherent',
         do_waterfall=True, do_foldspec=True, verbose=True,
         progress_interval=100):
    """FFT ARO data, fold by phase/time and make a waterfall series

    Parameters
    ----------
    fh1 : file handle
        handle to file holding voltage timeseries
    dtype : numpy dtype or '4bit' or '1bit'
        way the data are stored in the file
    samplerate : float
        rate at which samples were originally taken and thus double the
        band width (frequency units)
    fedge : float
        edge of the frequency band (frequency units)
    fedge_at_top: book
        whether edge is at top (True) or bottom (False)
    nchan : int
        number of frequency channels for FFT
    nt, ntint : int
        total number nt of sets, each containing ntint samples in each file
        hence, total # of samples is nt*ntint, with each sample containing
        a single polarisation
    nhead : int
        number of bytes to skip before reading (usually 0 for ARO)
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
        start of part of the file that is read (i.e., ignoring nhead)
    dedisperse : None or string
        None, 'incoherent', 'coherent', 'by-channel'
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

    # size in bytes of records read from file (simple for ARO: 1 byte/sample)
    recsize = nchan*ntint*{np.int8: 2, '4bit': 1}[dtype]
    if verbose:
        print('Reading from {}'.format(fh1))

    if nhead > 0:
        if verbose:
            print('Skipping {0} bytes'.format(nhead))
        fh1.seek(nhead)

    foldspec = np.zeros((nchan, ngate), dtype=np.int)
    icount = np.zeros((nchan, ngate), dtype=np.int)

    dt1 = (1./samplerate).to(u.s)
    # need 2*nchan real-valued samples for each FFT
    dtsample = nchan * 2 * dt1

    # pre-calculate time delay due to dispersion in course channels
    freq = (fedge - rfftfreq(nchan*2, dt1.value) * u.Hz
            if fedge_at_top
            else
            fedge + rfftfreq(nchan*2, dt1.value) * u.Hz)
    # [::2] sets frequency channels to numerical recipes ordering
    dt = (dispersion_delay_constant * dm *
          (1./freq[::2]**2 - 1./fref**2)).to(u.s).value
    if dedisperse in {'coherent', 'by-channel'}:
        # pre-calculate required turns due to dispersion
        fcoh = (fedge - rfftfreq(nchan*2*ntint, dt1.value) * u.Hz
                if fedge_at_top
                else
                fedge + rfftfreq(nchan*2*ntint, dt1.value) * u.Hz)
        # set frequency relative to which dispersion is coherently corrected
        if dedisperse == 'coherent':
            _fref = fref
        else:
            # _fref = np.round((fcoh * dtsample).to(1).value) / dtsample
            _fref = np.repeat(freq.value, ntint) * freq.unit
        # (check via eq. 5.21 and following in
        # Lorimer & Kramer, Handbook of Pulsar Astrono
        dang = (dispersion_delay_constant * dm * fcoh *
                (1./_fref-1./fcoh)**2) * 360. * u.deg
        # order of frequencies is r[0], r[1],i[1],...r[n-1],i[n-1],r[n]
        # for 0 and n need only real part, but for 1...n-1 need real, imag
        # so just get shifts for r[1], r[2], ..., r[n-1]
        dang = dang.to(u.rad).value[1:-1:2]
        dd_coh = np.exp(dang * 1j).conj().astype(np.complex64)

    for j in xrange(nt):
        if verbose and j % progress_interval == 0:
            print('Doing {:6d}/{:6d}; time={:18.12f}'.format(
                j+1, nt, dtsample.value*j*ntint))   # time since start

        # just in case numbers were set wrong -- break if file ends
        # better keep at least the work done
        try:
            # data just a series of bytes, each containing one 8 bit or
            # two 4-bit samples (set by dtype in caller)
            raw = fromfile(fh1, dtype, recsize)
        except(EOFError, IOError) as exc:
            print("Hit {}; writing pgm's".format(exc))
            break
        if verbose == 'very':
            print("Read {} items".format(raw.size), end="")

        vals = raw.astype(np.float32)
        if dedisperse in {'coherent', 'by-channel'}:
            fine = rfft(vals, axis=0, overwrite_x=True)
            fine_cmplx = fine[1:-1].view(np.complex64)
            fine_cmplx *= dd_coh  # this overwrites parts of fine, as intended
            vals = irfft(fine, axis=0, overwrite_x=True)
            if verbose == 'very':
                print("... dedispersed", end="")

        chan2 = rfft(vals.reshape(-1, nchan*2), axis=-1,
                     overwrite_x=True)**2
        # rfft: Re[0], Re[1], Im[1], ..., Re[n/2-1], Im[n/2-1], Re[n/2]
        # re-order to Num.Rec. format: Re[0], Re[n/2], Re[1], ....
        power = np.hstack((chan2[:,:1]+chan2[:,-1:],
                           chan2[:,1:-1].reshape(-1,nchan-1,2).sum(-1)))
        if verbose == 'very':
            print("... power", end="")

        # current sample positions in stream
        isr = j*ntint + np.arange(ntint)

        if do_waterfall:
            # loop over corresponding positions in waterfall
            for iw in xrange(isr[0]//ntw, isr[-1]//ntw + 1):
                if iw < nwsize:  # add sum of corresponding samples
                    waterfall[:,iw] += np.sum(power[isr//ntw == iw],
                                              axis=0)
            if verbose == 'very':
                print("... waterfall", end="")

        if do_foldspec:
            tsample = dtsample.value*isr  # times since start

            for k in xrange(nchan):
                if dedisperse == 'coherent':
                    t = tsample  # already dedispersed
                else:
                    t = tsample - dt[k]  # dedispersed times

                phase = phasepol(t)  # corresponding PSR phases
                iphase = np.remainder(phase*ngate,
                                      ngate).astype(np.int)
                # sum and count samples by phase bin
                foldspec[k] += np.bincount(iphase, power[:,k], ngate)
                icount[k] += np.bincount(iphase, None, ngate)

            if verbose == 'very':
                print("... folded", end="")

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
                if verbose == 'very':
                    print("... added", end="")
        if verbose == 'very':
            print("... done")

    if verbose:
        print('read {0:6d} out of {1:6d}'.format(j+1, nt))

    if do_waterfall:
        nonzero = waterfall == 0.
        waterfall -= np.where(nonzero,
                              np.sum(waterfall, 1, keepdims=True) /
                              np.sum(nonzero, 1, keepdims=True), 0.)

    return foldspec2, waterfall
