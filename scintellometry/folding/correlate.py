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
              dedisperse='incoherent', rfi_filter_raw=None, fref=_fref,
              save_xcorr=True, do_foldspec=1, phasepol=None,
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

    # initialize the folded spectrum and waterfall
    foldspec = np.zeros((nchan, ngate, ntbin))
    icount = np.zeros((nchan, ngate, ntbin), dtype=np.int64)
    waterfall = np.zeros((nchan, ngate, ntbin))
    nwsize = nt * ntint // ntw

    # find nearest starttime landing on same sample
    if t0 is None:
        t0 = max(fh1.time0, fh2.time0)
        print("Starting at %s" % t0)
    t0 = Time(t0, scale='utc')
    t1 = Time(t1, scale='utc')

    # prep the fhs for xcorr stream, setting up channelization, dedispersion...
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
                 ( 1./fh.freq**2 - 1./fref**2) ).to(u.s).value
        # number of time bins to np.roll the channels for incoherent dedisperse
        fh.ndt = (fh.dt / fh.dtsample.to(u.s).value)
        fh.ndt = -1 * np.rint(fh.ndt).astype(np.int)

        if dedisperse in ['coherent', 'by-channel']:
            # pre-calculate required turns due to dispersion
            if fh.nchan > 1:
                fcoh = (fh.freq[np.newaxis, :] +
                        fftfreq(ntint, fh.dtsample.value)[:, np.newaxis]
                        * u.Hz)
            else:
                if fh.fedge_at_top:
                    fcoh = fh.fedge - fh.thisfftfreq(nchan * 2 * ntint,
                                                     fh.dt1.value) * u.Hz
                else:
                    fcoh = fh.fedge + fh.thisfftfreq(nchan * 2 * ntint,
                                                     fh.dt1.value) * u.Hz

            #set frequency relative to which dispersion is coherently corrected
            if dedisperse == 'coherent':
                _fref = fref
            else:
                #fref = np.round((fcoh * fh.dtsample).to(1).value)/fh.dtsample
                _fref = np.repeat(fh.freq.value, ntint) * fh.freq.unit
            # (check via eq. 5.21 and following in
            # Lorimer & Kramer, Handbook of Pulsar Astrono
            dang = (dispersion_delay_constant * dm * fcoh *
                    (1./_fref-1./fcoh)**2) * 360. * u.deg

            if fh.thisfftfreq is rfftfreq:
                # order of frequencies is r[0], r[1],i[1],...r[n-1],i[n-1],r[n]
                # for 0 and n need only real part, but for 1...n-1 need real, imag
                # so just get shifts for r[1], r[2], ..., r[n-1]
                dang = dang.to(u.rad).value[1:-1:2]
            else:
                dang = dang.to(u.rad).value

            fh.dd_coh = np.exp(dang * 1j).conj().astype(np.complex64)
    #### done fh setup ###

    ## xcorr setup
    # data-reading params (to help read in same-size time chunks and average
    # onto the same time-and-frequency grids)
    (Rf, Tf, NUf, fkeep, freqs, rows) = data_averaging_coeffs(fh1, fh2)
    nrows = int(min(rows[0] * Tf[1] / Tf[0], rows[1] * Tf[0] / Tf[1]))

    if save_xcorr:
        # output dataset
        outname = "{0}_{1}_{2}_{3}.hdf5".format(fh1.telescope, fh2.telescope,
                                                t0, t1)
        fcorr = h5py.File(outname, 'w')  # driver='mpio', comm=comm)
        ## create the x-corr output file
        # save the frequency grids to help with future TODO: interpolate onto
        # same frequency grid. For now the frequencies fall within same bin
        fcorr.create_dataset('freqs', data=np.hstack([f.to(u.MHz).value
                                                      for f in freqs]))

        # the x-corr data [tsteps, channels]
        dset = fcorr.create_dataset('corr', (nrows, freqs[0].size),
                                    dtype='complex64', maxshape=(None, nchan))
        #tsteps = fcorr.create_dataset('tsteps', shape=(nrows),
        #                              dtype='float64', maxshape=(None))

    # start reading the data
    # this_nskip moves to 't0', rank is for MPI
    raws = [fh.seek_record_read((fh.this_nskip + rank * Rf[i])
                               * fh.blocksize, fh.blocksize * Rf[i])
            for i, fh in enumerate([fh1, fh2])]
    idx = 0
    endread = False
    while np.all([raw.size > 0 for raw in raws]):
        vals = raws #[raw1, raw2]
        chans = [None, None]
        tsample = [None, None]

        # prep the data (channelize, dedisperse, ...)

        for i, fh in enumerate([fh1, fh2]):
            if rfi_filter_raw is not None:
                raws[i], ok = rfi_filter_raw(raws[i], nchan)

            if fh.telescope == 'aro':
                vals[i] = raws[i].astype(np.float32)
            else:
                vals[i] = raws[i]

            if dedisperse in ['coherent', 'by-channel']:
                fine = fh.thisfft(vals[i], axis=0, overwrite_x=True,
                                  **_fftargs)
                if fh.thisfft is rfft:
                    fine_cmplx = fine[1:-1].view(np.complex64)
                    # overwrites parts of fine, as intended
                    fine_cmplx *= fh.dd_coh
                else:
                    fine *= dd_coh
                vals[i] = fh.thisifft(fine, axis=0, overwrite_x=True,
                                      **_fftargs)

            if fh.nchan == 1:
                # ARO data should fall here
                chans[i] = fh.thisfft(vals[i].reshape(-1, nchan * 2), axis=-1,
                                      overwrite_x=True, **_fftargs)
            else:  # lofar and gmrt-phased are already channelised
                chans[i] = vals[i]

            # TODO: profile
            if dedisperse == 'incoherent':
                #chans[i] = np.roll(chans[i], fh.ndt)
                for ci, v in enumerate(fh.ndt):
                    # print("CI",i,ci,v, type(chans[i]), type(chans[i][0,0]))
                    chans[i][...,ci] = np.roll(chans[i][...,ci], v, axis=0)

            # average onto same time grid
            chans[i] = chans[i].reshape(Tf[i], chans[i].shape[0] / Tf[i], -1)\
                      .mean(axis=0)

            # average onto same freq grid
            chans[i] = chans[i][..., fkeep[i]]
            chans[i] = chans[i].reshape(-1, chans[i].shape[1] / NUf[i],
                                        NUf[i]).mean(axis=-1)

            # current sample positions in stream
            # (each averaged onto same time grid)
            isr = idx * rows[i] + np.arange(rows[i])
            tsample[i] = (t0 + isr * fh.dtsample).mjd
            tsample[i] = tsample[i].reshape(-1, Tf[i]).mean(axis=-1)

        print("T",idx, tsample[0][0:4], tsample[1][0:4])
        # x-correlate
        xpower = chans[0] * chans[1].conjugate()

        if do_foldspec and 0:
            # time since start
            tsamples = np.mean(tsamples, axis=0)
            # bin in the time series: 0..ntbin-1
            ibin = idx * ntbin // nt

            # TODO: dedisperse the individual timeseries
            for k in xrange(xpower.shape[1]):
                if dedisperse == 'coherent':
                    t = tsample  # already dedispersed
                elif dedisperse in ['incoherent', 'by-channel']:
                    t = tsample - dt[k]  # dedispersed times
                elif dedisperse is None:
                    t = tsample  # do nothing
                else:
                    t = tsample - dt[k]

                phase = phasepol(t)  # corresponding PSR phases
                iphase = np.remainder(phase*ngate,
                                      ngate).astype(np.int)
                # sum and count samples by phase bin
                foldspec[k, :, ibin] += np.bincount(iphase, np.abs(xpower[:, k]), ngate)
                icount[k, :, ibin] += np.bincount(iphase, np.abs(xpower[:, k]) != 0.,
                                                  ngate)

        if save_xcorr:
            curshape = dset.shape
            nx = max(nrows * (idx + rank), curshape[0])
            dset.resize((nx + nrows, curshape[1]))
            dset[nrows * (idx + rank): nrows * (idx + rank + 1)] = xpower
            # tsteps.resize((nx + nrows))
            # tsteps[nrows * (idx + rank): nrows * (idx + rank + 1)] = tsample1

        # read in next dataset if we haven't hit t1 yet
        for fh in [fh1, fh2]:
            if (fh.time() - t1).sec > 0.:
                endread = True
        if endread:
            break
        else:
            raws = [fh.seek_record_read((fh.this_nskip + (rank + idx) * Rf[i])
                                        * fh.blocksize, fh.blocksize * Rf[i])
                    for i, fh in enumerate([fh1, fh2])]
        print("idx",idx, fh1.time(), fh2.time())
        idx += size

    fcorr.close()
    return foldspec, icount, waterfall

def data_averaging_coeffs(fh1, fh2):
    """
    return the time-and-frequency averaging parameters
    which help get the data on the same grid(s)

    """
    ## read the two filestreams in chunks of equal time
    sr = Fraction(fh1.dtsample.to(u.s).value * fh1.blocksize
                  / fh1.recordsize)
    sr /= Fraction(fh2.dtsample.to(u.s).value * fh2.blocksize
                   / fh2.recordsize)
    sr = sr.limit_denominator(5000)
    nf1 = sr.denominator
    nf2 = sr.numerator

    ## used for re-sizing hdf5 x-corr output
    raw1_nrows = int(fh1.blocksize * nf1 / fh1.recordsize)
    raw2_nrows = int(fh2.blocksize * nf2 / fh2.recordsize)

    ## time averaging params
    Tavg = Fraction(raw1_nrows, raw2_nrows).limit_denominator(1000)
    Tden = Tavg.denominator
    Tnum = Tavg.numerator

    ## channel averaging params
    f1info = (fh1.freq.max(), fh1.freq.min(), len(fh1.freq),
              np.sign(np.diff(fh1.freq).mean()))
    f2info = (fh2.freq.max(), fh2.freq.min(), len(fh2.freq),
              np.sign(np.diff(fh2.freq).mean()))
    f1keep = (fh1.freq > max(f1info[1], f2info[1])) \
        & (fh1.freq < min(f1info[0], f2info[0]))
    f2keep = (fh2.freq > max(f1info[1], f2info[1])) \
        & (fh2.freq < min(f1info[0], f2info[0]))
    # np.save('f1freq.npy', fh1.freq.value)
    # np.save('f2freq.npy', fh2.freq.value)
    Favg = abs(Fraction(np.diff(fh1.freq.value).mean()
                        / np.diff(fh2.freq.value).mean()))
    Favg = Favg.limit_denominator(5000)
    Fden = Favg.denominator
    Fnum = Favg.numerator
    freq1 = fh1.freq[f1keep]
    freq1 = freq1.reshape(freq1.size / Fden, Fden).mean(axis=-1)
    freq2 = fh2.freq[f2keep]
    freq2 = freq2.reshape(freq2.size / Fnum, Fnum).mean(axis=-1)
    # sort low freq to high freq
    if f1info[3] < 0:
        freq1 = freq1[::-1]
    if f2info[3] < 0:
        freq2 = freq2[::-1]

    return ((nf1, nf2), (Tnum, Tden), (Fden, Fnum), (f1keep, f2keep),
            (freq1, freq2), (raw1_nrows, raw2_nrows))
    #    return (nf1, nf2, Tnum, Tden, Fnum, Fden, f1keep, f2keep, freq1, freq2,
    #            raw1_nrows, raw2_nrows)
