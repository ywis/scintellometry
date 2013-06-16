from __future__ import division, print_function

import numpy as np


def fold(nhead, nblock, nt, ntint, ngate, ntbin, ntw,
         dm, t0, f0, f1, file1, file2, samplerate, fbottom, fband,
         do_waterfall=True, do_foldspec=True, verbose=True):

    foldspec2 = np.zeros((nblock, ngate,ntbin))
    nwsize = nt*ntint//ntw
    waterfall = np.zeros((nblock, nwsize))

    end_of_file = False

    recsize = nblock*ntint

    with open(file1, 'rb', recsize) as fh1, open(file2, 'rb', recsize) as fh2:

        if nhead > 0:
            print('Skipping {0} bytes'.format(nhead))
            fh1.seek(nhead)
            fh2.seek(nhead)

        foldspec = np.zeros((nblock, ngate), dtype=np.int)
        icount = np.zeros((nblock, ngate), dtype=np.int)

        # calculate time delay due to dispersion, relative to
        # the bottom of the band
        freq = fbottom + fband*np.arange(1,nblock+1)/nblock
        dt = 4149. * dm * (1./freq**2 - 1./fbottom**2)

        dtsample = 2*nblock/samplerate  # time for one sample = FFT of block
        for j in xrange(nt):
            if verbose and (j+1) % 100 == 0:
                print('Doing={:6d}/{:6d}; time={:18.12f}'.format(
                    j+1, nt, dtsample*j*ntint))   # equivalent time since start

            # in phased-array mode, only half the data get written?
            if j % 16 < 8*1:  # was mod(j,16) in fortran code, oddly
                for i in xrange(2):
                    fh = fh1 if i < 1 else fh2
                    try:
                        raw4 = np.fromfile(fh, dtype=np.int8,
                                           count=recsize).reshape(-1,nblock,2)
                    except:
                        end_of_file = True
                        break

                    cbufx = raw4.astype(np.int32)
                    cbufx[1::2] -= cbufx[::2]

                    abscbufx2 = cbufx[:,:,0]**2 + cbufx[:,:,1]**2

                    # if j % 16 == 0 and i == 0:
                    #     print("cbuf test",
                    #           np.sum(cbufx[1]*np.conjugate(cbufx[0])) /
                    #           np.sqrt(np.sum(np.abs(cbufx[1])**2) *
                    #                   np.sum(np.abs(cbufx[0])**2)))

                    if do_waterfall:
                        # current sample positions in stream
                        isr = j*ntint + i*ntint//2 + np.arange(ntint//2)
                        # loop over corresponding positions in waterfall
                        for iw in xrange(isr[0]//ntw, isr[-1]//ntw + 1):
                            if iw < nwsize:  # add sum of corresponding samples
                                waterfall[:,iw] += np.sum(
                                    abscbufx2[isr//ntw == iw], axis=0)

                    if do_foldspec:
                        tsample = dtsample*(j*ntint + i*ntint//2 +
                                            np.arange(ntint//2))

                        for k in xrange(nblock):
                            t = tsample - dt[k] - t0
                            phase = t*(f0 + t*f1/2)
                            iphase = np.remainder(phase*ngate,
                                                  ngate).astype(np.int)
                            foldspec[k] += np.bincount(iphase, abscbufx2[:,k],
                                                       ngate)
                            icount[k] += np.bincount(iphase, None, ngate)

            if end_of_file:
                break
            if do_foldspec:
                ibin = j*ntbin//nt  # bin in the time series: 0..ntbin-1
                if (j+1)*ntbin//nt > ibin:  # last addition to bin?
                    nonzero = icount > 0
                    nfoldspec = np.where(nonzero, foldspec/icount, 0.)
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
