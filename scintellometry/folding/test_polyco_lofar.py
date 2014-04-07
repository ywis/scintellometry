from __future__ import division, print_function

import numpy as np
from numpy.polynomial import Polynomial
import astropy.units as u

from fold_lofar import fold
from pmap import pmap

if __name__ == '__main__':
    # pulsar parameters
    # psr = 'B1919+21'
    # psr = 'B2016+28'
    # psr = 'B1957+20'
    # psr = 'B0329+54'
    psr = 'J1810+1744'

    dm_dict = {'B0329+54': 26.833 * u.pc / u.cm**3,
               'J1810+1744': 39.659298 * u.pc / u.cm**3,
               'B1919+21': 12.455 * u.pc / u.cm**3,
               'B1957+20': 29.11680*1.001 * u.pc / u.cm**3,
               'B2016+28': 14.172 * u.pc / u.cm**3}
    phasepol_dict = {'B0329+54': Polynomial([0., 1.399541538720]),
                     'J1810+1744': Polynomial([5123935.3179235281,
                                               601.3858344512422,
                                               -6.8670334150772988e-06,
                                               1.6851467436247837e-10,
                                               1.4924190280848832e-13,
                                               3.681791676784501e-18,
                                               3.4408214917205562e-22,
                                               2.3962705401172674e-25,
                                               2.7467843239802234e-29,
                                               1.3396130966170961e-33,
                                               3.0840132342990634e-38,
                                               2.7633775352567796e-43]),
                     'B1919+21': Polynomial([0.5, 0.7477741603725]),
                     'B2016+28': Polynomial([0., 1.7922641135652])}

    dm = dm_dict[psr]
    phasepol = phasepol_dict[psr]

    igate = None

    file_dict = {
        'J1810+1744': '/mnt/data-pen1/jhessels/J1810+1744/L166111/L166111',
        'B1919+21': '/mnt/data-pen1/jhessels/B1919+21/L166109/L166109'}

    file_fmt = file_dict[psr] + '_SAP000_B000_S{S:1d}_P{P:03d}_bf.raw'
    nskip = 0
    # frequency channels that file contains
    nchan = 20
    ntint = 2**25//4//16  # 16=~nchan -> power of 2, but sets of ~32 MB
    recsize = ntint * nchan
    nt = 110  # number of sets to fold
    ntbin = 10  # number of bins the time series is split into for folding
    ngate = 32  # for 1919 # number of bins over the pulsar period
    ntw = min(1000, nt*ntint)  # number of samples to combine for waterfall

    samplerate = 200. * u.MHz
    fwidth = samplerate / 1024.

    fref = 150. * u.MHz  # ref. freq. for dispersion measure

    coherent = True
    verbose = True
    do_waterfall = False

    foldspecs = []
    icounts = []
    waterfalls = []

    for P in range(20):
        file1 = file_fmt.format(S=0, P=P)
        file2 = file_fmt.format(S=1, P=P)

        fbottom = fwidth*(563.+P*20)

        f2, ic, wf = fold(file1, file2, '>f4',
                          fbottom, fwidth, nchan,
                          nt, ntint, nskip,
                          ngate, ntbin, ntw, dm, fref, phasepol,
                          coherent=coherent, do_waterfall=do_waterfall,
                          verbose=verbose, progress_interval=1)

        np.save("lofar{}foldspec2{}.npy".format(psr, P), f2)
        foldspecs.append(f2)
        icounts.append(ic)
        waterfalls.append(wf)

    foldspec2 = np.concatenate(foldspecs, axis=0)
    np.save("lofar{}foldspec2.npy".format(psr), foldspec2)

    icount = np.concatenate(icounts, axis=0)
    np.save("lofar{}icount.npy".format(psr), icount)

    if do_waterfall:
        waterfall = np.concatenate(waterfalls, axis=0)
        np.save("lofar{}waterfall.npy".format(psr), waterfall)

    f2 = foldspec2.copy()
    foldspec1 = f2.sum(axis=2)
    fluxes = foldspec1.sum(axis=0)
    foldspec3 = f2.sum(axis=0)
    if igate is not None:
        dynspect = foldspec2[:,igate[0]-1:igate[1],:].sum(axis=1)
        dynspect2 = foldspec2[:,igate[2]-1:igate[3],:].sum(axis=1)
        f = open('dynspect'+psr+'.bin', 'wb')
        f.write(dynspect.T.tostring())
        f.write(dynspect2.T.tostring())
        f.close()
    f = open('flux.dat', 'w')
    for i, flux in enumerate(fluxes):
        f.write('{0:12d} {1:12.9g}\n'.format(i+1, flux))
    f.close()
    plots = True
    if plots:
        if do_waterfall:
            w = waterfall.copy()
            pmap('waterfall.pgm', w, 1, verbose=True)
        pmap('folded'+psr+'.pgm', foldspec1, 0, verbose)
        pmap('foldedbin'+psr+'.pgm',
             f2.transpose(0,2,1).reshape(nchan,-1), 1, verbose)
        pmap('folded3'+psr+'.pgm', foldspec3, 0, verbose)
        # open(10,file='dynspect'//psr//'.bin',form='unformatted')
        # write(10) dynspect
        # write(10) dynspect2
        if igate is not None:
            dall = dynspect+dynspect2
            dall_sum0 = dall.sum(axis=0)
            dall_sum0 = np.where(dall_sum0, dall_sum0, 1.)
            dall = dall/(dall_sum0/nchan)
            dall[0,:] = 0
            pmap('dynspect'+psr+'.pgm', dall, 0, verbose)
            t1 = dynspect/(dynspect.sum(axis=0)/nchan)
            t2 = dynspect2/(dynspect2.sum(axis=0)/nchan)
            dsub = t1-t2
            dsub[0,:] = 0
            pmap('dynspectdiff'+psr+'.pgm', dsub, 0, verbose)
