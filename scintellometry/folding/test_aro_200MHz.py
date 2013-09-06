from __future__ import division, print_function

import os
import numpy as np
from numpy.polynomial import Polynomial
import astropy.units as u
from astropy.time import Time, TimeDelta

from fold_aro2 import fold
from pmap import pmap
from multifile import multifile

if __name__ == '__main__':
    # pulsar parameters
    # psr = 'B1919+21'
    # psr = 'B2016+28'
    # psr = 'B1957+20'
    # psr = 'B0329+54'
    # psr = 'B0823+26'
    psr = 'J1810+1744'
    date_dict = {'B0823+26': '2013-07-24T15:06:16',
                 # 'J1810+1744': '2013-07-27T16:55:17'}
                 'J1810+1744': '2013-07-26T16:30:37'}

    dm_dict = {'B0329+54': 26.833 * u.pc / u.cm**3,
               'B0823+26': 19.454 * u.pc / u.cm**3,
               'J1810+1744': 39.659298 * u.pc / u.cm**3,
               'B1919+21': 12.455 * u.pc / u.cm**3,
               'B1957+20': 29.11680*1.001 * u.pc / u.cm**3,
               'B2016+28': 14.172 * u.pc / u.cm**3,
               'noise': 0. * u.pc / u.cm**3}
    phasepol_dict = {'B0329+54': Polynomial([0., 1.399541538720]),
                     'B0823+26': Polynomial([0., 1.88444396743]),
                     '2013-07-27T16:55:17': Polynomial(  # J1810+1744
                         [-1252679.1986725251,
                          601.39629721056895,
                          -6.6664639926379228e-06,
                          -3.005404797321569e-10,
                          1.3404520057431192e-13,
                          3.5632030706667189e-18,
                          -1.0874017282180807e-21,
                          -1.8089896985287676e-26,
                          4.803545433801123e-30,
                          1.4787240038933893e-35,
                          -1.1792841185454315e-38,
                          2.6298912108944255e-43]),
                     '2013-07-26T16:30:37': Polynomial(  # J1810+1744
                         [-4307671.0917832768,
                          601.37394786958396,
                          -5.7640759068738662e-06,
                          6.2468664899676703e-10,
                          1.1429714466878334e-13,
                          -7.5191478615746773e-18,
                          -7.4658136316940933e-22,
                          -1.5804755712584567e-26,
                          1.3208008604369681e-29,
                          -9.5396362858203809e-34,
                          2.7444696554344206e-38,
                          -2.9004096379523196e-43]),
                     'B1919+21': Polynomial([0.5, 0.7477741603725]),
                     'B1957+20': Polynomial([0.18429825167498662,
                                             622.15422173840602,
                                             -9.1859244117739102e-08,
                                             -6.1635896559400589e-11,
                                             2.0877275767457872e-16],
                                            # polyco is for 00:00 UTC midtime,
                                            # while obs starts at 23:44:40,
                                            # or 5*60+20=320 s earlier
                                            [-3600+320, 3600+320],
                                            [-3600,3600]),
                     'B2016+28': Polynomial([0., 1.7922641135652]),
                     'noise': Polynomial([0., 1.])}

    dt = date_dict[psr]
    dm = dm_dict[psr]
    phasepol = phasepol_dict[dt] if dt in phasepol_dict else phasepol_dict[psr]

    igate = None

    # fndir1 = '/mnt/b/algonquin/'
    # fnbase = '/mnt/data-pen1/pen/njones/VLBI_july212013'
    fnbase = '/mnt/aro'
    disk_no = [2, 1, 3]
    seq_file = ('{}/hdd{}_node7/algonquin/sequence.{}.3.dat'
                .format(fnbase, disk_no[0], dt))
    raw_files = ['{}/hdd{}_node7/algonquin/raw_voltage.{}.{}.dat'
                 .format(fnbase, disk_no[i], dt, i) for i in range(3)]

    nchan = 128  # frequency channels to make
    ngate = 64  # number of bins over the pulsar period
    recsize = 2**25  # 32MB sets
    ntint = recsize//nchan  # number of samples after FFT
    # total_size = sum(os.path.getsize(fil) for fil in raw_files)
    # nt = total_size // recsize
    nt = 1000  # each sample 2**25/1e8=0.3355 s

    # possible offset
    nhead = recsize * 1000

    ntbin = 10  # number of bins the time series is split into for folding
    ntw = min(100000, nt*ntint)  # number of samples to combine for waterfall

    samplerate = 200 * u.MHz
    # account for offset, recalling there are 2 samples per byte
    phasepol.window += (nhead * 2 / samplerate).to(u.second).value

    fedge = 200. * u.MHz
    fedge_at_top = True

    fref = 150. * u.MHz  # ref. freq. for dispersion measure

    time0 = Time(date_dict[psr], scale='utc') + TimeDelta(4*3600, format='sec')

    verbose = 'very'
    do_waterfall = True
    do_foldspec = True
    dedisperse = 'by-channel'

    #with open(file1, 'rb') as fh1:
    with multifile(seq_file, raw_files) as fh1:

        foldspec2, waterfall = fold(fh1, '4bit', samplerate,
                                    fedge, fedge_at_top, nchan,
                                    nt, ntint, nhead,
                                    ngate, ntbin, ntw, dm, fref, phasepol,
                                    dedisperse=dedisperse,
                                    do_waterfall=do_waterfall,
                                    do_foldspec=do_foldspec,
                                    verbose=verbose, progress_interval=1)

    if do_waterfall:
        np.save("aro{}waterfall.npy".format(psr), waterfall)

    if do_foldspec:
        np.save("aro{}foldspec2.npy".format(psr), foldspec2)

        f2 = foldspec2.copy()
        f2[0] = 0.
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
        with open('flux.dat', 'w') as f:
            for i, flux in enumerate(fluxes):
                f.write('{0:12d} {1:12.9g}\n'.format(i+1, flux))

    plots = True
    if plots:
        if do_waterfall:
            w = waterfall.copy()
            w[0] = 0.
            pmap('waterfall.pgm', w, 1, verbose=True)
        if do_foldspec:
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
