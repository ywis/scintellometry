from __future__ import division, print_function

import numpy as np
from numpy.polynomial import Polynomial
import astropy.units as u
from astropy.time import Time, TimeDelta

from scintellometry.folding.fold_aro2 import fold
from scintellometry.folding.pmap import pmap
from scintellometry.folding.multifile import multifile

if __name__ == '__main__':
    # pulsar parameters
    psr = 'B1919+21'
    # psr = 'B2016+28'
    # psr = 'B0329+54'
    # psr = 'B0823+26'
    # psr = 'J1810+1744'
    date_dict = {'B0823+26': '2013-07-24T15:06:16',
                 'B1919+21': '2013-07-25T18:14:20',
                 # 'J1810+1744': '2013-07-27T16:55:17'}
                 'J1810+1744': '2013-07-26T16:30:37'}

    dm_dict = {'B0329+54': 26.833 * u.pc / u.cm**3,
               'B0823+26': 19.454 * u.pc / u.cm**3,
               'J1810+1744': 39.659298 * u.pc / u.cm**3,
               'B1919+21': 12.455 * u.pc / u.cm**3,
               'B1957+20': 29.11680*1.001 * u.pc / u.cm**3,
               'B2016+28': 14.172 * u.pc / u.cm**3,
               'noise': 0. * u.pc / u.cm**3}
    phasepol_dict = {'2013-07-27T16:55:17':  # J1810+1744
                     Polynomial([-1252679.1986725251,
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
                     '2013-07-26T16:30:37':  # J1810+1744
                     Polynomial([-4307671.0917832768,
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
                     'B1919+21':  # B1919+21/ 2013-07-25T18:14:20
                     Polynomial([0.5, 0.7477741603725])}

    dt = date_dict[psr]
    dm = dm_dict[psr]
    phasepol = phasepol_dict[dt] if dt in phasepol_dict else phasepol_dict[psr]

    # fndir1 = '/mnt/b/algonquin/'
    fnbase = '/mnt/data-pen1/pen/njones/VLBI_july212013'
    disk_no = [2, 1, 3]
    node = 9
    seq_file = ('{0}/hdd{1}_node{2}/sequence.{3}.3.dat'
                .format(fnbase, disk_no[0], node, dt))
    raw_files = ['{0}/hdd{1}_node{2}/raw_voltage.{3}.{4}.dat'
                 .format(fnbase, disk_no[i], node, dt, i) for i in range(3)]

    #***TODO: apply LOFAR polyphase instead
    nchan = 128  # frequency channels to make
    ngate = 128  # number of bins over the pulsar period
    recsize = 2**25  # 32MB sets
    ntint = recsize*2//2//nchan  # number of samples after FFT
    # total_size = sum(os.path.getsize(fil) for fil in raw_files)
    # nt = total_size // recsize
    nt = 900  # each 32MB set has 2*2**25/2e8=0.3355 s -> 5 min ~ 900

    #***TODO: ensure start times match LOFAR/GMRT
    # possible offset 240 ~ 40 seconds
    nhead = recsize * 240

    ntbin = 12  # number of bins the time series is split into for folding
    ntw = min(100000, nt*ntint)  # number of samples to combine for waterfall

    samplerate = 200 * u.MHz
    # account for offset, recalling there are 2 samples per byte
    phasepol.window += (nhead * 2 / samplerate).to(u.second).value

    fedge = 200. * u.MHz
    fedge_at_top = True

    fref = 150. * u.MHz  # ref. freq. for dispersion measure

    # convert time to UTC; dates given in EDT
    time0 = Time(date_dict[psr], scale='utc') + TimeDelta(4*3600, format='sec')

    verbose = 'very'
    do_waterfall = True
    do_foldspec = True
    dedisperse = 'incoherent'

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
        np.save("aro{0}waterfall.npy".format(psr), waterfall)

    if do_foldspec:
        np.save("aro{0}foldspec2_{1}".format(psr, node), foldspec2)
        f2 = foldspec2.copy()
        foldspec1 = f2.sum(axis=2)
        fluxes = foldspec1.sum(axis=0)
        foldspec3 = f2.sum(axis=0)

        with open('aro{0}flux_{1}.dat'.format(psr, node), 'w') as f:
            for i, flux in enumerate(fluxes):
                f.write('{0:12d} {1:12.9g}\n'.format(i+1, flux))

    plots = True
    if plots:
        if do_waterfall:
            w = waterfall.copy()
            pmap('aro{0}waterfall_{1}.pgm'.format(psr, node),
                 w, 1, verbose=True)
        if do_foldspec:
            pmap('aro{0}folded_{1}.pgm'.format(psr, node),
                 foldspec1, 0, verbose)
            pmap('aro{0}foldedbin_{1}.pgm'.format(psr, node),
                 f2.transpose(0,2,1).reshape(nchan,-1), 1, verbose)
            pmap('aro{0}folded3_{1}.pgm'.format(psr, node),
                 foldspec3, 0, verbose)
