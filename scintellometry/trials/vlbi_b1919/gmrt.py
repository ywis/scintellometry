from __future__ import division

from __future__ import division, print_function

import numpy as np
from numpy.polynomial import Polynomial
import astropy.units as u
from astropy.time import Time, TimeDelta

from scintellometry.folding.twofile import twofile
from scintellometry.folding.fold_gmrt_phased import fold
from scintellometry.folding.pmap import pmap


if __name__ == '__main__':
    # pulsar parameters
    psr = 'B1919+21'
    date = '26jul2013'
    # psr = 'B2016+28'
    # psr = 'B0329+54'
    # psr = 'B0823+26'
    # psr = 'J1810+1744'
    psrname = ['0809+74','1508+55','1957+20','1919+21']
    dm_dict = {'B0329+54': 26.833 * u.pc / u.cm**3,
               'B0823+26': 19.454 * u.pc / u.cm**3,
               'J1810+1744': 39.659298 * u.pc / u.cm**3,
               'B1919+21': 12.455 * u.pc / u.cm**3,
               'B1957+20': 29.11680*1.001 * u.pc / u.cm**3,
               'B2016+28': 14.172 * u.pc / u.cm**3,
               'noise': 0. * u.pc / u.cm**3}
    phasepol_dict = {'B0329+54': Polynomial([0., 1.399541538720]),
                     'B1919+21': Polynomial([0.5, 0.7477741603725]),
                     'B2016+28': Polynomial([0., 1.7922641135652])}

    dm = dm_dict[psr]
    phasepol = phasepol_dict[psr]

    file_dict = {
        'B1919+21': '/mnt/data-pen1/bahmanya/tape_6/temp1/phased_array'}

    file_template = (file_dict[psr] + '/node{0:2d}/' + date + '/' +
                     psr.lower() + '.raw.Pol-{1:1s}{2:1d}.dat')
    node = 33
    pol = 'R'
    file1 = file_template.format(node, pol, 1)
    file2 = file_template.format(node, pol, 2)

    nhead = 0
    # frequency samples in a block; every sample is two bytes: real, imag
    nchan = 512
    ngate = 128  # number of bins over the pulsar period
    ntint = 2**22 // 2 // nchan  # no. of bytes -> real,imag -> channels
    nt = 600  # 5 min/16.667 MHz=1200*2**22
    ntbin = 12  # number of bins the time series is split into for folding
    ntw = min(1000, nt*ntint)  # number of samples to combine for waterfall

    samplerate = 100.*u.MHz / 3.

    fedge = 156. * u.MHz
    fedge_at_top = True

    fref = 150. * u.MHz  # ref. freq. for dispersion measure

    time0 = (Time('2013-07-26T03:43:08', scale='utc') -
             TimeDelta(5.5/24., format='jd'))

    verbose = True
    do_waterfall = True
    do_foldspec = True
    dedisperse = 'incoherent'

    print(file1, file2)

    with twofile([file1, file2]) as fh1:

        foldspec2, waterfall = fold(fh1, np.int8, samplerate,
                                    fedge, fedge_at_top, nchan,
                                    nt, ntint, nhead,
                                    ngate, ntbin, ntw, dm, fref, phasepol,
                                    dedisperse=dedisperse,
                                    do_waterfall=do_waterfall,
                                    do_foldspec=do_foldspec,
                                    verbose=verbose, progress_interval=1)

    if do_waterfall:
        np.save("gmrt{0}waterfall.npy".format(psr), waterfall)

    if do_foldspec:
        np.save("gmrt{0}foldspec2_{1}".format(psr, pol), foldspec2)
        f2 = foldspec2.copy()
        foldspec1 = f2.sum(axis=2)
        fluxes = foldspec1.sum(axis=0)
        foldspec3 = f2.sum(axis=0)

        with open('gmrt{0}flux_{1}.dat'.format(psr, pol), 'w') as f:
            for i, flux in enumerate(fluxes):
                f.write('{0:12d} {1:12.9g}\n'.format(i+1, flux))

    plots = True
    if plots:
        if do_waterfall:
            w = waterfall.copy()
            pmap('gmrt{0}waterfall_{1}.pgm'.format(psr, pol),
                 w, 1, verbose=True)
        if do_foldspec:
            pmap('gmrt{0}folded_{1}.pgm'.format(psr, pol),
                 foldspec1, 0, verbose)
            pmap('gmrt{0}foldedbin_{1}.pgm'.format(psr, pol),
                 f2.transpose(0,2,1).reshape(nchan,-1), 1, verbose)
            pmap('gmrt{0}folded3_{1}.pgm'.format(psr, pol),
                 foldspec3, 0, verbose)
