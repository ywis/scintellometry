from __future__ import division, print_function

import numpy as np
from astropy.time import Time
from scintellometry.folding.fold import Folder, normalize_counts
from scintellometry.folding.pmap import pmap
from scintellometry.folding import correlate
from scintellometry.folding.filehandlers import (AROdata, LOFARdata_Pcombined,
                                                 GMRTdata)
from observations import obsdata

from mpi4py import MPI


def rfi_filter_raw(raw, nchan):
    # note this should accomodate all data (including's lofar raw = complex)
    rawbins = raw.reshape(-1, 2**11*nchan)  # note, this is view!
    std = rawbins.std(-1, keepdims=True)
    ok = std < MAX_RMS
    rawbins *= ok
    return raw, ok

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    # observing parameters
    verbose = True
    tel1 = 'lofar'
    tel2 = 'gmrt'
    date1 = '2013-07-25'
    date2 = '2013-07-25'
    t0 = Time('2013-07-25T22:25:01.0', scale='utc')
    t1 = Time('2013-07-25T22:25:12.0', scale='utc')
    do_waterfall = True
    do_foldspec = True
    nchan = 512
    ntbin = 12
    Obs = obsdata()
    obskey1 = Obs[tel1].nearest_observation(date1)
    obskey2 = Obs[tel2].nearest_observation(date2)
    psr1 = Obs[tel1][obskey1]['src']
    psr2 = Obs[tel2][obskey2]['src']
    files1 = Obs[tel1].file_list(obskey1)
    files2 = Obs[tel2].file_list(obskey2)

    assert psr1 == psr2
    dm = Obs['psrs'][psr1]['dm']
    with LOFARdata_Pcombined(*files1, comm=comm) as fh1,\
            GMRTdata(*files2, comm=comm) as fh2:
        phasepol1 = Obs['lofar'][obskey1].get_phasepol(fh1.time0)
        phasepol2 = Obs['gmrt'][obskey2].get_phasepol(fh2.time0)
        # out = (foldspec, icount, waterfall)
        out = correlate.correlate(fh1, fh2, dm=dm, nchan=nchan, ngate=512,
                                  ntbin=12, nt=12, ntw=10200,
                                  t0=t0, t1=t1,
                                  phasepol=(phasepol1, phasepol2),
                                  do_waterfall=do_waterfall,
                                  do_foldspec=do_foldspec,
                                  comm=comm)
        myfoldspec = out[0]
        myicount = out[1]
        mywaterfall = out[2]

    savepref = "{0}{1}_{2}chan{3}ntbin".format(tel1[0], tel2[0], nchan, ntbin)
    dt = t1 - t0
    if do_waterfall:
        waterfall = np.zeros_like(mywaterfall)
        comm.Reduce(mywaterfall, waterfall, op=MPI.SUM, root=0)
        if comm.rank == 0:
            # waterfall = normalize_counts(waterfall)
            np.save("{0}waterfall_{1}+{2:08}sec.npy"
                    .format(savepref, t0, dt.sec), waterfall)

    if do_foldspec:
        foldspec = np.zeros_like(myfoldspec)
        icount = np.zeros_like(myicount)
        comm.Reduce(myfoldspec, foldspec, op=MPI.SUM, root=0)
        comm.Reduce(myicount, icount, op=MPI.SUM, root=0)
        if comm.rank == 0:
            fname = ("{0}foldspec_{1}+{2:08}sec.npy")
            iname = ("{0}icount_{1}+{2:08}sec.npy")
            np.save(fname.format(savepref, t0, dt.sec), foldspec)
            np.save(iname.format(savepref, t0, dt.sec), icount)

            # get normalized flux in each bin (where any were added)
            f2 = normalize_counts(foldspec, icount)
            foldspec1 = f2.sum(axis=2)
            fluxes = foldspec1.sum(axis=0)
            foldspec3 = f2.sum(axis=0)

            with open('{0}flux_{1}+{2:08}sec.dat'
                      .format(savepref, t0, dt.sec), 'w') as f:
                for i, flux in enumerate(fluxes):
                    f.write('{0:12d} {1:12.9g}\n'.format(i + 1, flux))

    plots = True
    if plots and comm.rank == 0:
        if do_waterfall:
            w = waterfall.copy()
            try:
                pmap('{0}waterfall_{1}+{2:08}sec.pgm'
                     .format(savepref, t0, dt.sec), w, 1, verbose=True)
            except:
                pass
        if do_foldspec:
            pmap('{0}folded_{1}+{2:08}sec.pgm'
                 .format(savepref, t0, dt.sec), foldspec1, 0, verbose)
            # TODO: Note, I (aaron) don't think this works for LOFAR data
            # since nchan=20, but we concatenate several subband files
            # together, so f2.nchan = N_concat * nchan
            # It should work for my "new" LOFAR_Pconcate file class
            pmap('{0}foldedbin_{1}+{2:08}sec.pgm'
                 .format(savepref, t0, dt.sec),
                 f2.transpose(0, 2, 1).reshape(nchan, -1), 1, verbose)
            pmap('{0}folded3_{1}+{2:08}sec.pgm'
                 .format(savepref, t0, dt.sec), foldspec3, 0, verbose)
