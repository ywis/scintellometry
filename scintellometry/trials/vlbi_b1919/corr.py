from __future__ import division, print_function

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
    Obs = obsdata()
    obskey1 = Obs['lofar'].nearest_observation('2013-07-25')
    obskey2 = Obs['gmrt'].nearest_observation('2013-07-25')
    psr1 = Obs['lofar'][obskey1]['src']
    psr2 = Obs['gmrt'][obskey2]['src']
    files1 = Obs['lofar'].file_list(obskey1)
    files2 = Obs['gmrt'].file_list(obskey2)

    assert psr1 == psr2
    dm = Obs['psrs'][psr1]['dm']
    with LOFARdata_Pcombined(*files1, comm=comm) as fh1,\
            GMRTdata(*files2, comm=comm) as fh2:
        correlate.correlate(fh1, fh2, dm=dm, nchan=512, ngate=512,
                            ntbin=12, nt=12, ntint=12, ntw=10200,
                            t0='2013-07-25T22:25:01.0',
                            t1='2013-07-25T22:25:12.0')
