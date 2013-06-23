from __future__ import division, print_function
import numpy as np
from numpy.polynomial import Polynomial

# to compile fortran source, go to scintellometry/folding and run
# f2py --fcompiler=gfortran -m read_gmrt -c fortran/read_gmrt.f90
import read_gmrt
from pmap import pmap


if __name__ == '__main__':
    psr = '1957+20'
    # Fiddle with DM of 1957
    dm = 29.11680 * 1.001
    igate = [1,16,1,8]

    f0 = 622.1678503154773  # 1./p00[ipsr]
    f1 = 1.45706137397e-06/2.  # 0.
    t0 = -1/f0/3.
    phasepol = Polynomial([f0, f1]).integ(1, 0., t0)

    fndir1 = '/mnt/raid-project/gmrt/pen/B1937/1957+20/b'

    file1 = fndir1 + psr + '_pa.raw0.Pol-L1.dat'
    file2 = fndir1 + psr + '_pa.raw0.Pol-L2.dat'

    half_data_bug = True
    paired_samples_bug = True
    integer1_data = True
    real4_data = not integer1_data

    nhead = 0*32*1024*1024
    nblock = 512  # frequency samples in a block, each two bytes: real, imag
    # nt=45 for 1508, 180 for 0809 and 156 for 0531
    nt = 1024//2*8*2  # number of sets to fold  -> /128 for quick try
    ntint = 1024*32*1024//(nblock*2)//4  # total # of blocks per set
    ngate = 32//2  # number of bins over the pulsar period
    ntbin = 16*1  # number of bins the time series is split into for folding
    ntw = min(10000, nt*ntint)  # number of samples to combine for waterfall

    samplerate = 33333955.033217516*2

    # 327 MHz observations
    fbottom = 306.        # MHz
    fband = 2*16.6666666  # MHz
    # 150 MHz observations
    # fbottom = 156.
    # fband = -16.6666666

    verbose = True
    phase_coeff = phasepol.coef
    foldspec2, waterfall = read_gmrt.fold(nhead, nblock, nt, ntint,
                                          ngate, ntbin, ntw,
                                          dm, phase_coeff,
                                          file1, file2, samplerate,
                                          fbottom, fband,
                                          integer1_data, real4_data,
                                          half_data_bug, paired_samples_bug,
                                          verbose=verbose,
                                          progress_interval=10)
    foldspec1 = foldspec2.sum(axis=2)
    fluxes = foldspec1.sum(axis=0)
    foldspec3 = foldspec2.sum(axis=0)
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
        pmap('waterfall.pgm', waterfall, 1, verbose=True)
        pmap('folded'+psr+'.pgm', foldspec1, 0, verbose)
        pmap('foldedbin'+psr+'.pgm', foldspec2.reshape(nblock,-1), 1, verbose)
        pmap('folded3'+psr+'.pgm', foldspec3, 0, verbose)
        # open(10,file='dynspect'//psr//'.bin',form='unformatted')
        # write(10) dynspect
        # write(10) dynspect2
        dall = dynspect+dynspect2
        dall_sum0 = dall.sum(axis=0)
        dall_sum0 = np.where(dall_sum0, dall_sum0, 1.)
        dall = dall/(dall_sum0/nblock)
        dall[0,:] = 0
        pmap('dynspect'+psr+'.pgm', dall, 0, verbose)
        t1 = dynspect/(dynspect.sum(axis=0)/nblock)
        t2 = dynspect2/(dynspect2.sum(axis=0)/nblock)
        dsub = t1-t2
        dsub[0,:] = 0
        pmap('dynspectdiff'+psr+'.pgm', dsub, 0, verbose)
