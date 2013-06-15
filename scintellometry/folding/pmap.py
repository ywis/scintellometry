from __future__ import division, print_function
import numpy as np

def pmap(fn, rmap1, iscale, verbose=False):
    assert rmap1.ndim == 2
    rmap = rmap1
    iscale1 = iscale
    while iscale1 > 1:
        rmap = np.sqrt(np.abs(rmap))*np.sign(rmap)
        iscale1 -= 1
    rmin, rmax = rmap.min(), rmap.max()
    if verbose:
       print('Contructing {0}; min,max={1},{2}'.format(fn, rmin, rmax))
       
    imap = np.uint8(255*(rmap-rmin)/(rmax-rmin))
    f = open(fn, 'wb')
    pgmHeader = 'P5\n{0:12d}{1:12d}\n{2:12d}\n'.format(rmap.shape[0], 
                                                        rmap.shape[1], 255)
    f.write(pgmHeader)
    f.write(imap.T.tostring())
    f.close()
