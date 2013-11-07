# -*- coding: utf-8 -*-
"""Measure LOFAR rms in sub-bands"""
from __future__ import division, print_function

import numpy as np
import astropy.units as u

from scintellometry.folding.fromfile import fromfile

if __name__ == '__main__':
    psr = 'J1810+1744'
    psr = 'B1919+21'

    file_dict = {
        'J1810+1744': '/mnt/data-pen1/jhessels/J1810+1744/L166110/L166110',
        'B1919+21': '/mnt/data-pen1/jhessels/B1919+21/L166109/L166109'}

    file_fmt = file_dict[psr] + '_SAP000_B000_S{S:1d}_P{P:03d}_bf.raw'
    # frequency channels that file contains
    nchan = 20
    ntint = 2**25//4//16  # 16=~nchan -> power of 2, but sets of ~32 MB
    recsize = ntint * nchan
    dtype = '>f4'
    itemsize = np.dtype(dtype).itemsize

    samplerate = 200. * u.MHz
    fwidth = samplerate / 1024.

    allrms1 = []
    allrms2 = []

    for P in range(20):
        file1 = file_fmt.format(S=0, P=P)
        file2 = file_fmt.format(S=1, P=P)
        print('Reading from {}\n         and {}'.format(file1, file2))

        with open(file1, 'rb', recsize*itemsize) as fh1, \
             open(file2, 'rb', recsize*itemsize) as fh2:
            rms1 = []
            rms2 = []
            while True:
                try:
                    # data stored as series of floats in two files,
                    # one for real and one for imaginary
                    raw1 = fromfile(fh1, dtype, recsize).reshape(-1,nchan)
                    raw2 = fromfile(fh2, dtype, recsize).reshape(-1,nchan)
                except(EOFError, ValueError):
                    break
                rms1.append(raw1.std(0))
                rms2.append(raw2.std(0))
                print("rms1,2 = {}, {}".format(rms1[-1],rms2[-1]))

            allrms1 += rms1
            allrms2 += rms2

            out1 = file1.split("/")[-1].replace(".raw","_rms.npy")
            np.save(out1, np.array(rms1))
            out2 = file2.split("/")[-1].replace(".raw","_rms.npy")
            np.save(out2, np.array(rms2))

        np.save("allrms1.npy", np.array(allrms1))
        np.save("allrms2.npy", np.array(allrms2))
