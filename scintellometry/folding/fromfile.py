from __future__ import division, print_function

import numpy as np

shift40 = np.array([4,0], np.int8)
shift76543210 = np.array([7,6,5,4,3,2,1,0], np.int8)
msblsb_bits = np.array([-16, 15], np.int8)
twopiby256 = 2.*np.pi / 256.


def fromfile(fh, dtype, recsize, verbose=False):
    """Read recsize byts, with type dtype which can be bits."""
    npdtype = np.int8 if dtype in ('1bit', '4bit') else dtype
    if verbose:
        print("Reading {} units of dtype={}".format(recsize, npdtype))
    raw = np.fromfile(fh, dtype=npdtype, count=recsize)
    if dtype == '1bit':
        # For a given int8 byte containing bits 76543210
        # left_shift(byte[:,np.newaxis], shift76543210):
        #    [0xxxxxxx, 10xxxxxx, 210xxxxx, 3210xxxx,
        #     43210xxx, 543210xx, 6543210x, 76543210]
        split = np.left_shift(raw[:,np.newaxis], shift76543210).flatten()
        # right_shift(..., 6):
        #    [0000000x, 11111110, 22222221, 33333332,
        #     44444443, 55555554, 66666665, 77777776]
        # so least significant bits go first.
        np.right_shift(split, 6, split)  # explicitly give output for speedup
        # | 1 -> value becomes +1 or -1
        split |= 1
        return split
    elif dtype == '4bit':
        # For a given int8 byte containing bits 76543210
        # left_shift(byte[:,np.newaxis], shift40):  [3210xxxx, 76543210]
        split = np.left_shift(raw[:,np.newaxis], shift40).flatten()
        # right_shift(..., 4):                      [33333210, 77777654]
        # so least significant bits go first.
        np.right_shift(split, 4, split)  # explicitly give output for speedup
        return split
    elif dtype == 'nibble':
        # For a given int8 byte containing bits 76543210
        split = np.bitwise_and(raw[:,np.newaxis], msblsb_bits)
        # now interpretation:
        # lsb=amplitude=0..15; msb=phase=16*(-8..7)=-128..112
        # calculate sqrt(-2*log(1-((idf/16+0.5)/16.)); inplace for speed
        iph = split[:,0]
        idf = split[:,1]
        # idf += 0.5
        # idf /= 16.
        # idf = np.subtract(1., idf, out=idf)
        # idf = np.log(idf, out=idf)
        # idf *= 2.
        # amp = np.sqrt(idf, out=idf)
        amp = (idf.astype(np.float32)+0.5) / 16.
        amp = np.sqrt(-2.*np.log(1.-amp))
        phase = (iph.astype(np.float32) + 8.) * twopiby256
        return amp * np.exp(1.j * np.pi * phase)
    else:
        return raw
