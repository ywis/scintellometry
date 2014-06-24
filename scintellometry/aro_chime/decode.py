
import sys
import os
import warnings

import numpy as np

def mk_packet_dtype(nframe, nfreq, ninput):

    packet_dtype = np.dtype([('valid', 'u1'), ('unused_header', '29b'),
                             ('n_frames', 'u4'), ('n_input', 'u4'), ('n_freq', 'u4'), ('offset_freq', 'u4'),
                             ('seconds', 'u4'), ('micro_seconds', 'u4'), ('seq', '<u4'), ('data', '(%i,%i,%i)u1' % (nframe, nfreq, ninput)) ])

    return packet_dtype

def pfile_size(fname):
    """Get the length of a packet file (in packets).

    Parameters
    ----------
    fname : string
        Filename of the file.

    Returns
    -------
    packets : integer
    """
    fsize = os.path.getsize(fname)
    dt = _dtype_from_pinfo(_query_packet(fname))
    psize = dt.itemsize

    if fsize % psize != 0:
        warnings.warn('File not a whole length of packets.')

    return fsize / psize


def _first_seq(fname):
    seq, valid, data = load_pfile(fname, count=1)

    if valid[0]:
        return seq[0]
    else:
        raise Exception("Invalid sequence number.")


def _query_packet(fname):

    dt = mk_packet_dtype(0, 0, 0)

    header = np.fromfile(fname, dtype=dt, count=1)[0]

    if not header['valid']:
        raise Exception("Invalid file.")

    pinfo = { 'nframe': header['n_frames'],
              'nfreq': header['n_freq'],
              'ninput': header['n_input'],
              'sfreq': header['offset_freq'],
              'stime': header['seconds'] + 1e-6 * header['micro_seconds'] }

    return pinfo


def _dtype_from_pinfo(pinfo):

    return mk_packet_dtype(pinfo['nframe'], pinfo['nfreq'], pinfo['ninput'])




def load_pfile(fname, count=None):
    """Load a packet file.

    Parameters
    ----------
    fname : string
        Name of file to load.
    count : integer, optional
        Number of packets to load.

    Returns
    -------
    seq : np.ndarray[npack] dtype=np.int32
        Sequence number from FPGA.
    valid : np.ndarray[npack] dtype=np.bool
        Valid packet.
    cdata : np.ndarray[4*npack, 1024, 2, 2]
        Complex datastream, packed as (frames, freq, antenna, real/imag)
    """

    if not os.path.exists(fname):
        raise Exception("File does not exist.")

    pinfo = _query_packet(fname)
    dt = _dtype_from_pinfo(pinfo)

    npacket = pfile_size(fname)

    if count is None:
        count = npacket
    else:
        if count > npacket:
            warnings.warn("Not enough packets.")
            count = npacket

    dfile = np.fromfile(fname, dtype=dt, count=count)

    seq = dfile['seq'].copy()
    valid = dfile['valid'].astype(np.bool)

    cdata = np.zeros(dfile['data'].shape + (2,), dtype=np.int8)

    cdata[..., 0] = (dfile['data'] / 16).view(np.int8) - 8
    cdata[..., 1] = (dfile['data'] % 16).view(np.int8) - 8

    cdata = cdata.reshape((-1,) + cdata.shape[2:])

    return seq, valid, cdata




def files_to_hdf5(fnames, outfile):
    """Transform a set of packet files into HDF5.

    Parameters
    ----------
    fnames : list of fnames
        Names of files to convert.
    outfile : string
        Name of HDF5 file to output.
    """
    import h5py

    sorted_fnames = sorted(fnames, key=_first_seq)

    pnums = np.array([ pfile_size(fname) for fname in sorted_fnames ])

    cs = np.cumsum(np.concatenate(([0], pnums)))
    pstart = cs[:-1]
    pend = cs[1:]

    pinfo = _query_packet(sorted_fnames[0])

    with h5py.File(outfile) as f:

        sh = (pnums.sum()*pinfo['nframe'], pinfo['nfreq'], pinfo['ninput'], 2)

        f.create_dataset('data', shape=sh, dtype=np.int8)
        f.create_dataset('seq', shape=(pnums.sum(),), dtype=np.int32)

        f.attrs['start_time'] = pinfo['stime']
        f.attrs['start_freq_channel'] = pinfo['sfreq']

        for s, e, fname in zip(pstart, pend, sorted_fnames):
            print "Processing:", fname
            seq, valid, data = load_pfile(fname)
            f['seq'][s:e] = seq
            f['data'][(4*s):(4*e)] = data


def files_to_dat(fnames, outfile):
    """Transform a set of packet files into a binary dump.

    Parameters
    ----------
    fnames : list of fnames
        Names of files to convert.
    outfile : string
        Name of binary dat file to output.
    """
    sorted_fnames = sorted(fnames, key=_first_seq)

    pnum = np.array([ pfile_size(fname) for fname in sorted_fnames ]).sum()

    pinfo = _query_packet(sorted_fnames[0])

    oldsh = None

    with open(outfile, 'w+') as f:

        for fname in sorted_fnames:
            print "Processing:", fname
            seq, valid, data = load_pfile(fname)

            if oldsh is None or oldsh == data.shape:
                oldsh = data.shape
            else:
                raise Exception("Incompatible packet files %s" % fname)

            data.tofile(f)

    print "========================================================"
    print "    Filename: %s" % outfile
    print "    Array shape (%i, %i, %i, %i)" % (pnum * pinfo['nframe'], pinfo['nfreq'], pinfo['ninput'], 2)
    print "    Datatype: signed char"
    print "    Start freq channel: %i" % pinfo['sfreq']
    print "    NTP start time (epoch): %f" % pinfo['stime']
    print "========================================================"


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Decode packet files.')
    parser.add_argument('-f', type=str, choices=['hdf5', 'dat'], default='dat', help='Format to output [default: dat]')
    parser.add_argument('outputfile', type=str, help='Name of output file.')
    parser.add_argument('inputfiles', type=str, nargs='+', help='Input files to process.')

    args = parser.parse_args()

    if os.path.exists(args.outputfile):
        print "Output file already exists. Exiting."
        sys.exit()

    if args.f == 'hdf5':
        files_to_hdf5(args.inputfiles, args.outputfile)
    else:
        files_to_dat(args.inputfiles, args.outputfile)




