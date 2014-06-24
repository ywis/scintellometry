import sys
import os

import numpy as np

def mk_packet_dtype(nframe, nfreq, ninput):

    packet_dtype = np.dtype([('valid', 'u4'), ('unused_header', '26b'),
                             ('n_frames', 'u4'), ('n_input', 'u4'), ('n_freq', 'u4'), ('offset_freq', 'u4'),
                             ('seconds', 'u4'), ('micro_seconds', 'u4'), ('seq', '<u4'), ('data', '(%i,%i,%i)u1' % (nframe, nfreq, ninput)) ])

    return packet_dtype

# Outputfile
outfile = sys.argv[1]

## Try and read the first header to determine the file format.

# Create the header datatype
header_dtype = mk_packet_dtype(0, 0, 0)
head_size = header_dtype.itemsize

# Read the header and parse it using numpy
head_buf = sys.stdin.read(head_size)
header = np.fromstring(head_buf, dtype=header_dtype, count=1)[0]

if not header['valid']:
    raise Exception("Invalid stream.")

# Format header info.
pinfo = { 'nframe': header['n_frames'],
          'nfreq': header['n_freq'],
          'ninput': header['n_input'],
          'sfreq': header['offset_freq'],
          'stime': header['seconds'] + 1e-6 * header['micro_seconds'] }

# Print out metadata
print "========================================================"
print "    Number freq: %i" % pinfo['nfreq']
print "    Start freq channel: %i" % pinfo['sfreq']
print "    Number input: %i" % pinfo['ninput']
print "    Array shape (..., %i, %i, %i)" % (pinfo['nfreq'], pinfo['ninput'], 2)
print "    Datatype: signed char"
print "    NTP start time (epoch): %f" % pinfo['stime']
print "========================================================"

# Write meta data to file
metafile = outfile + ".meta"

print "Writing metadata to %s" % metafile
with open(metafile, 'w+') as f:
    f.write(repr(pinfo) + '\n')

# Create datatype representing the standard packet
pk_dtype = mk_packet_dtype(pinfo['nframe'], pinfo['nfreq'], pinfo['ninput'])

# Define the amount of packets we will read at each loop
npk = 1024
pksize = pk_dtype.itemsize
read_size = pksize * npk

first_run = True

print "Writing binary data to %s" % outfile

tframes = 0

with open(outfile, 'wb+') as f:

    while True:

        # If this is the first run we need to make up for the fact we already read
        # the first header.
        if first_run:
            buf = head_buf + sys.stdin.read(read_size - head_size)
            first_run = False
        else:
            # Otherwise just read the normal buffer amount.
            buf = sys.stdin.read(read_size)

        # Check to see if we are at the end.
        if not buf:
            print "Hit end of the pipe. Exiting."
            break

        # Check if we have a shorter than requested amount (probably near the end of the stream)
        npkread = len(buf) / pksize
        if npkread != npk:
            print "Read only %i packets." % npkread

        tframes += npkread * pinfo['nframe']

        # Use numpy to parse the data
        npb = np.fromstring(buf, dtype=pk_dtype)

        ## Check if the data is valid
        valid = npb['valid']

        # The first few bytes is standard, unless it's all zeros, anything else
        # and bad things are happening
        if np.logical_and(valid != 0xffffffff, valid != 0).any():
            print "Corrupt data, or got confused about offset."
            sys.exit(-1)

        # Check to see if there were any lost packets.
        if (valid == 0).any():
            print "Found %i lost packets" % (valid == 0).sum()

        # Decode the data.
        # cdata = np.zeros(npb['data'].shape + (2,), dtype=np.int8)
        # cdata[..., 0] = (npb['data'] / 16).view(np.int8) - 8
        # cdata[..., 1] = (npb['data'] % 16).view(np.int8) - 8
        # cdata = cdata.reshape((-1,) + cdata.shape[2:])

        # Write the data to the given output file
        #cdata.tofile(f)
        npb['data'].copy().tofile(f)

print "Processed %i frames" % tframes
