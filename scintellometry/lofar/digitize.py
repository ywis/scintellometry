# -*- coding: utf-8 -*-
""" Routines to convert raw LOFAR data between 4-byte floats and 1-byte integers.

 Main routine is convert_dtype

 Command Line usage: digitize.py [-h] [-o OUTDIR] [-s NSIG] [--refloat] [-nc] [-v]
                   file_SAPXXX_BEAMXXX_SX_name.h5
                   [file_SAPXXX_BEAMXXX_SX_name.h5 ...]

 Notes:
 * we can only process files with one SUB_ARRAY_POINTING, one Beam, and one STOKES
 * we assume the .raw files can be found from file.replace('.h5', '.raw')
 * the float32 to int8 conversion creates a new h5 Dataset at the Beam level
   with the name convention STOKES_X_i2f. 
   This array provides the scaling/mapping from int8 back to float32.
   The dataset also has attributes 'STOKES_X_nsig', preserving the NSIG argument
                                   'STOKES_X_recsize', preserving the read buffer size
   The dataset is deleted if direction='i2f', i..e. 'refloat'ing the byte stream
 * digitization clips extreme fluctuations (nsig), so you may not recover the original
   float stream. Noise will be clipped.
 * We modify the STOKES_X 'DATATYPE' attribute, changing it from 'float' to 'int8' if direction == 'f2i'
                                                                 'int8' to 'float' if direction == 'i2f'
   
 """
from __future__ import division, print_function

import argparse
import numpy as np
import os
import re
import shutil

import h5py

_lofar_dtypes = {'float':'>f4', 'int8':'>i1'}

def convert_dtype(files, outdir, nsig=5., direction='f2i', check=True, verbose=None):
    """
    Given a list of lofar h5 files, digitize the raw data to 'int8', 
    outputting it in outdir.

    Args:
    files : list of h5 files
    outdir : directory to output the digitized raw data and the h5 file
             with the processing history

    nsig : digitize/map raw data stream from 
             (-Nsig*std(), +Nsig*std())
           to
             [-128, 128) (signed int8)
           This process clips outliers.
           (default: 5.)

    direction : One of 'f2i' or 'i2f'. The 'i2f' routine undoes the original
                digitization and should reproduce the original data.
            (deafult: 'f2i')

    check : Check the h5 structure is consistent with the filename.
            (default: True)

    Notes:
     * we can only process files with one SUB_ARRAY_POINTING, one Beam, and one STOKES
     * we assume the .raw files can be found from file.replace('.h5', '.raw')
     * the float32 to int8 conversion creates a new h5 Dataset at the Beam level
       with the name convention STOKES_X_i2f. 
       This array provides the scaling/mapping from int8 back to float32.
       The dataset also has attributes 'STOKES_X_nsig', preserving the NSIG argument
                                       'STOKES_X_recsize', preserving the read buffer size
       The dataset is deleted if direction='i2f', i..e. 'refloat'ing the byte stream
     * digitization clips extreme fluctuations (nsig), so you may not recover the original
       float stream. Noise will be clipped.
     * We modify the STOKES_X 'DATATYPE' attribute, changing it from 'float' to 'int8' if direction == 'f2i'
                                                                     'int8' to 'float' if direction == 'i2f'

    """
    if not isinstance(files, list):
        files = [files]

    assert direction in ['f2i', 'i2f']

    for fname in files:
        fraw = fname.replace('.h5', '.raw')
        fraw_out = os.path.join(outdir, os.path.basename(fraw))
        if direction == 'f2i':
            print("Digitizing %s to %s" % (fraw, fraw_out))
        elif direction == 'i2f':
            print("unDigitizing %s to %s" % (fraw, fraw_out))
 
        if os.path.abspath(fraw) == os.path.abspath(fraw_out):
            print("Warning, this will overwrite input files")

        # copy the original h5 file to the outdir
        fnew = os.path.join(outdir, os.path.basename(fname))
        shutil.copy2(fname, fnew)

        # get necessary processing information, possibly checking for consistency
        nchan, ntint, dtype, nsamples, sap, beam, stokes =  lofar_h5info(fnew, check=check)
        if verbose:
            print("\t Processing SAP%s_B%s_S%s" % (sap, beam, stokes))

        recsize = ntint * nchan
        itemsize_i = np.dtype(dtype).itemsize
        if direction == 'f2i':
            itemsize_o = np.dtype('>i1').itemsize
        elif direction == 'i2f':
            itemsize_o = np.dtype('>f4').itemsize

        with open(fraw, 'rb', recsize*itemsize_i) as fh1,\
                open(fraw_out, 'wb', recsize*itemsize_o) as fh1out,\
                h5py.File(fnew, 'a') as h5:
            if verbose >= 1:
                print("\t",fraw," has dtype", dtype)
            # create/access dataset used to convert back to floats from int8's
            if direction == 'f2i':            
                h5[sap][beam][stokes].attrs['DATATYPE'] = 'int8'
                diginfo = h5[sap][beam].create_dataset("%s_i2f" % stokes, (0, nchan), maxshape=(None, nchan), dtype='f')
                diginfo.attrs['%s_recsize' % stokes] = recsize
                diginfo.attrs['%s_nsig' % stokes] = nsig
            elif direction == 'i2f':
                h5[sap][beam][stokes].attrs['DATATYPE'] = 'float'
                diginfo = h5[sap][beam]["%s_i2f" % stokes]

            idx = 0
            while True:
                raw = fh1.read(recsize*itemsize_i)
                if len(raw) == 0:
                    break

                raw1 = np.fromstring(raw, dtype=dtype).reshape(-1,nchan)

                if direction == 'f2i':
                    # record the stddev for possible recovery
                    std = raw1.std(axis=0)
                    scale = nsig*std
                    digshape = diginfo.shape
                    diginfo.resize(digshape[0] + 1, axis=0)

                    # apply the digitization
                    # by variance in individual channels
                    scale = nsig*std
                    intmap = (128*(raw1/scale))
                    diginfo[idx] = scale/128.
                    # clip
                    if verbose >= 2:
                        nclip = np.sum(intmap> 127) + np.sum(intmap<-128)
                        print("Read# {}, Clipping {}/{} samples".format(idx, nclip, intmap.size))
                    raw1_o = np.clip(intmap, -128, 127).astype('>i1')
                    
                elif direction == 'i2f':
                    # convert back to float32
                    scale = diginfo[idx]
                    raw1_o = (raw1*scale).astype('>f4')

                raw1_o.tofile(fh1out)

                if verbose >= 3:
                    print("f2i rescale = {}".format(scale))
                idx += 1

            if direction == 'i2f':
                del h5[sap][beam]["%s_i2f" % stokes]


def lofar_h5info(fname, check=True):
    """
    open the lofar hdf5 file and return:
    (nchan, ntint, recsize, dtype, SUBARRRAY_POINTING, BEAM, STOKES)

    Note: we only process files with one SUB_ARRAY_POINTING
    and one BEAM per SUB_ARRAY_POINTING

    Args:
    fname : h5 file
    check : check that some of the h5 structure is consistent with
            the filename. For instance, there is only one 
            SUB_ARRAY_POINTING and only one BEAM per file.
            *_SAP???_B???_S?_*

    """
    f5 = h5py.File(fname, 'r')

    # get SAP, beam, stokes from filename
    info = re.search('SAP\d{3}_B\d{3}_S\d{1}_', fname) 
    if info is None:
        print("small warning, filename conventions not met. ", end='')
        #if not check:
        #    print("Processing first SAP, BEAM, and STOKES.")

    # get SAP, beam, stokes from hdf5 file.
    # recall, we assume there is only one SAP, one BEAM and one STOKES per file
    h5saps = sorted([i for i in f5.keys() if 'SUB_ARRAY_POINTING' in i])
    s0 = f5[h5saps[0]]
    h5beams = sorted([i for i in s0.keys() if 'BEAM' in i])
    b0 = s0[h5beams[0]]
    h5stokes = sorted([i for i in b0.keys() if 'STOKES' in i and 'i2f' not in i])
    st0 = b0[h5stokes[0]]

    if check:
        force_fileconsistency(fname, h5saps, h5beams, h5stokes)

    # future fix: currently only handle NOF_SUBBANDS
    # with 1 channel in each subband
    nchan = st0.attrs['NOF_SUBBANDS']
    dtype = _lofar_dtypes[st0.attrs['DATATYPE']]
    # process sets of ~32 MB, hence // np.log2(nchan)
    ntint = 2**25//4//int(np.log2(nchan))  # 16=~nchan -> power of 2, but sets of ~32 MB
    nsamples = b0.attrs['NOF_SAMPLES']
    f5.close()
    return (nchan, ntint, dtype, nsamples, h5saps[0], h5beams[0], h5stokes[0])


def force_fileconsistency(fname, h5saps, h5beams, h5stokes):
    """ Raise Warning if the inferred sap, beam, and stokes
        differ from the content of the hdf5 file
    """
    # get subarray_pointing information from the filename
    info = re.search('SAP\d{3}_B\d{3}_S\d{1}_', fname)
    if info is None:
        txt = "Filename %s doesn't adhere to convention *_SAP???_B???_S?_* "
        txt += "processing may proceed with the -nc switch"
        raise Warning(txt)
    fsap = re.search('SAP\d{3}', info.group()).group().strip('SAP')
    fbeam = re.search('_B\d{3}', info.group()).group().strip('_B')
    fstokes = re.search('_S\d', info.group()).group().strip('_S')

    # parse the info from the hdf5 file
    hsap = re.search('\d{3}', h5saps[0]).group()
    hbeam = re.search('\d{3}', h5beams[0]).group()
    hstokes = re.search('\d', h5stokes[0]).group()

    # compare the results, raising warnings as necessary
    if len(h5saps) == 0:
        raise Exception("No SUB_ARRAY_POINTINGS found")
    elif len(h5saps) > 1:
        raise Warning("Too many SUB_ARRAY_POINTINGS in %s." % fname, "Processing first.")
    elif fsap != hsap:
        raise Warning("Filename %s had SAP %s" % (fname, hsap))

    if len(h5beams) == 0:
        raise Exception("No BEAMS found in SAP%s" % saps[0])
    elif len(h5beams) > 1:
        raise Warning("Too many BEAMS in SAP%s." % saps[0], "Processing first")
    elif fbeam != hbeam:
        raise Warning("Filename %s had BEAM %s" % (fname, hbeam))

    if len(h5stokes) == 0:
        raise Exception("No STOKES found in SAP%s_B%s" % (saps[0],beams[0]))
    elif len(h5stokes) > 1:
        raise Warning("Too many STOKES found in SAP%s_B%s." % (saps[0], beams[0]), "Processing first")
    elif fstokes != hstokes:
        raise Warning("Filename %s had STOKES %s" % (fname, hstokes))


def CL_parser():
    parser = argparse.ArgumentParser(prog='digitize.py',
                                     description='digitize beamformed lofar data to int8, or convert back to original float32. '
                                     "Takes argument list of lofar hdf5 header files, raw files determined by filename.replace('.h5','.raw'). "
                                     'The digitized output copies the hdf5 file to ODIR, but with the processing information. '
                                     "Note, we can only process files with one SUB_ARRAY_POINTING, one BEAM, and one STOKES", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('files', metavar='file_SAPXXX_BEAMXXX_SX_name.h5', nargs='+',
                        help='lofar hdf5 header files to process.')
    
    parser.add_argument('-o', '--output_dir', type=str, dest='outdir',
                        default='./',
                        help='output directory for the digitized files')
 
    parser.add_argument('-s', '--nsigma', type=float, dest='nsig',
                        default=5.0,
                        help='Clip raw data above this many stddevs.')

    parser.add_argument('--refloat', action='store_true',
                        help='Convert int8 back to float32 ("i2f").'
                             'This only works on files having gone through the digitization process.')

    parser.add_argument('-nc', '--nocheck', action='store_true',
                        help='do not enforce filename consistency of *_SAP???_B???_S?_* with the actual hdf5 '
                        'structure for SUB_ARRAY_POINTING_???, BEAM_???, and STOKES_? .')

    parser.add_argument('-v', '--verbose', action='append_const', const=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = CL_parser()
    check = not args.nocheck
    args.verbose = 0 if args.verbose is None else sum(args.verbose)
    if args.refloat:
        direction = 'i2f'
    else:
        direction = 'f2i'
    convert_dtype(args.files, args.outdir, args.nsig, direction=direction, check=check, verbose=args.verbose)
