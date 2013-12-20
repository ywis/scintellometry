""" work in progress: need to do lofar-style waterfall and foldspec """
from __future__ import division, print_function

import argparse
import numpy as np
import astropy.units as u
from astropy.time import Time

from scintellometry.folding.fold import Folder, normalize_counts
from scintellometry.folding.pmap import pmap
from scintellometry.folding.filehandlers import (AROdata, LOFARdata,
                                                 LOFARdata_Pcombined, GMRTdata)

from observations import obsdata

from mpi4py import MPI

MAX_RMS = 4.
_fref = 150. * u.MHz  # ref. freq. for dispersion measure


def rfi_filter_raw(raw, nchan):
    # note this should accomodate all data (including's lofar raw = complex)
    rawbins = raw.reshape(-1, 2**11*nchan)  # note, this is view!
    std = rawbins.std(-1, keepdims=True)
    ok = std < MAX_RMS
    rawbins *= ok
    return raw, ok


def rfi_filter_power(power):
    return np.clip(power, 0., MAX_RMS**2 * power.shape[-1])


def reduce(telescope, obsdate, tstart, tend, nchan, ngate, ntbin,
           ntw_min=10200, fref=_fref,
           rfi_filter_raw=None,
           do_waterfall=True, do_foldspec=True, dedisperse=None,verbose=True):
    comm = MPI.COMM_WORLD
    Obs = obsdata()
    # find nearest observation to 'date',
    # warning if > 1s off requested start time of observation
    obskey = Obs[telescope].nearest_observation(obsdate)
    # target of this observation
    psr = Obs[telescope][obskey]['src']
    assert psr in Obs['psrs'].keys()

    dm = Obs['psrs'][psr]['dm']

    files = Obs[telescope].file_list(obskey)
    if telescope == 'aro':
        GenericOpen = AROdata
    elif telescope == 'lofar':
        GenericOpen = LOFARdata
        GenericOpen = LOFARdata_Pcombined
    elif telescope == 'gmrt':
        GenericOpen = GMRTdata

    with GenericOpen(*files, comm=comm) as fh:
        # nchan = None means data is channelized already, so we get this
        # property directly from the file
        if nchan is None or telescope == 'lofar':
            if comm.rank == 0:
                print("LOFAR data already channelized: overriding nchan to {0}"
                      "\n\t as configured in observations.conf"
                      .format(fh.nchan))
            nchan = fh.nchan
        time0 = fh.time0
        phasepol = Obs[telescope][obskey].get_phasepol(time0)
        nt = fh.ntimebins(tstart, tend)
        ntint = fh.ntint(nchan)
        # number of samples to combine for waterfall
        ntw = min(ntw_min, nt*ntint)
        # number of records to skip
        tstart = Time(tstart, scale='utc')
        tend = Time(tend, scale='utc')
        dt = tend - tstart
        nskip = fh.nskip(tstart)    # number of records to skip
        if verbose and comm.rank == 0:
            print("Using start time {0} and phase polynomial {1}"
                  .format(time0, phasepol))
            print("Skipping {0} blocks and folding {1} blocks to cover "
                  "time span {2} to {3}"
                  .format(nskip, nt, tstart, tend))
        fh.seek(nskip * fh.blocksize)
        # set the default parameters to fold
        # Note, some parameters may be in fh's HDUs, or fh.__getitem__
        # but these are overwritten if explicitly sprecified in Folder
        folder = Folder(fh, nchan=nchan,
                        nt=nt, ntint=ntint, ngate=ngate,
                        ntbin=ntbin, ntw=ntw, dm=dm, fref=fref,
                        phasepol=phasepol, dedisperse=dedisperse,
                        do_waterfall=do_waterfall, do_foldspec=do_foldspec,
                        verbose=verbose, progress_interval=1,
                        rfi_filter_raw=rfi_filter_raw,
                        rfi_filter_power=None)
        myfoldspec, myicount, mywaterfall, subint_table = folder(fh, comm=comm)
    # end with

    savepref = "{0}{1}_{2}chan{3}ntbin".format(telescope, psr, nchan, ntbin)
    if do_waterfall:
        waterfall = np.zeros_like(mywaterfall)
        comm.Reduce(mywaterfall, waterfall, op=MPI.SUM, root=0)
        if comm.rank == 0:
            # waterfall = normalize_counts(waterfall)
            np.save("{0}waterfall_{1}+{2:08}sec.npy"
                    .format(savepref, tstart, dt.sec), waterfall)

    if do_foldspec:
        foldspec = np.zeros_like(myfoldspec)
        icount = np.zeros_like(myicount)
        comm.Reduce(myfoldspec, foldspec, op=MPI.SUM, root=0)
        comm.Reduce(myicount, icount, op=MPI.SUM, root=0)
        if comm.rank == 0:
            fname = ("{0}foldspec_{1}+{2:08}sec.npy")
            iname = ("{0}icount_{1}+{2:08}sec.npy")
            np.save(fname.format(savepref, tstart, dt.sec), foldspec)
            np.save(iname.format(savepref, tstart, dt.sec), icount)

            # get normalized flux in each bin (where any were added)
            f2 = normalize_counts(foldspec, icount)
            foldspec1 = f2.sum(axis=2)
            fluxes = foldspec1.sum(axis=0)
            foldspec3 = f2.sum(axis=0)

            with open('{0}flux_{1}+{2:08}sec.dat'
                      .format(savepref, tstart, dt.sec), 'w') as f:
                for i, flux in enumerate(fluxes):
                    f.write('{0:12d} {1:12.9g}\n'.format(i+1, flux))

    plots = True
    if plots and comm.rank == 0:
        if do_waterfall:
            w = waterfall.copy()
            pmap('{0}waterfall_{1}+{2:08}sec.pgm'
                 .format(savepref, tstart, dt.sec), w, 1, verbose=True)
        if do_foldspec:
            pmap('{0}folded_{1}+{2:08}sec.pgm'
                 .format(savepref, tstart, dt.sec), foldspec1, 0, verbose)
            # TODO: Note, I (aaron) don't think this works for LOFAR data
            # since nchan=20, but we concatenate several subband files
            # together, so f2.nchan = N_concat * nchan
            # It should work for my "new" LOFAR_Pconcate file class
            pmap('{0}foldedbin_{1}+{2:08}sec.pgm'
                 .format(savepref, tstart, dt.sec),
                 f2.transpose(0,2,1).reshape(nchan,-1), 1, verbose)
            pmap('{0}folded3_{1}+{2:08}sec.pgm'
                 .format(savepref, tstart, dt.sec), foldspec3, 0, verbose)

    savefits = False
    if savefits and comm.rank == 0:
        print("Saving FITS files...")
        # assign the folded data ( [f2] = [nchan, ngate, ntbin]
        #                          want [ntbin, npol, nchan, ngate]

        # TODO: figure out lofar and aro data, which concatenates the channels
        # so f2 = len(range(P))*nchan
        if telescope == 'lofar':
            f2 = f2[-nchan:]
        elif telescope == 'aro':
            f2 = f2[0:nchan]
        nchan, ngate, ntbin = f2.shape
        f2 = f2.transpose(2, 0, 1)
        f2 = f2.reshape(ntbin, np.newaxis, nchan, ngate)
        std = f2.std(axis=0)
        subint_table[1].data.field('DATA')[:] = (2**16*f2/std).astype(np.int16)
        fout = '{0}folded3_{1}+{2:08}sec.fits'.format(savepref, tstart, dt.sec)
        # Note: output_verify != 'ignore' resets the cards for some reason
        try:
            subint_table.writeto(fout, output_verify='ignore', clobber=True)
        except ValueError:
            print("FITS writings is a work in progress: "
                  "need to come to terms with array shapes")
            print("f2.shape, subint_table[1].data.field('DATA')[:].shape = ",
                  f2.shape, subint_table[1].data.field('DATA')[:].shape)


def CL_parser():
    parser = argparse.ArgumentParser(
        prog='reduce_data.py',
        description='Process data, with options to dedisperse and fold.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--reduction_defaults', type=str,
        help="One of ['aro', 'lofar', 'gmrt'].\n"
        "A convenience flag to set the default parameters as "
        "configured in aro.py, lofar.py, gmrt.py")

    d_parser = parser.add_argument_group(
        "Data-related parameters \n"
        "They specify which observation run to process "
        "(consistent with [telescope], [[date]] entries\n"
        "in observations.conf), and the start and finish timestamps.")
    d_parser.add_argument(
        '-t','--telescope', type=str, default='aro',
        help="The data to reduce. One of ['aro', 'lofar', 'gmrt'].")
    d_parser.add_argument(
        '-d','--date', type=str, default='2013-07-25T18:14:20',
        help="The date of the data to reduce. "
        "Recall observations.conf stores the observation "
        "runs keyed by telescope and date.")
    d_parser.add_argument(
        '-t0', '--starttime', type=str, default='2013-07-25T22:25:01.0',
        help="Timestamp within the observation run to start processing.")
    d_parser.add_argument(
        '-t1', '--endtime', type=str, default='2013-07-25T22:25:12.0',
        help="Timestamp within the observation run to end processing "
        "(replaces the 'nt' argument).")
    d_parser.add_argument(
        '--rfi_filter_raw', action='store_true',
        help="Apply the 'rfi_filter_rwa' routine to the raw data.")

    f_parser = parser.add_argument_group("folding related parameters")
    f_parser.add_argument(
        '-f','--foldspec', type=bool, default=True,
        help="Produce a folded spectrum")
    f_parser.add_argument(
        '-nc', '--nchan', type=int, default=512,
        help="Number of channels in folded spectrum.")
    f_parser.add_argument(
        '-ng', '--ngate', type=int, default=512,
        help="number of bins over the pulsar period.")
    # f_parser.add_argument('-nt', '--nt', type=int, default=1800,
    #   help="number of time bins to fold the data into. ")
    f_parser.add_argument(
        '-nb', '--ntbin', type=int, default=3,
        help="number of time bins the time series is split into for folding.")

    w_parser = parser.add_argument_group("Waterfall related parameters")
    w_parser.add_argument(
        '-w','--waterfall', type=bool, default=True,
        help="Produce a waterfall plot")
    w_parser.add_argument(
        '-nwm', '--ntw_min', type=int, default=10200,
        help="number of samples to combine for waterfall")

    d_parser = parser.add_argument_group("Dedispersion related parameters.")
    d_parser.add_argument(
        '--dedisperse', type=str, default='incoherent',
        help="One of ['None', 'coherent', 'by-channel', 'incoherent'].\n"
        "Note: None really does nothing.")
    d_parser.add_argument(
        '--fref', type=float, default=_fref,
        help="ref. freq. for dispersion measure")

    parser.add_argument('-v', '--verbose', action='append_const', const=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = CL_parser()
    args.verbose = 0 if args.verbose is None else sum(args.verbose)

    if args.rfi_filter_raw:
        args.rfi_filter_raw = rfi_filter_raw
    else:
        args.rfi_filter_raw = None

    if args.reduction_defaults == 'lofar':
        args.telescope = 'lofar'
        # already channelized, determined from filehandle
        # (previously args.nchan = 20)
        args.nchan = None
        args.ngate = 512
        args.date = '2013-07-25'
        args.ntbin = 5
        args.ntw_min = 10200
        args.waterfall = True
        args.verbose += 1
        args.dedisperse = 'incoherent'
        args.rfi_filter_raw = None

    elif args.reduction_defaults == 'aro':
        # do nothing, args are already set to aro.py defaults
        args.verbose += 1
        args.rfi_filter_raw = rfi_filter_raw

    elif args.reduction_defaults == 'gmrt':
        args.telescope = 'gmrt'
        args.date = '2013-07-25'  # Note, gmrt dates made up for now
        args.nchan = 512
        args.ngate = 512
        args.ntbin = 5
        # 170 /(100.*u.MHz/6.) * 512 = 0.0052224 s = 256 bins/pulse
        args.ntw_min = 170
        args.rfi_filter_raw = None
        args.waterfall = True
        args.verbose += 1
        args.dedisperse = 'incoherent'
    reduce(
        args.telescope, args.date, tstart=args.starttime, tend=args.endtime,
        nchan=args.nchan, ngate=args.ngate, ntbin=args.ntbin,
        ntw_min=args.ntw_min, rfi_filter_raw=args.rfi_filter_raw,
        do_waterfall=args.waterfall, do_foldspec=args.foldspec,
        dedisperse=args.dedisperse, verbose=args.verbose)
