""" work in progress: need to do lofar-style waterfall and foldspec """
from __future__ import division, print_function

import argparse
import numpy as np
import astropy.units as u
from astropy.time import Time

from scintellometry.folding.fold import Folder, normalize_counts
from scintellometry.folding.pmap import pmap
from scintellometry.folding.filehandlers import AROdata, LOFARdata, GMRTdata

from observations import obsdata

from mpi4py import MPI

MAX_RMS = 4.2
_fref = 150. * u.MHz  # ref. freq. for dispersion measure


def rfi_filter_raw(raw):
    # note this should accomodate all data (including's lofar raw = complex)
    rawbins = raw.reshape(-1, raw.size)  # note, this is view!
    rawbins *= (rawbins.std(-1, keepdims=True) < MAX_RMS)
    return raw


def rfi_filter_power(power):
    return np.clip(power, 0., MAX_RMS**2 * power.shape[-1])
       

def reduce(telescope, obsdate, tstart, tend, nchan, ngate, ntbin, ntw_min=10200, fref=_fref,
           rfi_filter_raw=None,
           do_waterfall=True, do_foldspec=True, dedisperse=None,verbose=True):
    comm = MPI.COMM_WORLD
    Obs = obsdata()
    # find nearest observation to 'date', warning if > 1s off requested start time of observation
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
    elif telescope == 'gmrt':
        GenericOpen = GMRTdata

    # need to fix for lofar and gmrt
    node = Obs[telescope]['node']

    foldspecs = []
    icounts = []
    waterfalls = []
    
    for idx, fname in enumerate(files):
      with GenericOpen(*fname, comm=comm) as fh:
        time0 = fh.time0
        phasepol = Obs[telescope][obskey].get_phasepol(time0)
        nt = fh.ntimebins(tstart, tend)
        ntint = fh.ntint(nchan)
        ntw = min(ntw_min, nt*ntint)  # number of samples to combine for waterfall

        samplerate = fh.samplerate

        # number of records to skip
        nskip = fh.nskip(tstart)    # number of records to skip
        if verbose and comm.rank == 0:
            print("Using start time {0} and phase polynomial {1}"
                  .format(time0, phasepol))
            print("Skipping {0} records and folding {1} records to cover "
                  "time span {2} to {3}"
                  .format(nskip, nt,
                          time0 + nskip * fh.recsize * 2 / samplerate,
                          time0 + (nskip+nt) * fh.recsize * 2 / samplerate))

        # set the default parameters to fold
        # Note, some parameters may be in fh's HDUs, or fh.__getitem__
        # but these are overwritten if explicitly sprecified in Folder
        folder = Folder(
                        fh, nchan=nchan,
                        nt=nt, ntint=ntint, nskip=nskip, ngate=ngate,
                        ntbin=ntbin, ntw=ntw, dm=dm, fref=fref,
                        phasepol=phasepol,
                        dedisperse=dedisperse, do_waterfall=do_waterfall,
                        do_foldspec=do_foldspec, verbose=verbose, progress_interval=1,
                        rfi_filter_raw=None, #AARrfi_filter_raw,
                        rfi_filter_power=None)
        myfoldspec, myicount, mywaterfall = folder(fh, comm=comm)
        
        if do_waterfall:
            waterfall = np.zeros_like(mywaterfall)
            comm.Reduce(mywaterfall, waterfall, op=MPI.SUM, root=0)
            if comm.rank == 0:
                waterfalls.append(waterfall)

        if do_foldspec:
            foldspec = np.zeros_like(myfoldspec)
            icount = np.zeros_like(myicount)
            comm.Reduce(myfoldspec, foldspec, op=MPI.SUM, root=0)
            comm.Reduce(myicount, icount, op=MPI.SUM, root=0)
            if comm.rank == 0:
                foldspecs.append(foldspec)
                icounts.append(icount)
                this_f = normalize_counts(foldspec, icount)
                fname = ("rd_{0}{1}foldspec_idx{2}_node{3}.npy")
                iname = ("rd_{0}{1}icount_idx{2}_node{3}.npy")
                np.save(fname.format(telescope, psr, idx, node), foldspec)
                np.save(iname.format(telescope, psr, idx, node), icount)
    # end file loop (mostly for lofar subbands)


    if do_waterfall and  comm.rank == 0:
            waterfall = normalize_counts(np.concatenate(waterfalls, axis=0))
            np.save("rd_{0}{1}waterfall_{2}.npy"
                    .format(telescope, psr, node), waterfall)

    if do_foldspec and comm.rank == 0:
        foldspec = np.concatenate(foldspecs, axis=0)
        icount = np.concatenate(icounts, axis=0)
        np.save("rd_{0}{1}foldspec_{2}".format(telescope,psr, node), foldspec)
        np.save("rd_{0}{1}icount_{2}".format(telescope, psr, node), icount)

        # get normalized flux in each bin (where any were added)
        f2 = normalize_counts(foldspec, icount)
        foldspec1 = f2.sum(axis=2)
        fluxes = foldspec1.sum(axis=0)
        foldspec3 = f2.sum(axis=0)

        with open('rd_{0}{1}flux_{2}.dat'.format(telescope, psr, node), 'w') as f:
            for i, flux in enumerate(fluxes):
                f.write('{0:12d} {1:12.9g}\n'.format(i+1, flux))

    plots = True
    if plots and comm.rank == 0:
        if do_waterfall:
            w = waterfall.copy()
            pmap('rd_{0}{1}waterfall_{2}.pgm'.format(telescope, psr, node),
                 w, 1, verbose=True)
        if do_foldspec:
            pmap('rd_{0}{1}folded_{2}.pgm'.format(telescope, psr, node),
                 foldspec1, 0, verbose)
            pmap('rd_{0}{1}foldedbin_{2}.pgm'.format(telescope, psr, node),
                 f2.transpose(0,2,1).reshape(nchan,-1), 1, verbose)
            pmap('rd_{0}{1}folded3_{2}.pgm'.format(telescope, psr, node),
                 foldspec3, 0, verbose)


def CL_parser():
    parser = argparse.ArgumentParser(prog='reduce_data.py',
                 description='Process data, with options to dedisperse and fold.',
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--reduction_defaults', type=str,
                        help="One of ['aro', 'lofar', 'gmrt'].\n"
                             "A convenience flag to set the default parameters as "
                             "configured in aro.py, lofar.py, gmrt.py")

    d_parser = parser.add_argument_group("Data-related parameters \nThey specify "
                   "which observation run to process (consistent with [telescope], [[date]] entries "
                   "\nin observations.conf), at the start and finish timestamps.")
    d_parser.add_argument('-t','--telescope', type=str, default='aro',
                        help="The data to reduce. One of ['aro', 'lofar', 'gmrt']." ) 
    d_parser.add_argument('-d','--date', type=str, default='2013-07-25T18:14:20',
                        help="The date of the data to reduce"
                        "Recall observations.conf stores the observations"
                        "runs keyed by telescope and date.")
    d_parser.add_argument('-t0', '--starttime', type=str, default='2013-07-25T22:25:04.767',
                          help="Timestamp within the observation run to start processing.")
    d_parser.add_argument('-t1', '--endtime', type=str, default='2013-07-25T22:25:19.767',
                          help="Timestamp within the observation run to end processing. "
                          "(replaces the 'nt' argument)")
    

    f_parser = parser.add_argument_group("folding related parameters")
    f_parser.add_argument('-f','--foldspec', type=bool, default=True,
                        help="Produce a folded spectrum")
    f_parser.add_argument('-nc', '--nchan', type=int, default=512,
                          help="Number of channels in folded spectrum.")
    f_parser.add_argument('-ng', '--ngate', type=int, default=512,
                          help="number of bins over the pulsar period.")
    #    f_parser.add_argument('-nt', '--nt', type=int, default=1800,
    #                          help="number of time bins to fold the data into. ")
    f_parser.add_argument('-nb', '--ntbin', type=int, default=12,
                          help="number of time bins the time series is split into for folding. ")

    w_parser = parser.add_argument_group("Waterfall related parameters")
    w_parser.add_argument('-w','--waterfall', type=bool, default=True,
                        help="Produce a waterfall plot")
    w_parser.add_argument('-nwm', '--ntw_min', type=int, default=10200,
                          help="number of samples to combine for waterfall")


    d_parser = parser.add_argument_group("Dedispersion related parameters.")
    d_parser.add_argument('--dedisperse', type=str, default=None,
                        help="One of ['None', 'coherent', 'by-channel'].")
    d_parser.add_argument('--fref', type=float, default=_fref,
                          help="ref. freq. for dispersion measure")


    parser.add_argument('-v', '--verbose', action='append_const', const=1)
    return parser.parse_args()

   
if __name__ == '__main__':
    args = CL_parser()
    args.verbose = 0 if args.verbose is None else sum(args.verbose)
    args.rfi_filter_raw = rfi_filter_raw
    if args.reduction_defaults == 'lofar':
        args.telescope = 'lofar'
        args.nchan = 20
        args.ngate =  512
        args.date = '2013-05-05' # Note, dates are made up for now
        #args.nt = 180
        args.ntbin = 6
        args.ntw_min = 10200
        args.waterfall = False
        args.verbose += 1
        args.dedisperse = None
        args.rfi_filter_raw = None

    elif args.reduction_defaults == 'aro':
        # do nothing, args are already set to aro.py defaults
        pass

    elif args.reduction_defaults == 'gmrt':
        # to-do...
        args.telescope = 'gmrt'
        args.date = '2013-07-26' # Note, gmrt dates made up for now
        args.nchan = 512
        args.ngate = 512
        #args.nt = 500 # 5 min/16.667 MHz=2400*(recsize / 2)
        args.ntbin = 5 
        args.ntw_min = 170 # 170 /(100.*u.MHz/6.) * 512 = 0.0052224 s = 256 bins/pulse
        args.rfi_filter_raw = None
        args.waterfall = True
        args.dedisperse = 'incoherent'

    reduce(
        args.telescope, args.date, tstart=args.starttime, tend=args.endtime,
        nchan=args.nchan, ngate=args.ngate, ntbin=args.ntbin, ntw_min=args.ntw_min,
        rfi_filter_raw=args.rfi_filter_raw,
        do_waterfall=args.waterfall, do_foldspec=args.foldspec,
        dedisperse=args.dedisperse, verbose=args.verbose)
