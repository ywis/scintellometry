from __future__ import division, print_function

import numpy as np

from novas.compat import sidereal_time

from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import ICRSCoordinates
from astropy.table import Table
from astropy.constants import c as SPEED_OF_LIGHT

#SOURCE = ICRSCoordinates('19h21m44.815s +21d53m02.25s')
#SOURCE=ICRSCoordinates('19h59m36.76988s +20d48m15.1222s')
SOURCE=ICRSCoordinates('20h18m03.92s +28d39m55.2s')

#OUTFILE = 'July2_1919_uwvcoords.dat'
#OUTFILE = 'July2_1957_uvwcoords.dat'
OUTFILE = 'July2_2016_uvwcoords.dat'

# first time stamp of all.  Maybe should be rounded to minute?
#TIME_STAMP0 = '2013 07 02 04 30 00 0.694016'
#TIME_STAMP0 = '2013 07 02 05 11 00 0.133888'
TIME_STAMP0 = '2013 07 02 07 08 00 0.129280'

#MAX_NO_IN_SEQ_FILE = 3234
#MAX_NO_IN_SEQ_FILE = 20546
MAX_NO_IN_SEQ_FILE = 1814
N_BLOCK = MAX_NO_IN_SEQ_FILE - 1
DT_SAMPLE = TimeDelta(0., (3/(200*u.MHz)).to(u.s).value, format='sec')
DT_BLOCK = 2.**24*DT_SAMPLE

TEL_LONGITUDE = 74*u.deg+02*u.arcmin+59.07*u.arcsec
TEL_LATITUDE = 19*u.deg+05*u.arcmin+47.46*u.arcsec

NPOD = 30  # Number of baselines (only used as sanity check)

ANTENNA_FILE = '/Users/Natalie/scintellometry/scintellometry/phasing/' \
               'antsys.hdr'
OUR_ANTENNA_ORDER = 'CWES'   # and by number inside each group
NON_EXISTING_ANTENNAS = ('C07', 'S05')  # to remove from antenna file


USE_UT1 = False
if USE_UT1:
    IERS_A_FILE = '/home/mhvk/packages/astropy/finals2000A.all'
    from astropy.utils.iers import IERS_A
    iers_a = IERS_A.open(IERS_A_FILE)

IST_UTC = TimeDelta(0., 5.5/24., format='jd')


def timestamp_to_Time(line):
    """Convert a timestamp item to a astropy Time instance.
    Store telescope lon, lat as well for full precision in possible
    TDB conversion (not used so far)
    """
    tl = line.split()
    seconds = float(tl[5])+float(tl[6])
    return Time(tl[0] + '-' + tl[1] + '-' + tl[2] + ' ' +
                tl[3] + ':' + tl[4] + ':{}'.format(seconds), scale='utc',
                lat=TEL_LATITUDE, lon=TEL_LONGITUDE)


def UTC_to_gast(times):
    """Approximate conversion: ignoring UT1-UTC difference."""
    gast = np.zeros(len(times))
    for i,t in enumerate(times):
        gast[i] = sidereal_time(t.utc.jd1, t.utc.jd2,
                                delta_t=(t.tt.mjd-t.utc.mjd)*24*3600)
    return gast*(np.pi/12.)*u.rad


def UT1_to_gast(times):
    """Fairly precise conversion to GAST.  Includes unmodelled parts of the
    Earth rotation (in UT1), but not yet of polar wander."""
    times.delta_ut1_utc = iers_a.ut1_utc(times)
    gast = np.zeros(len(times))
    for i,t in enumerate(times):
        gast[i] = sidereal_time(t.ut1.jd1, t.ut1.jd2,
                                delta_t=(t.tt.mjd-t.ut1.mjd)*24*3600)
    return gast*(np.pi/12.)*u.rad


def get_antenna_coords(filename):
    """Read antenna coordinates from GMRT .hdr file.  First store them all
    in a dictionary, indexed by the antenna name, remove non-existing
    antennas, then get them in the order used in Ue-Li's phasing code,
    and finally make it a Table, which is easier to access than a
    dictionary.  Probably could be done more directly.
    """
    with open(filename, 'r') as fh:
        antennas = {}
        line = fh.readline()
        while line != '':
            if line[:3] == 'ANT':
                al = line.split()
                antennas[al[2]] = np.array([float(item) for item in al[3:8]])
            line = fh.readline()

    for bad in NON_EXISTING_ANTENNAS:
        antennas.pop(bad)

    antenna_names = order_antenna_names(antennas)
    # store all antenna's in a Table
    ant_tab = Table()
    ant_tab['ant'] = antenna_names
    ant_tab['xyz'] = [antennas[ant][:3] for ant in ant_tab['ant']]
    ant_tab['delay'] = [antennas[ant][3:] for ant in ant_tab['ant']]
    return ant_tab


def order_antenna_names(antennas, order=OUR_ANTENNA_ORDER):
    """Get antenna in the correct order, grouped by C, W, E, S, and
    by number within each group.
    """
    names = list(antennas)

    def cmp_names(x, y):
        value_x, value_y = [order.index(t[0])*100+int(t[1:]) for t in x, y]
        return -1 if value_x < value_y else 1 if value_x > value_y else 0

    names.sort(cmp_names)
    return names


def get_uvw(ha, dec, antennas, ref_ant):
    """Get delays in UVW directions between pairs of antenna's for
    given hour angle and declination of a source.
    """
    h = ha.to(u.rad).value
    d = dec.to(u.rad).value
    dxyz = antennas['xyz'][ref_ant] - antennas['xyz']
    #  unit vectors in the U, V, W directions
    xyz_u = np.array([-np.sin(d)*np.cos(h), -np.sin(d)*np.sin(h), np.cos(d)])
    xyz_v = np.array([-np.sin(h), np.cos(h), 0.])
    xyz_w = np.array([np.cos(d)*np.cos(h), np.cos(d)*np.sin(h), np.sin(d)])
    return np.vstack([(xyz_u*dxyz).sum(1),
                      (xyz_v*dxyz).sum(1),
                      (xyz_w*dxyz).sum(1)]).T


if __name__ == '__main__':
    # start time in UTC
    t0 = timestamp_to_Time(TIME_STAMP0) - IST_UTC
    # set of times encomassing the whole scan
    times = t0 + DT_BLOCK*np.arange(N_BLOCK)
    # precess source coordinate to mid-observation time
    tmid = times[len(times)//2]
    source = SOURCE.fk5.precess_to(tmid)
    # calculate Greenwich Apparent Sidereal Time
    if USE_UT1:
        gast = UT1_to_gast(times)
    else:
        gast = UTC_to_gast(times)
    # for possible testing
    # for t, g in zip(times, gast):
    #     print("{0:14.8f} {1:11.8f}".format(t.mjd-40000.,
    #                                        g.to(u.rad).value*np.pi/12.))

    # with Sidereal time, we can calculate the hour hangle
    # (annoyingly, which source.ra is in units of angle, cannot subtract
    #  other angles; this should get better in future versions of astropy)
    ha = source.ra.radians * u.rad - gast - TEL_LONGITUDE
    # print(times,gast.to(u.deg).value/15.,ha.to(u.deg).value/15. % 24.)

    # calculate parallactic angle for possible use in polarimetry
    chi = np.arctan2(np.cos(TEL_LATITUDE.to(u.rad).value) *
                     np.sin(ha.to(u.rad).value),
                     np.sin(TEL_LATITUDE.to(u.rad).value) *
                     np.cos(source.dec.radians) -
                     np.cos(TEL_LATITUDE.to(u.rad).value) *
                     np.sin(source.dec.radians) *
                     np.cos(ha.to(u.rad).value)) * u.rad
    # print(times,gast.to(u.deg).value/15.,ha.to(u.deg).value/15. % 24.,
    #       chi.to(u.deg))

    # antennas and their coordinates are will be ordered by OUR_ANTENNA_ORDER
    antennas = get_antenna_coords(ANTENNA_FILE)
    # sanity check
    assert NPOD == len(antennas)

    # write out delays for all time stamps, looping over baselines
    ref_index = 2   # note, this is not the GMRT default, of 'C02' => index 2
    with open(OUTFILE, 'w') as fo:
        for h, c in zip(ha, chi):
            # get UVW coordinates for this HA
            uvw = get_uvw(h, source.dec.radians * u.rad, antennas, ref_index)
            # print them by pair
            for j in range(len(uvw)):
                uvw_us = (uvw[j]*u.m/SPEED_OF_LIGHT).to(u.us).value
                uvw_m = (uvw[j]*u.m).value
                fo.write("{:02d} {:02d} {:f} {:f} {:f} {:f}\n".format(
                    ref_index, j, uvw_m[0], uvw_m[1], uvw_m[2],
                    c.to(u.rad).value))
