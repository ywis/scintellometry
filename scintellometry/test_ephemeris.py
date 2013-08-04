import numpy as np
from astropy.time import Time, TimeDelta
from astropy.constants import c
import astropy.units as u

from novas.compat import sidereal_time

import de421  #  de405

from pulsar.barycentre import JPLEphemeris
from pulsar.pulsar import ELL1Ephemeris
import observability

"""Try to reproduce output of
tempo2 -tempo1 -f ~/projects/scintellometry/timing/ForMarten.par \
    -polyco "56470 56477 300 6 8 gmrt 327.0"
1959+2048  26-Jun-13  153000.00   56469.64583333330            29.124911 -0.418 -2.199
 280025704586.495026  622.122028785585 gmrt  300    6   327.000 0.2463   2.6180
  2.84746502271613234e-03  2.19133766142904252e+00  1.25943848337391678e-04
 -1.38036451565109372e-05 -1.47479174299025577e-09  7.88819178781333662e-11

Hence, F(26-jun-2013T15:30:00) = F+coef[1]/60. = 622.1585510799424
coef[1]/60./F = 5.8705997645581104e-05

checks:
eph1957['F'] = 622.1220287855853  -> same as F above
(v_earth+v_topo)*1e4 = -0.41751344373113197 -> consistent w/ -0.418 above
eph1957.evaluate('F', t) = 622.12202585922876 (very similar, low spin-down)
/(1+rv)
"""

if __name__ == '__main__':
    eph1957 = ELL1Ephemeris('timing/ForMarten.par')
    jpleph = JPLEphemeris(de421)
    t = Time('2013-06-26 15:30', scale='utc',
             lon=(74*u.deg+03*u.arcmin+23.62*u.arcsec).to(u.deg).value,
             lat=(19*u.deg+05*u.arcmin+34.81*u.arcsec).to(u.deg).value)
    # ) + TimeDelta(np.linspace(0.,0.1, 101), format='jd')

    # orbital delay and velocity (lt-s and v/c)
    d_orb = eph1957.orbital_delay(t)
    v_orb = eph1957.radial_velocity(t)

    # direction to target
    dir_1957 = eph1957.pos(t)

    # Delay from and velocity of centre of earth to SSB (lt-s and v/c)
    # tdb_jd = []
    # for utc1, utc2 in zip(mjd.utc.jd1-2400000.5, mjd.utc.jd2):
    #     rf_tdb = rf_ephem.utc_to_tdb(int(utc1), utc1-int(utc1)+utc2)
    #     tdb_jd += [rf_tdb['tdb']+rf_tdb['tdb_mjd']]
    #tdb_jd = np.asarray(tdb_jd, dtype=np.longfloat)+2400000.5
    posvel_earth = jpleph.compute('earth', t)
    pos_earth = posvel_earth[:3]/c.to(u.km/u.s).value
    vel_earth = posvel_earth[3:]/c.to(u.km/u.day).value

    d_earth = np.sum(pos_earth*dir_1957, axis=0)
    # positive velocity towards pulsar means blueshift -> change sign
    v_earth = -np.sum(vel_earth*dir_1957, axis=0)

    #GMRT from tempo2-2013.3.1/T2runtime/observatory/observatories.dat
    xyz_gmrt = (1656318.94, 5797865.99, 2073213.72)
    #xyz_gmrt = (0.,0.,0.)
    # Rough delay from observatory to center of earth
    # greenwich apparent sidereal time from novas (in hours)
    gast = sidereal_time(t.utc.jd1, t.utc.jd2,
                         delta_t=(t.tt.mjd-t.utc.mjd)*86400.)
    last = (gast/24. + t.lon/360.)*2.*np.pi
    coslast, sinlast = np.cos(last), np.sin(last)
    # rotate observatory vector
    xy = np.sqrt(xyz_gmrt[0]**2+xyz_gmrt[1]**2)
    pos_gmrt = np.array([xy*coslast, xy*sinlast,
                         xyz_gmrt[2]*np.ones_like(last)])/c.si.value
    vel_gmrt = np.array([-xy*sinlast, xy*coslast,
                         np.zeros_like(last)]
                        )*2.*np.pi*366.25/365.25/c.to(u.m/u.day).value
    # take inner product with direction to pulsar
    d_topo = np.sum(pos_gmrt*dir_1957, axis=0)
    v_topo = -np.sum(vel_gmrt*dir_1957, axis=0)
    delay = d_topo + d_earth + d_orb
    rv = v_topo + v_earth + v_orb

    # if True:
    #     # try SOFA routines (but without UTC -> UT1)
    #     import sidereal
    #     # SHOULD TRANSFER TO UT1!!
    #     gmst = sidereal.gmst82(mjd.utc.jd1, mjd,utc.jd2)

    if False:
        # check with Fisher's ephemeris
        import rf_ephem
        rf_ephem.set_ephemeris_dir('/data/mhvk/packages/jplephem', 'DEc421')
        rf_ephem.set_observer_coordinates(*xyz_gmrt)
        rf_delay = []
        rf_rv = []
        for m in t.utc.mjd:
            rf_delay += rf_ephem.pulse_delay(
                eph1957.evaluate('RAJ',m)/15., eph1957.evaluate('DECJ',m),
                int(m), m-int(m), 1, 0.)['delay']
            rf_rv += rf_ephem.doppler_fraction(
                eph1957.evaluate('RAJ',m)/15., eph1957.evaluate('DECJ',m),
                int(m), m-int(m), 1, 0.)['frac']

        import matplotlib.pylab as plt
        plt.ion()
        #plt.plot(t.utc.mjd, delay-rf_delay-d_orb)
        plt.plot(t.utc.mjd, d_earth-rf_delay)
        #plt.plot(t.utc.mjd, (rv-rf_rv-v_orb)*c.to(u.km/u.s).value)
        plt.draw()

    if False:
        for utc, tdb1, tdb2 in zip(t.utc.mjd, t.tdb.jd1, t.tdb.jd2):
            rf_tdb = rf_ephem.utc_to_tdb(int(utc), utc-int(utc))['tdb']
            print utc, tdb, rf_tdb, tdb1-0.5-int(tdb1-0.5)+tdb2-rf_tdb
    if False:
        tdb = np.linspace(0.,1.,5)
        index = 10*np.ones(len(tdb), dtype=np.int)
        import numpy.polynomial.chebyshev as chebyshev
        coefficient_sets = jpleph.load('earthmoon')
        number_of_sets, axis_count, coefficient_count = coefficient_sets.shape
        coefficients = np.rollaxis(coefficient_sets[index], 1)

        T = np.empty((coefficient_count, len(tdb)))
        T[0] = 1.0
        T[1] = t1 = 2.0 * tdb - 1.0
        twot1 = t1 + t1
        for i in range(2, coefficient_count):
            T[i] = twot1 * T[i-1] - T[i-2]
        result = (T.T * coefficients).sum(axis=2)
        V = chebyshev.chebvander(t1, coefficient_count-1)
