from __future__ import division

import numpy as np
import matplotlib.pylab as plt
import astropy.units as u

import ppf

if __name__ == '__main__':
    fppf = ppf.FastPPF()
    ippf = ppf.FastiPPF()
    size, shape = fppf.weights.size, fppf.weights.shape
    samplerate = 200*u.MHz
    t = (np.arange(-size + 0., size) / samplerate).to(u.s)
    # centre
    omega = np.array([700.,
                      800.49,
                      900.5,
                      600. + 7./16.,
                      650. - 6./16.]) / 1024. * u.cycle * samplerate
    timedata = np.sin(omega[:, np.newaxis] * t).sum(0)
    unfiltered = np.fft.rfft(timedata[size//2:-size//2].reshape(shape) *
                             fppf.weights.sum(0))
    filtered = np.array([np.fft.rfft(fppf(timedata[s-size//2:s+size//2]
                                          .reshape(shape)))
                         for s in xrange(size//2,
                                         len(timedata)-size//2,
                                         shape[-1])])
    freq = samplerate - np.fft.rfftfreq(shape[-1], d=1./samplerate)
    plt.plot(np.arange(1024., 511, -1), np.abs(unfiltered.T))
    plt.plot(np.arange(1024., 511, -1), np.abs(filtered.T))

    inverse = np.fft.irfft(filtered)
    timeback = ippf(inverse)
    plt.clf()
    plt.plot(timeback)
    # plt.plot(inverse.flatten()[size/2-1024:size/2])
    plt.plot(timedata[size-1024:size].value)
    # good to ~0.06/2 = 3%
