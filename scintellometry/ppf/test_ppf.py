import numpy as np
import matplotlib.pylab as plt

import ppf

if __name__ == '__main__':
    fppf = ppf.FastPPF()
    ippf = ppf.FastiPPF()
    size, shape = fppf.weights.size, fppf.weights.shape
    t = np.arange(-size, size)
    timedata = np.sin(t+0.35*np.pi) + np.sin(t*0.33)
    unfiltered = np.fft.rfft(timedata[size/2:-size/2].reshape(shape) *
                             fppf.weights.sum(0))
    filtered = np.array([np.fft.rfft(fppf(timedata[s-size/2:s+size/2]
                                          .reshape(shape)))
                                     for s in xrange(size/2,
                                                     len(timedata)-size/2,
                                                     shape[-1])])
    plt.plot(np.abs(unfiltered.T))
    plt.plot(np.abs(filtered.T))

    inverse = np.fft.irfft(filtered)
    timeback = ippf(inverse)
    plt.clf()
    plt.plot(timeback)
    # plt.plot(inverse.flatten()[size/2-1024:size/2])
    plt.plot(timedata[size-1024:size])
    # good to ~0.06/2 = 3%
