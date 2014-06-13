import numpy as np
import astropy.units as u
from scipy.fftpack import ifft
from astropy.modeling import models


class ThermalEmission(object):
    """Simulate thermal emission, for a given spectral shape

    fnu = fnu0 * (nu / nu0)**spectral_index

    Parameters
    ----------
    spectral_index : float

    nu0 : Quantity
        reference frequency
    fnu0 : Quantity
        power at the reference frequency
    """
    def __init__(self, spectral_index=0, nu0=1.*u.GHz, fnu0=1.*u.Jy):
        self.spectral_index = spectral_index
        self.fnu0 = fnu0
        self.nu0 = nu0

    def __call__(self, duration, samplerate):
        """Produce simulated voltages for given duration and samplerate

        Parameters
        ----------
        duration : Quantity
            Should be time units
        samplerate : Quantity
            Rate at which samples should be generated

        The samples are complex, so the real and imaginary parts can be
        used as separate time streams, or can be thought of as complex
        voltages.  The total number of samples is duration * samplerate.
        """
        times = (np.arange(0., (duration * samplerate).to(1).value,
                           dtype=np.float32) / samplerate).to(duration.unit)
        nbins = times.shape[0]
        spectral_power = (np.arange(0., (duration * samplerate).to(1).value,
                                    dtype=np.float32) /
                          (duration * self.nu0)).to(1) ** self.spectral_index
        spectral_power *= self.fnu0.astype(np.float32)
        spectral_phase = np.random.uniform(size=nbins) * u.cycle
        with u.add_enabled_equivalencies(u.dimensionless_angles()):
            spectrum = np.sqrt(spectral_power) * np.exp(1j * spectral_phase)
        spectrum[0] = 0.
        return times, ifft(spectrum, overwrite_x=True)


class PulseProfile(models.Gaussian1D):
    """Pulse profile modelled as a set of Gaussians

    Parameters are like astropy.modeling.models.Gaussian1D:
    amplitude, mean, stddev, with mean and stddev in units of pulse phase

    Upon evaluation, the various contributions are summed.
    """
    def __call__(self, phase):
        """Evaluate the pulse profile at the given phase(s)"""
        result = super(PulseProfile, self).__call__(phase)
        if result.ndim > 0:
            result = result.sum(-1)
        return result


if __name__ == '__main__':
    pp = PulseProfile(amplitude=[1., 1.],
                      mean=[0.05, 0.1],
                      stddev=[0.01, 0.03])
    te = ThermalEmission(-0.5)
    times, em = te(0.2*u.s, 20.*u.MHz)
    em *= pp(times)
    # just delay; 1000/(20*u.MHz) = 0.05 ms
    obs = (em[:-1000]+em[1000:]).real[:3276800]
    obsfft = ifft(obs.reshape(-1,16384), axis=-1)[:, 8192:8192+4096]
    obspow = (np.abs(obsfft)**2).reshape(-1, 10, 4096).sum(1)
    obsnorm = obspow / obspow.sum(1, keepdims=True)
    # need shift & delay routine
