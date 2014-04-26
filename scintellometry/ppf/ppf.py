
## Imports
#  Only import modules that are actually needed and avoid
#  "from module import *" as much as possible to prevent name clashes.
import numpy as np
from astropy.utils.data import get_pkg_data_filename


class FastPPF():
    """Poly-phase filter for (timeseries) data of shape N x 16 x 1024"""

    def __init__(self, coefficients_file='Coeffs16384Kaiser-quant.dat'):
        """Constructor"""
        #print "Obtaining Kaiser coefficient from file"
        self.weights = np.loadtxt(get_pkg_data_filename(coefficients_file),
                                  dtype=np.float64).reshape(16,1024)

    def __call__(self, data):
        """Apply the polyphase filter to the input"""
        return np.sum(self.weights*data, axis=len(data.shape)-2)


class FastiPPF():
    """Inverted poly-phase filter for (timeseries) data of shape
    N x 16 x 1024"""

    def __init__(self,coefficients_file='ppf_inv.dat'):
        """Constructor"""
        #print "Obtaining inverse PPF coefficient from file"
        # stored as sample, 16 bins, but want 16 bins, samples
        self.weights = (np.loadtxt(get_pkg_data_filename(coefficients_file))
                        .reshape(-1, 16).T)

    def __call__(self, data):
        '''Get the inverse PPF of myinput'''
        return np.sum(self.weights*data,axis=len(data.shape)-2)
