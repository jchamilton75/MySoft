from __future__ import division
from numpy import *
from matplotlib.pyplot import *

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import pysimulators
import pyoperators
from qubic import QubicAcquisition, QubicInstrument, create_sweeping_pointings
from astropy.time import Time, TimeDelta

### Dome C
domec = np.array([-(75 + 6 / 60), 123 + 20 / 60])
#SITELAT = -(75 + 6 / 60)
#SITELON = 123 + 20 / 60
### San Antonio de los Cobres
sadlc = np.array([-24.18947, -66.472016])
#SITELAT = -24.18947
#SITELON = -66.472016

q2g = pysimulators.SphericalEquatorial2GalacticOperator(degrees=True)
g2q = pysimulators.SphericalGalactic2EquatorialOperator(degrees=True)





