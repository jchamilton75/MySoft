from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import pi
import pycamb
from qubic import QubicInstrument
from scipy.constants import c
from Sensitivity import qubic_sensitivity
from Homogeneity import SplineFitting
from pysimulators import FitsArray
from Cosmo import interpol_camb as ic
from Sensitivity import dualband_lib as db

### QUBIC instrument
inst = QubicInstrument()

### Binning (as BICEP2)
ellbins = np.array([21, 56, 91, 126, 161, 196, 231, 266, 301, 335])
ellmin = ellbins[:len(ellbins)-1]
ellmax = ellbins[1:len(ellbins)]

### DUST Parameters 
dldust_80_353 = 13.4
alphadust = -2.42
betadust = 1.59
Tdust = 19.6
defaultpars = np.array([dldust_80_353, alphadust, betadust, Tdust])

### r default value
thervalue = 0.

### Camblib
ellcamblib = FitsArray('camblib_ell.fits')
rcamblib = FitsArray('camblib_r.fits')
clcamblib = FitsArray('camblib_cl.fits')
camblib = [ellcamblib, rcamblib, clcamblib]

### Instruments
netplanck_353 = 850
net150 = 550.
net220 = 1450.

### QUBIC 150 GHz two focal planes
freqsA = [150]
typeA = ['bi']
NETsA = [net150/np.sqrt(2)]
nameA = ['150x2']
colA = 'k'
fskyA = 0.01
instrumentA = [inst, ellbins, freqsA, typeA, NETsA, fskyA, nameA, colA]
bla = db.get_multiband_covariance(instrumentA, thervalue, doplot=True, dustParams=defaultpars, verbose=True, camblib=camblib)
specA = bla[3]
all_neq_nhA = bla[6]
dataA = {"specin":specA, "inst_info":instrumentA, "all_neq_nh":all_neq_nhA, "camblib":camblib}
