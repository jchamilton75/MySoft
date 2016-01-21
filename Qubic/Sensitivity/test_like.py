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
from numpy import *

### QUBIC instrument
inst = QubicInstrument()

### Binning (as BICEP2)
ellbins = np.array([21, 56, 91, 126, 161, 196, 231, 266, 301, 336, 371, 406])#, 441, 476, 511, 546, 581])
ellmin = ellbins[:len(ellbins)-1]
ellmax = ellbins[1:len(ellbins)]

### DUST Parameters 
dldust_80_353 = 13.4 * 0.45 * 0##(to match Planck XXX measurement on Bicep2)
alphadust = -2.42
betadust = 1.59
Tdust = 19.6
defaultpars = np.array([dldust_80_353, alphadust, betadust, Tdust])

### r default value
thervalue = 0.

### Camblib
ellcamblib = FitsArray('/Users/hamilton/CMB/Interfero/DualBand/camblib600_ell.fits')
rcamblib = FitsArray('/Users/hamilton/CMB/Interfero/DualBand/camblib600_r.fits')
clcamblib = FitsArray('/Users/hamilton/CMB/Interfero/DualBand/camblib600_cl.fits')
camblib = [ellcamblib, rcamblib, clcamblib]

netplanck_353 = 850.
### Concordia Average between summer and winter
net150_concordia = 0.5*(291.+369.)*np.sqrt(2)
net220_concordia = 0.5*(547.+840.)*np.sqrt(2)

def prepare_inst(thervalue, inst, ellbins, freqs, type, NETs, name, col, fsky, duration, epsilon, camblib=None, dustParams=None):
	instrument = [inst, ellbins, freqs, type, NETs, fsky, duration, epsilon, name, col]
	bla = db.get_multiband_covariance(instrument, thervalue, doplot=True, dustParams=dustParams, verbose=True, camblib=camblib)
	spec = bla[3]
	all_neq_nh = bla[6]
	data = {"specin":spec, "inst_info":instrument, "all_neq_nh":all_neq_nh, "camblib":camblib}
	return data


qubic_duration = 2*365.*24.*3600.
qubic_epsilon = 1.
### these numbers give roughly the correct error bars for Planck 353GHz
planck_duration = 1*365.*24.*3600.
planck_epsilon = 0.3


### QUBIC 150 and 220 GHz
data = prepare_inst(thervalue, inst, ellbins,
	[150, 220],
	['bi', 'bi'],
	[net150_concordia, net220_concordia],
	['150, 220'],
	'm',
	0.01,
	[qubic_duration, qubic_duration],
	[qubic_epsilon, qubic_epsilon],
	camblib=camblib, dustParams=defaultpars)


index = 0
valsr = np.linspace(0.,0.07,30)
pars_with_r = np.array([thervalue, 13.4 * 0.45, -2.42, 1.59, 19.6])
clf()
thelike = db.like_1d(data['specin'], index, valsr, data['inst_info'], camblib=camblib, paramsdefault=pars_with_r, CL=0.95)

indexmarg = 3
valsmarg = np.linspace(1.5, 1.8, 30)
thelike = db.like_1d_marginalize(data['specin'], index, valsr, indexmarg, valsmarg, data['inst_info'], camblib=camblib, paramsdefault=pars_with_r, CL=0.95)


