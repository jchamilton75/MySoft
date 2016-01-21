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
dldust_80_353 = 13.4 * 0.45##(to match Planck XXX measurement on Bicep2)
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

### Instruments NOISE
### For QUBIC from email by M. Piat on feb 9th 2015
# DC: 
# 	150GHz: 291 / 369 uK.sqrt(s)
# 	220GHz: 547 / 840 uK.sqrt(s)
# Atacama (avec 230K / 266K et 5% / 10%):
# 	150GHz: 369 / 516 uK.sqrt(s)
# 	220GHz: 840 / 1356 uK.sqrt(s)
### need to apply sqrt(2) in order to account for polarization. But no other sqrt(2) as the required value needs to be in sqrt(s) not sqrt(Hz). One Hz corresponds to 1/2 seconds of integration.
netplanck_353 = 850.
### Concordia Average between summer and winter
### OLD numbers net150_concordia = 550.
### OLD numbers net220_concordia = 1450.
net150_concordia = 0.5*(291.+369.)*np.sqrt(2)
net220_concordia = 0.5*(547.+840.)*np.sqrt(2)
### Atacama Average between summer and winter
net150_atacama = 0.5*(369.+516.)*np.sqrt(2)
net220_atacama = 0.5*(840.+1356.)*np.sqrt(2)

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

#### QUBIC at concordia

### QUBIC 150 GHz two focal planes
dataA = prepare_inst(thervalue, inst, ellbins,
	[150],
	['bi'],
	[net150_concordia/np.sqrt(2)],
	['150x2'],
	'k',
	0.01,
	[qubic_duration],
	[qubic_epsilon],
	camblib=camblib, dustParams=defaultpars)




### QUBIC 150 and 220 GHz
dataB = prepare_inst(thervalue, inst, ellbins,
	[150, 220],
	['bi', 'bi'],
	[net150_concordia, net220_concordia],
	['150, 220'],
	'm',
	0.01,
	[qubic_duration, qubic_duration],
	[qubic_epsilon, qubic_epsilon],
	camblib=camblib, dustParams=defaultpars)


ellav = 0.5 * (ellmin + ellmax)
deltal = ellmax - ellmin
fact = ellav * (ellav + 1) / (2 * np.pi)

def deltaclnoise(ell, fsky, deltal, fwhmdeg, mukarcmin):
    omegaarcmin2 = (fsky*4*np.pi*(180/np.pi)**2) * 60**2
    bl = exp(-0.5*ell**2 * (np.radians(fwhmdeg/2.35))**2)
    return np.sqrt(2./((2 * ell + 1)*fsky*deltal)) * mukarcmin**2 / bl**2 / omegaarcmin2

#fct = 10*exp(-ellav/100)
#plot(ellav, deltaclnoise(ellav, fs, deltal, 0.65, 2.1)*fact, 'red')
#plot(ellav, deltaclnoise(ellav, fs, deltal, 0.65, 4.5)*fact, 'blue')
#plot(ellav, deltaclnoise(ellav, fs, deltal, 0.65, np.sqrt(4.5*2.1))*fact, 'green')



### QUBIC 150GHzx2 + Planck 353 GHz
dataC = prepare_inst(thervalue, inst, ellbins,
	[150, 353],
	['bi', 'im'],
	[net150_concordia/np.sqrt(2), netplanck_353],
	['150x2, 353'],
	'b',
	0.01, 
	[qubic_duration, planck_duration],
	[qubic_epsilon, planck_epsilon],
	camblib=camblib, dustParams=defaultpars)

### QUBIC 150, 220 GHz and Planck 353 GHz
dataD = prepare_inst(thervalue, inst, ellbins,
	[150, 220, 353],
	['bi', 'bi', 'im'],
	[net150_concordia, net220_concordia, netplanck_353],
	['150, 220, 353'],
	'r',
	0.01,
	[qubic_duration, qubic_duration, planck_duration],
	[qubic_epsilon, qubic_epsilon, planck_epsilon],
	camblib=camblib, dustParams=defaultpars)

### QUBIC 150, 220 GHz no foregrounds
noforegrounds = np.array([0, alphadust, betadust, Tdust])
dataNofg = prepare_inst(thervalue, inst, ellbins,
	[150, 220],
	['bi', 'bi'],
	[net150_concordia, net220_concordia],
	['150, 220 No Foregrounds'],
	'r',
	0.01,
	[qubic_duration, qubic_duration, planck_duration],
	[qubic_epsilon, qubic_epsilon, planck_epsilon],
	camblib=camblib, dustParams=noforegrounds)



# ### QUBIC 220x2
# dataE = prepare_inst(thervalue, inst, ellbins,
# 	[220],
# 	['bi'],
# 	[net220_concordia/sqrt(2)],
# 	['220x2'],
# 	'g',
# 	0.01,
# 	[qubic_duration],
# 	[qubic_epsilon],
# 	camblib=camblib)

# ### QUBIC 220x2+353
# dataF = prepare_inst(thervalue, inst, ellbins,
# 	[220, 353],
# 	['bi','im'],
# 	[net220_concordia/sqrt(2), netplanck_353],
# 	['220x2, 353'],
# 	'c',
# 	0.01,
# 	[qubic_duration, planck_duration],
# 	[qubic_epsilon, planck_epsilon],
# 	camblib=camblib)



# ####### Same but at Atacama
# ### QUBIC 150 GHz two focal planes
# dataAa = prepare_inst(thervalue, inst, ellbins,
# 	[150],
# 	['bi'],
# 	[net150_atacama/np.sqrt(2)],
# 	['150x2'],
# 	'k',
# 	0.01,
# 	[qubic_duration],
# 	[qubic_epsilon],
# 	camblib=camblib)

# ### QUBIC 150 and 220 GHz
# dataBa = prepare_inst(thervalue, inst, ellbins,
# 	[150, 220],
# 	['bi', 'bi'],
# 	[net150_atacama, net220_atacama],
# 	['150, 220'],
# 	'm',
# 	0.01,
# 	[qubic_duration, qubic_duration],
# 	[qubic_epsilon, qubic_epsilon],
# 	camblib=camblib)

# ### QUBIC 150GHzx2 + Planck 353 GHz
# dataCa = prepare_inst(thervalue, inst, ellbins,
# 	[150, 353],
# 	['bi', 'im'],
# 	[net150_atacama/np.sqrt(2), netplanck_353],
# 	['150x2, 353'],
# 	'b',
# 	0.01, 
# 	[qubic_duration, planck_duration],
# 	[qubic_epsilon, planck_epsilon],
# 	camblib=camblib)

# ### QUBIC 150, 220 GHz and Planck 353 GHz
# dataDa = prepare_inst(thervalue, inst, ellbins,
# 	[150, 220, 353],
# 	['bi', 'bi', 'im'],
# 	[net150_atacama, net220_atacama, netplanck_353],
# 	['150, 220, 353'],
# 	'r',
# 	0.01,
# 	[qubic_duration, qubic_duration, planck_duration],
# 	[qubic_epsilon, qubic_epsilon, planck_epsilon],
# 	camblib=camblib)

# ### QUBIC 220x2
# dataEa = prepare_inst(thervalue, inst, ellbins,
# 	[220],
# 	['bi'],
# 	[net220_atacama/sqrt(2)],
# 	['220x2'],
# 	'g',
# 	0.01,
# 	[qubic_duration],
# 	[qubic_epsilon],
# 	camblib=camblib)

# ### QUBIC 220x2+353
# dataFa = prepare_inst(thervalue, inst, ellbins,
# 	[220, 353],
# 	['bi','im'],
# 	[net220_atacama/sqrt(2), netplanck_353],
# 	['220x2, 353'],
# 	'c',
# 	0.01,
# 	[qubic_duration, planck_duration],
# 	[qubic_epsilon, planck_epsilon],
# 	camblib=camblib)






########## Use PyMC instead ##########################################################
import pymc
from Sensitivity import data4mcmc
reload(data4mcmc)

default = np.array([thervalue, dldust_80_353, alphadust, betadust, Tdust])

def runit(data, variables, filename, niter=120000, nburn=20000, nthin=1):
	chain = pymc.MCMC(data4mcmc.generic_model(data, variables=variables, default=default),db='pickle',dbname=filename)
	chain.use_step_method(pymc.AdaptiveMetropolis,chain.stochastics,delay=1000)
	chain.sample(iter=niter,burn=nburn,thin=nthin)
	chain.db.close()
	return chain


import sys
datanum = int(sys.argv[1])

datas = [dataA, dataB, dataC, dataD, dataNofg]

filenames = ['instrumentA_r_dl_b.db', 
			'instrumentB_r_dl_b.db', 
			'instrumentC_r_dl_b.db', 
			'instrumentD_r_dl_b.db',
			'instrumentNofg_r.db']

variables = [['r','dldust_80_353','betadust'],
			['r','dldust_80_353','betadust'],
			['r','dldust_80_353','betadust'],
			['r','dldust_80_353','betadust'], 
			['r']]


chain = runit(datas[datanum], variables[datanum], filenames[datanum])





