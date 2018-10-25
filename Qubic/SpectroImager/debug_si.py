#!/bin/env python
from __future__ import division
import sys
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
from qubic import (create_random_pointings, gal2equ,
                  read_spectra,
                  compute_freq,
                  QubicScene,
                  QubicMultibandInstrument,
                  QubicMultibandAcquisition,
                  PlanckAcquisition,
                  QubicMultibandPlanckAcquisition)
import qubic
from SpectroImager import SpectroImLib as si
from pysimulators import FitsArray







# center = qubic.gal2equ(0,0)
# effective_duration=2.
# #x0 = fits.getdata( dbdir+'/input_cmb_ns128.fits')
# #x0 = x0*0.
# x0.T[0] = 1e2/1.36e-17
# x0.T[1:] = 0.

# p = qubic.create_random_pointings(center, 1000, 10.)
# s = qubic.QubicScene(nside=nside, kind=‘IQU’)

# q = qubic.QubicInstrument()
# atod = qubic.QubicAcquisition(q, p, s, effective_duration=effective_duration)
# TOD,m = atod.get_observation(x0, noiseless=True, convolution=True)





######## Default configuration
### Sky 
nside = 128
center_gal = 0, 90
center = qubic.gal2equ(center_gal[0], center_gal[1])
dust_coeff = 1.39e-2

### Detectors (for now using random pointing)
band = 150
relative_bandwidth = 0.25
sz_ptg = 10
nb_ptg = 100
effective_duration = 2.
ripples = False   
noiseless = True

### Mapmaking
tol = 1e-4

### Number of sub-bands to build the TOD
nf_sub_build = 15
nf_sub_rec = 2

parameters = {'nside':nside, 'center':center, 'dust_coeff': dust_coeff, 
				'band':band, 'relative_bandwidth':relative_bandwidth,
				'sz_ptg':sz_ptg, 'nb_ptg':nb_ptg, 'effective_duration':effective_duration, 
				'tol': tol, 'ripples':ripples,
				'nf_sub_build':nf_sub_build, 
				'nf_sub_rec': nf_sub_rec,
                        'noiseless':noiseless}

for k in parameters.keys(): print(k, parameters[k])

nf_build_vals = [2,5]
ls = ['-','--']

allTOD = []

for i in xrange(len(nf_build_vals)):
      print(i)
      parameters['nf_sub_build'] = nf_build_vals[i]
      x0 = si.create_input_sky(parameters)
      x0[:,:,0]=1
      x0[:,:,1]=0
      x0[:,:,2]=0
      p = si.create_random_pointings(parameters['center'], parameters['nb_ptg'], parameters['sz_ptg'])
      p.pitch = p.pitch*0
      TOD = si.create_TOD(parameters, p, x0)
      allTOD.append(TOD)

clf()
plot(allTOD[0][:,0], lw=3)
plot(allTOD[1][:,0], 'r--', lw=2)



clf()
for i in xrange(len(nf_build_vals)):
      subplot(2,1,1)
      plot(np.mean(allTOD[i], axis=1),label='Nf build = {0:}'.format(nf_build_vals[i]),lw=4/(i+1))
      title('Mean')
      subplot(2,1,2)
      plot(np.std(allTOD[i], axis=1),label='Nf build = {0:}'.format(nf_build_vals[i]),lw=3/(i+1))
      title('RMS')
legend()







############### Code Matt
from qubic import (create_random_pointings, gal2equ,
                  read_spectra,
                  compute_freq,
                  QubicScene,
                  QubicMultibandInstrument,
                  QubicMultibandAcquisition,
                  PlanckAcquisition,
                  QubicMultibandPlanckAcquisition)
import qubic
center = qubic.gal2equ(0,0)
effective_duration=2.
nside=128
#x0 = fits.getdata( dbdir+“/input_cmb_ns128.fits”)
x0 = np.zeros((12*nside**2,3))
x0.T[0] = 1.#1e2/1.36e-17
x0.T[1:] = 0.

p = qubic.create_random_pointings(center, 1000, 10.)
s = qubic.QubicScene(nside=nside, kind='IQU')

q = qubic.QubicInstrument()
atod = qubic.QubicAcquisition(q, p, s, effective_duration=effective_duration)
TOD,m = atod.get_observation(x0, noiseless=True, convolution=True)

Nbfreq_in, nus_edge_in, nus_in, deltas_in, Delta_in, Nbbands_in = qubic.compute_freq(150, 0.25, 5)
q = qubic.QubicMultibandInstrument(filter_nus=nus_in * 1e9, filter_relative_bandwidths=deltas_in / nus_in)
atod = qubic.QubicMultibandAcquisition(q, p, s, nus_edge_in, effective_duration=effective_duration)
TODm,m = atod.get_observation(array([x0]*len(q)), noiseless=True, convolution=True)

Nbfreq_in, nus_edge_in, nus_in, deltas_in, Delta_in, Nbbands_in = qubic.compute_freq(150, 0.25, 5)
q = qubic.QubicMultibandInstrument(filter_nus=nus_in * 1e9, filter_relative_bandwidths=deltas_in / nus_in)
atod = qubic.QubicPolyAcquisition(q, p, s, effective_duration=effective_duration)
TODp,m = atod.get_observation(x0, noiseless=True, convolution=True)


clf()
plot(TOD[:,0], lw=3)
plot(TODm[:,0],'r--', lw=3)
plot(TODp[:,0], 'g-.', lw=3)







