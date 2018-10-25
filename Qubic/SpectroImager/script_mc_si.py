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
import time


######## Default configuration
### Sky 
nside = 256
center_gal = 0, 90
center = gal2equ(center_gal[0], center_gal[1])
dust_coeff = 1.39e-2
seed=None

### Detectors (for now using random pointing)
band = 150
relative_bandwidth = 0.25
sz_ptg = 10.
nb_ptg = 1000
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
				'nf_sub_rec': nf_sub_rec, 'noiseless':noiseless, 'seed':seed }


name = sys.argv[1]
nfoutmax = int(sys.argv[2])
arguments = sys.argv[3:]
nargs = int(len(arguments)/2)
print(nargs)

for i in xrange(nargs):
      print('seeting: {0} to {1}'.format(arguments[2*i],arguments[2*i+1]))
      parameters[arguments[2*i]] = float(arguments[2*i+1])

parameters['nf_sub_build']=int(parameters['nf_sub_build'])
parameters['nb_ptg']=int(parameters['nb_ptg'])
parameters['nside']=int(parameters['nside'])
parameters['seed']=int(parameters['seed'])

for k in parameters.keys(): print(k, parameters[k])



print('Creating input sky')
t0 = time.time()
x0 = si.create_input_sky(parameters)
t1 = time.time()
print('done in {} seconds'.format(t1-t0))
print('Creating pointing')
p = si.create_random_pointings(parameters['center'], parameters['nb_ptg'], parameters['sz_ptg'])
t2 = time.time()
print('done in {} seconds'.format(t2-t1))
print('Creating TOD')
TOD = si.create_TOD(parameters, p, x0)
t3 = time.time()
print('done in {} seconds'.format(t3-t2))

print('Now looping over nf_sub_rec for map-making:')
#for i in np.arange(nfoutmax)+1:
i = nfoutmax
print('Doing Mapmaking on {} sub-map(s)'.format(i))
tinit = time.time()
parameters['nf_sub_rec'] = i
maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, parameters, p, x0=x0)
if int(parameters['nf_sub_rec'])==1: maps_recon=np.reshape(maps_recon, np.shape(maps_convolved))
cov = np.sum(cov, axis=0)
maxcov = np.max(cov)
unseen = cov < maxcov*0.1
diffmap = maps_convolved - maps_recon
maps_convolved[:,unseen,:] = hp.UNSEEN
maps_recon[:,unseen,:] = hp.UNSEEN
diffmap[:,unseen,:] = hp.UNSEEN
therms = np.std(diffmap[:,~unseen,:], axis = 1)

FitsArray(therms).save(name+'_nf{0}'.format(i)+'_rms.fits')
FitsArray(nus).save(name+'_nf{0}'.format(i)+'_nus.fits')
FitsArray(nus_edge).save(name+'_nf{0}'.format(i)+'_nus_edges.fits')
FitsArray(maps_convolved).save(name+'_nf{0}'.format(i)+'_maps_convolved.fits')
FitsArray(maps_recon).save(name+'_nf{0}'.format(i)+'_maps_recon.fits')
tfin = time.time()
print('      done in {} seconds'.format(tfin-tinit))


tend = time.time()
print('Finished all in {} seconds'.format(tend-t0))



