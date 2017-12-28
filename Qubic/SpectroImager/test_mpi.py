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

from mpi4py import MPI
import os


#### MPI stuff
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

if rank == 0:
      print('**************************')
      print('Master rank {} is speaking:'.format(rank))
      print('mpi is in')
      print('There are {} ranks'.format(size))
      print('**************************')



print '========================================================== Hello ! I am rank number {}'.format(rank)


######## Default configuration ################################################################
### Sky 
nside = 256
center_gal = 0, 90
center = gal2equ(center_gal[0], center_gal[1])
dust_coeff = 1.39e-2
seed=1

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
####################################################################################################




#### Reading input parameters and replacing in the default configuration ###########################
name = sys.argv[1]
arguments = sys.argv[2:]
nargs = int(len(arguments)/2)

for i in xrange(nargs):
      print('seeting: {0} to {1}'.format(arguments[2*i],arguments[2*i+1]))
      parameters[arguments[2*i]] = float(arguments[2*i+1])

parameters['nf_sub_build']=int(parameters['nf_sub_build'])
parameters['nf_sub_rec']=int(parameters['nf_sub_rec'])
parameters['nb_ptg']=int(parameters['nb_ptg'])
parameters['nside']=int(parameters['nside'])
parameters['seed']=int(parameters['seed'])

if rank==0: 
      for k in parameters.keys(): 
            print(k, parameters[k])
####################################################################################################





##### Sky Creation made ony on rank 0 #################################################################################
if rank==0:
      t0 = time.time()
      x0 = si.create_input_sky(parameters)
      t1 = time.time()
      print('********************* Input Sky - Rank {} - done in {} seconds'.format(rank, t1-t0))
else:
      x0 = None
      t0 = time.time()


x0 = MPI.COMM_WORLD.bcast(x0)
#print('rank {} {} {}'.format(rank, x0[0], x0[1]))


##### Pointing in not picklable so cannot be broadcasted => done on all ranks simultaneously
t1 = time.time()
p = si.create_random_pointings(parameters['center'], parameters['nb_ptg'], parameters['sz_ptg'], seed=parameters['seed'])
t2 = time.time()
print('************************** Pointing - rank {} - done in {} seconds'.format(rank, t2-t1))


##### TOD making is intrinsically parallelized (use of pyoperators)
print('-------------------------- TOD - rank {} Starting'.format(rank))
TOD = si.create_TOD(parameters, p, x0)
print('************************** TOD - rank {} Done - elaplsed time is {}'.format(rank,time.time()-t0))

##### Wait for all the TOD to be done (is it necessary ?)
MPI.COMM_WORLD.Barrier()
if rank == 0:
      t1 = time.time()
      print('************************** All TOD OK in {} minutes'.format((t1-t0)/60))


##### Mapmaking
print('-------------------------- Map-Making on {} sub-map(s) - Rank {} Starting'.format(nf_sub_rec,rank))
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
print('************************** Map-Making on {} sub-map(s) - Rank {} Done'.format(nf_sub_rec,rank))


if rank == 0:
      FitsArray(nus).save(name+'_nf{0}'.format(nf_sub_rec)+'_nus.fits')
      FitsArray(nus_edge).save(name+'_nf{0}'.format(nf_sub_rec)+'_nus_edges.fits')
      FitsArray(maps_convolved).save(name+'_nf{0}'.format(nf_sub_rec)+'_maps_convolved.fits')
      FitsArray(maps_recon).save(name+'_nf{0}'.format(nf_sub_rec)+'_maps_recon.fits')
      print('************************** rank {} saved fits files'.format(rank))
      t1 = time.time()
      print('************************** All Done in {} minutes'.format((t1-t0)/60))




