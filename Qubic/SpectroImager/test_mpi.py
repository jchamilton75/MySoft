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

from pyoperators import MPI
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



print ' Hello ! I am rank number {}'.format(rank)


######## Default configuration ################################################################
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
####################################################################################################



#### Reading input parameters and replacing in the default configuration ###########################
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

if rank==0: 
      for k in parameters.keys(): 
            print(k, parameters[k])
####################################################################################################





##### Now the code #################################################################################
if rank==0:
      print '**************************'
      print('rank {}'.format(rank))
      print('Creating input sky')
      t0 = time.time()
      x0 = si.create_input_sky(parameters)
      t1 = time.time()
      print('done in {} seconds'.format(t1-t0))
      MPI.COMM_WORLD.bcast(x0)
      print('Brodacasted')
      print '**************************'



#MPI.COMM_WORLD.Barrier()

if rank==0:
      print '**************************'
      print('rank {}'.format(rank))
      print('Creating pointing')
      t1 = time.time()
      p = si.create_random_pointings(parameters['center'], parameters['nb_ptg'], parameters['sz_ptg'])
      t2 = time.time()
      print('done in {} seconds'.format(t2-t1))
      MPI.COMM_WORLD.bcast(p)
      print('Brodacasted')
      print '**************************'


#MPI.COMM_WORLD.Barrier()

print('rank {} Creating TOD'.format(rank))
TOD = si.create_TOD(parameters, p, x0)

MPI.COMM_WORLD.Barrier()


print('rank {}: TOD values are: {} {} {}'.format(rank, TOD[0,1], TOD[0,2], TOD[0,3]))






