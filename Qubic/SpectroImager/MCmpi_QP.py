#!/bin/env python
from __future__ import division
import sys
import healpy as hp
import numpy as np

import matplotlib.pyplot as mp
import qubic
from SpectroImager import SpectroImLibQP as si
from pysimulators import FitsArray
import time

from mpi4py import MPI
#from pyoperators import MPI
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



#print '========================================================== Hello ! I am rank number {}'.format(rank)






#### Reading input dictionary and replacing with command line arguments the default params ###########################
# dictfilename = sys.argv[1]
# name = sys.argv[2]
# tol = float(sys.argv[3])
# minnfreq = int(sys.argv[4])
# maxnfreq = int(sys.argv[5])
# import distutils.util
# noI = distutils.util.strtobool(sys.argv[6])
# arguments = sys.argv[7:]
# nargs = int(len(arguments)/2)

dictfilename = '/Users/hamilton/Qubic/SpectroImager/testFI.dict'
name = 'Test_Q'
tol = 1e-4
minnfreq = 1
maxnfreq = 3
noI = False
arguments = ['npointings', '1000', 'seed',  '1', 'nf_sub', '15', 'photon_noise', False, 'detector_nep', '1e-20']
nargs = int(len(arguments)/2)


d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

print(sys.argv)
print(arguments)
for i in xrange(nargs):
      print('seeting: {0} to from {1} to {2}'.format(arguments[2*i],d[arguments[2*i]],arguments[2*i+1]))
      d[arguments[2*i]] = type(d[arguments[2*i]])(arguments[2*i+1])


if rank==0: 
      for k in d.keys(): 
            print(k, d[k])
      print('Dictionnary File: '+dictfilename)
      print('Simulation General Name: '+name)
      print('Mapmaking Tolerance: {}'.format(tol))
      print('Maximum Number of Sub Frequencies: {}'.format(maxnfreq))
## Input sky parameters
skypars = {'dust_coeff':1.39e-2, 'r':0}
####################################################################################################


##### Sky Creation made ony on rank 0 #################################################################################
if rank==0:
      t0 = time.time()
      x0 = si.create_input_sky(d, skypars)
      print('Imean:',np.mean(x0[:,:,0]))
      print(noI)
      if noI==True:
            x0[:,:,0] = 0
      print('Imean:',np.mean(x0[:,:,0]))
      t1 = time.time()
      print('********************* Input Sky - Rank {} - done in {} seconds'.format(rank, t1-t0))
else:
      x0 = None
      x0_Planck = None
      t0 = time.time()


x0 = MPI.COMM_WORLD.bcast(x0)


##### Pointing in not picklable so cannot be broadcasted => done on all ranks simultaneously
t1 = time.time()
p = qubic.get_pointing(d)
t2 = time.time()
#print('************************** Pointing - rank {} - done in {} seconds'.format(rank, t2-t1))


##### TOD making is intrinsically parallelized (use of pyoperators)
#print('-------------------------- TOD - rank {} Starting'.format(rank))
TOD = si.create_TOD(d, p, x0)
#print('************************** TOD - rank {} Done - elaplsed time is {}'.format(rank,time.time()-t0))

##### Wait for all the TOD to be done (is it necessary ?)
MPI.COMM_WORLD.Barrier()
if rank == 0:
      t1 = time.time()
      print('************************** All TOD OK in {} minutes'.format((t1-t0)/60))




for nf_sub_rec in np.arange(minnfreq,maxnfreq+1):
      ##### Mapmaking
      if rank == 0:
            print('-------------------------- Map-Making on {} sub-map(s) - Rank {} Starting'.format(nf_sub_rec,rank))
      maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p, nf_sub_rec, tol=tol, x0=x0, PlanckMaps=x0)
      if nf_sub_rec==1: maps_recon=np.reshape(maps_recon, np.shape(maps_convolved))
      cov = np.sum(cov, axis=0)
      maxcov = np.max(cov)
      unseen = cov < maxcov*0.1
      diffmap = maps_convolved - maps_recon
      maps_convolved[:,unseen,:] = hp.UNSEEN
      maps_recon[:,unseen,:] = hp.UNSEEN
      diffmap[:,unseen,:] = hp.UNSEEN
      therms = np.std(diffmap[:,~unseen,:], axis = 1)
      if rank == 0:
            print('************************** Map-Making on {} sub-map(s) - Rank {} Done'.format(nf_sub_rec,rank))

      MPI.COMM_WORLD.Barrier()

      if rank == 0:
            FitsArray(nus).save(name+'_nf{0}'.format(nf_sub_rec)+'_nus.fits')
            FitsArray(nus_edge).save(name+'_nf{0}'.format(nf_sub_rec)+'_nus_edges.fits')
            FitsArray(maps_convolved).save(name+'_nf{0}'.format(nf_sub_rec)+'_maps_convolved.fits')
            FitsArray(maps_recon).save(name+'_nf{0}'.format(nf_sub_rec)+'_maps_recon.fits')
            print('************************** rank {} saved fits files'.format(rank))
            t1 = time.time()
            print('************************** All Done in {} minutes'.format((t1-t0)/60))

      MPI.COMM_WORLD.Barrier()

MPI.Finalize()

