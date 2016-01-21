from __future__ import division
from pylab import *
import healpy as hp
from matplotlib.pyplot import *
import numpy as np
import os
import sys
import qubic
import pycamb
import string
import random
from pyoperators import DenseBlockDiagonalOperator, Rotation3dOperator
from pysimulators import FitsArray
from pyoperators import MPI

from qubic import (
    QubicAcquisition, create_sweeping_pointings, equ2gal, map2tod, tod2map_all,
    tod2map_each, create_random_pointings, QubicInstrument)

rank = MPI.COMM_WORLD.rank


def random_string(nchars):
    lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(nchars)]
    str = "".join(lst)
    return(str)


############# Reading arguments ################################################
print 'Number of arguments:', len(sys.argv), 'arguments.'
if len(sys.argv) > 1:
	print 'Argument List:', str(sys.argv)
	noise = np.float(sys.argv[1])
	nside = np.int(sys.argv[2])
	sigptg = np.float(sys.argv[3])
else:
	noise = 1.
	nside = 256
	sigptg = 60.

print('Noise level is set to '+np.str(noise))
print('nside is set to '+np.str(nside))
print('Sigma Pointing is set to '+np.str(sigptg))
##################################################################################




############# Input Power spectrum ##############################################
H0 = 67.04
omegab = 0.022032
omegac = 0.12038
h2 = (H0/100.)**2
scalar_amp = np.exp(3.098)/1.E10
omegav = h2 - omegab - omegac
Omegab = omegab/h2
Omegac = omegac/h2
print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',omegav/h2

params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':0.1,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}

lmaxcamb = 3*512
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
lll = np.arange(1,lmaxcamb+1)
fact = (lll*(lll+1))/(2*np.pi)
spectra = [lll, T/fact, E/fact, B/fact, X/fact]
##################################################################################




############# Parameters ##########################################################
racenter = 0.0      # deg
deccenter = -57.0   # deg
center = equ2gal(racenter, deccenter)
duration = 24       # hours
ts = 0.1            # seconds
ang = 20            # degrees
##################################################################################




############## Instrument ########################################################
instFull = QubicInstrument(detector_tau=0.0001,
                            detector_sigma=noise,
                            detector_fknee=0.,
                            detector_fslope=1)
##################################################################################


for ii in np.arange(100):
  ############## True Pointing #####################################################
  sampling = create_random_pointings([racenter, deccenter], duration*3600/ts, ang, period=ts)
  hwp_angles = np.random.random_integers(0, 7, len(sampling)) * 11.25 
  sampling.pitch = 0
  sampling.angle_hwp = hwp_angles
  npoints = len(sampling)
  ##################################################################################

  ############## Spoiled Pointing #####################################################
  new_sampling=create_random_pointings([racenter, deccenter], duration*3600/ts, ang, period=ts)
  dtheta=np.random.randn(npoints)*sigptg/3600
  dphi=np.random.randn(npoints)*sigptg/3600/np.sin( (90-sampling.elevation)*np.pi/180)
  new_sampling.elevation = sampling.elevation+dtheta
  new_sampling.azimuth = sampling.azimuth+dphi
  new_sampling.pitch = sampling.pitch
  new_sampling.angle_hwp = sampling.angle_hwp
  ##################################################################################

  ############## Input maps ########################################################
  x0 = None
  if rank == 0:
    print('Rank '+str(rank)+' is Running Synfast')
    x0 = np.array(hp.synfast(spectra[1:],nside,fwhm=0,pixwin=True,new=True)).T

  x0 = MPI.COMM_WORLD.bcast(x0)
  x0_noI = x0.copy()
  x0_noI[:,0] = 0
  print('Initially I map RMS is : '+str(np.std(x0[:,0])))
  print('Initially Q map RMS is : '+str(np.std(x0[:,1])))
  print('Initially U map RMS is : '+str(np.std(x0[:,2])))
  print('new I map RMS is : '+str(np.std(x0_noI[:,0])))
  print('new Q map RMS is : '+str(np.std(x0_noI[:,1])))
  print('new U map RMS is : '+str(np.std(x0_noI[:,2])))
  ##################################################################################





  ############# Make TODs ###########################################################
  acquisition = QubicAcquisition(instFull, new_sampling,
                                 nside=nside,
                                 synthbeam_fraction=0.99)
  tod, x0_convolved = map2tod(acquisition, x0, convolution=True)
  tod_noI, x0_noI_convolved = map2tod(acquisition, x0_noI, convolution=True)
  todnoise = acquisition.get_noise()
  ##################################################################################




  ############# Make maps ###########################################################
  th_acquisition = QubicAcquisition(instFull, sampling,
                                 	nside=nside,
                                 	synthbeam_fraction=0.99)

  strrnd = random_string(10)

  fits_noI_input = 'maps_ns'+str(nside)+'_noise'+str(noise)+'_sigptg'+str(sigptg)+'_noI_input_'+strrnd+'.fits'
  fits_input = 'maps_ns'+str(nside)+'_noise'+str(noise)+'_sigptg'+str(sigptg)+'_input_'+strrnd+'.fits'


  fits_noiseless = 'maps_ns'+str(nside)+'_noise'+str(noise)+'_sigptg'+str(sigptg)+'_noiseless_'+strrnd+'.fits'
  fits_noise = 'maps_ns'+str(nside)+'_noise'+str(noise)+'_sigptg'+str(sigptg)+'_noisy_'+strrnd+'.fits'
  fits_spoiled_noiseless = 'maps_ns'+str(nside)+'_noise'+str(noise)+'_sigptg'+str(sigptg)+'_spoiled_noiseless_'+strrnd+'.fits'
  fits_spoiled_noise = 'maps_ns'+str(nside)+'_noise'+str(noise)+'_sigptg'+str(sigptg)+'_spoiled_noisy_'+strrnd+'.fits'
  fits_noI_noiseless = 'maps_ns'+str(nside)+'_noise'+str(noise)+'_sigptg'+str(sigptg)+'_noI_noiseless_'+strrnd+'.fits'
  fits_noI_noise = 'maps_ns'+str(nside)+'_noise'+str(noise)+'_sigptg'+str(sigptg)+'_noI_noisy_'+strrnd+'.fits'
  fits_noI_spoiled_noiseless = 'maps_ns'+str(nside)+'_noise'+str(noise)+'_sigptg'+str(sigptg)+'_noI_spoiled_noiseless_'+strrnd+'.fits'
  fits_noI_spoiled_noise = 'maps_ns'+str(nside)+'_noise'+str(noise)+'_sigptg'+str(sigptg)+'_noI_spoiled_noisy_'+strrnd+'.fits'

  fits_cov = 'cov_ns'+str(nside)+'_noise'+str(noise)+'_sigptg'+str(sigptg)+'_'+strrnd+'.fits'
  fits_spoiled_cov = 'cov_ns'+str(nside)+'_noise'+str(noise)+'_sigptg'+str(sigptg)+'_spoiled_'+strrnd+'.fits'


  coverage_threshold = 0.01
  print('Making noiseless map with true coverage')
  maps_noiseless, cov_noiseless = tod2map_all(acquisition, tod, tol=1e-4, coverage_threshold=coverage_threshold)
  if rank == 0:
    print('Saving the noiseless map with true coverage: '+fits_noiseless)
    qubic.io.write_map(fits_noiseless,maps_noiseless) 



  mask = cov_noiseless == 0
  x0_convolved[mask,:] = 0
  x0_noI_convolved[mask,:] = 0

  if rank == 0:
    print('Saving the input map: '+fits_input)
    qubic.io.write_map(fits_input,x0_convolved) 
    print('Saving the noI input map: '+fits_noI_input)
    qubic.io.write_map(fits_noI_input, x0_noI_convolved) 




  print('Making noiseless map with true coverage')
  maps_noiseless, cov_noiseless = tod2map_all(acquisition, tod, tol=1e-4, coverage_threshold=coverage_threshold)
  if rank == 0:
    print('Saving the noiseless map with true coverage: '+fits_noiseless)
    qubic.io.write_map(fits_noiseless,maps_noiseless) 
    qubic.io.write_map(fits_cov,cov_noiseless)


  print('Making noisy map with true coverage')
  maps_noise, cov_noise = tod2map_all(acquisition, tod + todnoise, tol=1e-4, coverage_threshold=coverage_threshold)
  if rank == 0:
    print('Saving the noisy map with true coverage: '+fits_noise)
    qubic.io.write_map(fits_noise,maps_noise) 


  print('Making noiseless map with spoiled coverage')
  maps_spoiled_noiseless, cov_spoiled_noiseless = tod2map_all(th_acquisition, tod, tol=1e-4, coverage_threshold=coverage_threshold)
  if rank == 0:
    print('Saving the noiseless map with true coverage: '+fits_spoiled_noiseless)
    qubic.io.write_map(fits_spoiled_noiseless,maps_spoiled_noiseless) 
    qubic.io.write_map(fits_spoiled_cov,cov_spoiled_noiseless) 

  print('Making noisy map with spoiled coverage')
  maps_spoiled_noise, cov_spoiled_noise = tod2map_all(th_acquisition, tod + todnoise, tol=1e-4, coverage_threshold=coverage_threshold)
  if rank == 0:
    print('Saving the noisy map with true coverage: '+fits_spoiled_noise)
    qubic.io.write_map(fits_spoiled_noise,maps_spoiled_noise) 






  print('Making noI noiseless map with true coverage')
  maps_noI_noiseless, cov_noI_noiseless = tod2map_all(acquisition, tod_noI, tol=1e-4, coverage_threshold=coverage_threshold)
  if rank == 0:
  	print('Saving the noI noiseless map with true coverage: '+fits_noI_noiseless)
  	qubic.io.write_map(fits_noI_noiseless,maps_noI_noiseless) 

  print('Making noI noisy map with true coverage')
  maps_noI_noise, cov_noI_noise = tod2map_all(acquisition, tod_noI + todnoise, tol=1e-4, coverage_threshold=coverage_threshold)
  if rank == 0:
  	print('Saving the noI noisy map with true coverage: '+fits_noI_noise)
  	qubic.io.write_map(fits_noI_noise,maps_noI_noise) 

  print('Making noI noiseless map with spoiled coverage')
  maps_noI_spoiled_noiseless, cov_noI_spoiled_noiseless = tod2map_all(th_acquisition, tod_noI, tol=1e-4, coverage_threshold=coverage_threshold)
  if rank == 0:
  	print('Saving the noI noiseless map with true coverage: '+fits_noI_spoiled_noiseless)
  	qubic.io.write_map(fits_noI_spoiled_noiseless,maps_noI_spoiled_noiseless) 

  print('Making noI noisy map with spoiled coverage')
  maps_noI_spoiled_noise, cov_noI_spoiled_noise = tod2map_all(th_acquisition, tod_noI + todnoise, tol=1e-4, coverage_threshold=coverage_threshold)
  if rank == 0:
  	print('Saving the noI noisy map with true coverage: '+fits_noI_spoiled_noise)
  	qubic.io.write_map(fits_noI_spoiled_noise,maps_noI_spoiled_noise) 


















