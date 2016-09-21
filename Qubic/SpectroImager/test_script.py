from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
#from Quad import qml
#from Quad import pyquad
import healpy as hp
from pyoperators import MPI, BlockDiagonalOperator, BlockRowOperator,BlockColumnOperator, DiagonalOperator, PackOperator, pcg , asoperator , MaskOperator
from pysimulators.interfaces.healpy import (HealpixConvolutionGaussianOperator)
from qubic import (
    QubicAcquisition, QubicInstrument,
    QubicScene, create_sweeping_pointings, equ2gal, create_random_pointings, PlanckAcquisition, QubicPlanckAcquisition)
#from MYacquisition import PlanckAcquisition, QubicPlanckAcquisition
from qubic.io import read_map, write_map
import gc
import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)
import sys
import os
from qubic.data import PATH
from functools import reduce
import pycamb

from SpectroImager import SpectroImagerGohar as sig


######################################################
### Cosmological parameters and CMB power spectrum ###
######################################################

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
             'tensor_ratio':0.05,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}

lmaxcamb = 3*512
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
T[0:50]=0
ell = np.arange(1,lmaxcamb+1)
fact = (ell*(ell+1))/(2*np.pi)
spectra = [ell, T/fact, E/fact, B/fact, X/fact]


#############
### Scene ###
#############

nside = 256
Nbpixels = 12*nside**2
scene = QubicScene(nside, kind='IQU')


###############
### Cmb map ###
###############

mapI,mapQ,mapU=hp.synfast(spectra[1:],nside,new=True, pixwin=True)
cmb=np.array([mapI,mapQ,mapU]).T

#########################################
### Dust power spectrum, map, scaling ###
#########################################

coef=1.39e-2
spectra_dust = [ell, np.zeros(len(ell)), coef*(ell/80)**(-0.42)/(fact*0.52), coef*(ell/80)**(-0.42)/fact, np.zeros(len(ell))]
dustT,dustQ,dustU= hp.synfast(spectra_dust[1:],nside,new=True, pixwin=True)
## dust map at 150GHz :
dust=np.array([dustT,dustQ,dustU]).T


################
### Sampling ###
################

maxiter = 100
tol = 5e-6
racenter = 0.0      # deg
deccenter = -57.0   # deg
angspeed = 1        # deg/sec
delta_az = 20.      # deg
angspeed_psi = 0.1  # deg/sec
maxpsi = 45.        # deg
nsweeps_el = 300
duration = 24      # hours
ts = 1000        # seconds
center = equ2gal(racenter, deccenter)




reload(sig)
dnu_nu_all = 0.25
nu_min=150*(1.-dnu_nu_all/2)
nu_max=150*(1.+dnu_nu_all/2)
effective_duration=5000 
subdelta_construction=0.01
subdelta_convolved=0.01
subdelta_reconstruction=0.025
racenter = 0.0      # deg
deccenter = -57.0   # deg
center = equ2gal(racenter, deccenter)
#all_delta_nu = np.linspace(0.03, 0.25, 11)
all_delta_nu = np.array([0.125, 0.25])
allres = []
allnoise = []
### Scanning:
sampling = create_random_pointings([racenter, deccenter], 300,20)
### TOD
Y,obs=sig.TOD(nu_min,nu_max,subdelta_construction,cmb,dust,sampling,scene,effective_duration,verbose=True, photon_noise=True)
for i in xrange(len(all_delta_nu)):
    dnu_nu = all_delta_nu[i]
    print('############ DELTANU/NU {0:6.3f} number {1:} over {2:}'.format(dnu_nu, i, len(all_delta_nu)))
    x0_convolved=sig.convolved_true_maps(nu_min,nu_max,dnu_nu,subdelta_convolved,cmb,dust)
    x0_convolved[:,~obs]=0
    ### Maps
    maps, bands, deltas=sig.reconstruct(Y,nu_min,nu_max,delta_nu,sdr,
        sampling,scene,effective_duration)
    res=maps-x0_convolved
    noise_poly=np.std(res[:,obs], axis=1)
    allres.append(res)
    allnoise.append(noise_poly)


nb = 0
for i in xrange(len(all_delta_nu)):
    subnoises = np.array(allnoise[i])
    print("Nu:",i, subnoises.shape)
    print(subnoises)
    sh = np.shape(subnoises)
    #recombine = np.sqrt(np.sum(subnoises**2, axis = 0))/np.sqrt(sh[0])



for j in xrange(2):
    theres = allres[j]
    sh=theres.shape
    nbbands = sh[0]
    figure(j) 
    [(hp.gnomview((theres[i,:,1]).T,sub=(1,nbbands,i+1),
               title='Q polarization - band={0:} / {1:} - RMS={2:5.2f}'.format(i+1,nbbands,allnoise[j][i,1]),
                  rot=center,reso=reso,min=-3,max=3)) for i in xrange(nbbands)]






