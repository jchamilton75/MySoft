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
from pysimulators import FitsArray
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

#### CMB Maps
params['tensor_ratio']=0.02
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
T[0:50]=0
ell = np.arange(1,lmaxcamb+1)
fact = (ell*(ell+1))/(2*np.pi)
spectra = [ell, T/fact, E/fact, B/fact, X/fact]
nside = 256
Nbpixels = 12*nside**2
scene = QubicScene(nside, kind='IQU')
mapI,mapQ,mapU=hp.synfast(spectra[1:],nside,new=True, pixwin=True)
cmb=np.array([mapI,mapQ,mapU]).T

FitsArray(cmb).save('CMB_MAPS.fits')

#### Dust Maps
coef=1.39e-2
spectra_dust = [ell, np.zeros(len(ell)), coef*(ell/80)**(-0.42)/(fact*0.52), coef*(ell/80)**(-0.42)/fact, np.zeros(len(ell))]
dustT,dustQ,dustU= hp.synfast(spectra_dust[1:],nside,new=True, pixwin=True)
dust=np.array([dustT,dustQ,dustU]).T
FitsArray(dust).save('DUST_MAPS.fits')


#### Scanning Startegy
racenter = 0.0      # deg
deccenter = -57.0   # deg
center = equ2gal(racenter, deccenter)
sampling = create_random_pointings([racenter, deccenter], 1000,20)


#create_tod.py True 0. 0.25 150. 5e-28 0.01 100 20

reload(sig)
photon_noise = sys.argv[1]
detector_nep = 2.7e-17*np.float(sys.argv[2])
dnu_nu_all = np.float(sys.argv[3])
nu_center = np.float(sys.argv[4])
nu_min=nu_center*(1.-dnu_nu_all/2)
nu_max=nu_center*(1.+dnu_nu_all/2)
effective_duration=np.float(sys.argv[5])
subdelta_construction=np.float(sys.argv[6])
#subdelta_convolved=0.01
#subdelta_reconstruction=0.01
racenter = 0.0      # deg
deccenter = -57.0   # deg
center = equ2gal(racenter, deccenter)




### Scanning:
npts = np.int(float(sys.argv[7]))
ang = np.int(float(sys.argv[8]))
sampling = create_random_pointings([racenter, deccenter], npts, ang)

FitsArray([sampling.azimuth, sampling.elevation, sampling.pitch, sampling.angle_hwp], copy=False).save('SAMPLING_{0:}_{1:}_{2:}_{3:}_{4:}_{5:}_{6:}_{7:}.fits'.format(photon_noise, 
	detector_nep, dnu_nu_all, nu_center, effective_duration, subdelta_construction, npts, ang))

### TOD
print('######## Making TOD')
Y,obs=sig.TOD(nu_min,nu_max,subdelta_construction,cmb,dust,sampling,scene,effective_duration,
              verbose=False, photon_noise=photon_noise, detector_nep=detector_nep)


FitsArray(Y, copy=False).save('TOD_{0:}_{1:}_{2:}_{3:}_{4:}_{5:}_{6:}_{7:}.fits'.format(photon_noise, 
	detector_nep, dnu_nu_all, nu_center, effective_duration, subdelta_construction, npts, ang))










