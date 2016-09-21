from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
from Quad import qml
from Quad import pyquad
from Quad import mapmake_jc_lib as mm
import healpy as hp
from pyoperators import MPI, DiagonalOperator, PackOperator, pcg
from qubic import (
    QubicAcquisition, QubicInstrument,
    QubicScene, create_sweeping_pointings, equ2gal, create_random_pointings)
from MYacquisition import PlanckAcquisition, QubicPlanckAcquisition
from qubic.io import read_map, write_map
import gc
import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)
import sys
import os
from qubic.data import PATH

def powspec_inst(ts, tod):
    ps = np.abs(fft(tod, axis=1))**2
    avps = np.mean(ps, axis=0)
    freq = fftfreq(len(avps), ts)
    mask = freq>0
    return(freq[mask], avps[mask])
    



#### Get input Power spectra
################# Input Power spectrum ###################################
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
import pycamb
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
ell = np.arange(1,lmaxcamb+1)
fact = (ell*(ell+1))/(2*np.pi)
spectra = [ell, T/fact, E/fact, B/fact, X/fact]


clf()
plot(ell,np.sqrt(spectra[1]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TT}$')
plot(ell,np.sqrt(abs(spectra[4])*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TE}$')
plot(ell,np.sqrt(spectra[2]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{EE}$')
plot(ell,np.sqrt(spectra[3]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{BB} (r=0.1)$')
yscale('log')
xlim(0,600)
ylim(0.01,100)
xlabel('$\ell$')
ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
legend(loc='lower right',frameon=False)

### Input map
nside_in=64
mapi,mapq,mapu=hp.synfast(spectra[1:],nside_in,new=True)
input_maps=np.array([mapi,mapq,mapu]).T

######## Simulation parameters
maxiter = 1000
tol = 5e-6
racenter = 0.0      # deg
deccenter = -57.0   # deg
angspeed = 1        # deg/sec
delta_az = 20.      # deg
angspeed_psi = 0.1  # deg/sec
maxpsi = 45.        # deg
nsweeps_el = 300
duration = 24       # hours
npow = 23
ts = 24*3600/2**npow            # seconds Chosen in order to have a power of 2 in 24 hours
np.random.seed(0)
center = equ2gal(racenter, deccenter)
covlim =0.1

####### Create some sampling
sampling = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)

    
### Prepare input / output data
nside = 64
scene = QubicScene(nside, kind='IQU')
####### Full instrument
f = 1
fslope =1
signoise = 4.7e-17
instrument = QubicInstrument(filter_nu=150e9,detector_nep=signoise,detector_fknee = f,detector_fslope = fslope)

####### Only one detector
det = 231
instrument_one= instrument[231]

acq = QubicAcquisition(instrument_one, sampling, scene, photon_noise=False)
noise = acq.get_noise()

sigexp = signoise /sqrt(2) / np.sqrt(ts)
print(std(noise), sigexp)

import scipy.ndimage.filters as filt
std(noise)
freq, ps = powspec_inst(ts, noise)
clf()
plot(freq,filt.gaussian_filter1d(ps,100))
plot(freq, sigexp**2*(1+ (f/freq)**fslope)*len(sampling), 'r', lw=2)
yscale('log')
xscale('log')
xlim(1e-3, 100)



