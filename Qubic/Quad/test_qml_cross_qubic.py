from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
from Quad import qml
from Quad import pyquad
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
ts = 5.             # seconds
np.random.seed(0)
center = equ2gal(racenter, deccenter)



######### try a first QUBIC mapmaking
sampling = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)
instrument = QubicInstrument(filter_nu=150e9,
                    detector_nep=0.0001*4.7e-17*np.sqrt(len(sampling) * sampling.period / (365 * 86400)))

### Input map
nside_in=64
mapi,mapq,mapu=hp.synfast(spectra[1:],nside_in,new=True)
#hp.mollview(mapi)
#hp.mollview(mapq)
#hp.mollview(mapu)
x0=np.array([mapi,mapq,mapu]).T

### Prepare input / output data
nside = 64
scene = QubicScene(nside, kind='IQU')
acq = QubicAcquisition(instrument, sampling, scene)
C = acq.get_convolution_peak_operator()
coverage = acq.get_coverage()
observed = coverage > 0.1*np.max(coverage)
acq_restricted = acq[:, :, observed]
H = acq_restricted.get_operator()
x0_convolved = C(x0)
acq_planck = PlanckAcquisition(150, acq.scene, true_sky=x0_convolved, fix_seed=True, factor=1)
acq_fusion = QubicPlanckAcquisition(acq, acq_planck)
map_planck_obs=acq_planck.get_observation()
H = acq_fusion.get_operator()
invntt = acq_fusion.get_invntt_operator()
A = H.T * invntt * H

### Questions
# - how to have input map at 256 and reconstructed at 64 ?
# - am I getting new QUBIC noise by calling twice acq_fusion.get_observation()
#       if not: how to do it ?
# - Actually I want to make maps with 50% of the detectors rather than calling it twice
# - The Planck noise will be on both maps, and will therefore remain as a bias ?
# - I need to try with pure QUBIC reconstruction as well



#### EB Only
mask=~observed
maskok=observed

ip=np.arange(12*nside**2)
ipok=ip[~mask]

#ellbins=[0,50,100,150,200,250,300,350,3*nside]
ellbins = [0,50,75,100,125,150,175,200]
nbins=len(ellbins)-1
ellmin=np.array(ellbins[0:nbins])
ellmax=np.array(ellbins[1:nbins+1])-1
ellval=(ellmin+ellmax)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)

reload(qml)
fwhmrad = instrument.synthbeam.peak.fwhm
ll=np.arange(int(np.max(ellbins))+1)
bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)

#ds_dcb=qml.compute_ds_dcb_parpix(ellbins,nside,ipok,bl,polar=True,temp=False,nprocs=8)

### Data 1
y = acq_fusion.get_observation()
b = H.T * invntt * y
solution_fusion = pcg(A, b, disp=True, maxiter=1000, tol=1e-3)
map_fusion = solution_fusion['x']
map_fusion[~observed,:] = 0
themaps = [map_fusion[:,1], map_fusion[:,2]]

### Data 2
y2 = acq_fusion.get_observation()
b2 = H.T * invntt * y2
solution_fusion2 = pcg(A, b2, disp=True, maxiter=1000, tol=1e-3)
map_fusion2 = solution_fusion2['x']
map_fusion2[~observed,:] = 0
themaps2 = [map_fusion2[:,1], map_fusion2[:,2]]

mapq_o = x0_convolved[:,1]
mapq_o[mask]=0
mapu_o = x0_convolved[:,2]
mapu_o[mask]=0
themaps_o = [mapq_o,mapu_o]
themaps2_o = [mapq_o,mapu_o]


### Likelihood
covmap=np.zeros(len(ipok)*len(themaps))
guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],0]
specout,error,invfisher,ds_dcb=qml.qml_cross_noiter(themaps, themaps2,mask,covmap,ellbins,fwhmrad,guess,
    ds_dcb,spectra,cholesky=False,temp=False,polar=True,plot=True)

specout_o,error_o,invfisher,ds_dcb=qml.qml_cross_noiter(themaps_o, themaps2_o,mask,covmap,ellbins,fwhmrad,guess,
    ds_dcb,spectra,cholesky=False,temp=False,polar=True,plot=True)

clf()
subplot(1,2,1)
xlim(0,np.max(ellmax)*1.2)
plot(ell,E,lw=3)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')
ylim(0,3)
plot(ellval,specout[2]*ellval*(ellval+1)/(2*np.pi), 'ro',label='QUBIC+Planck')
plot(ellval,specout_o[2]*ellval*(ellval+1)/(2*np.pi), 'go',label='Direct')
legend(numpoints=1)
subplot(1,2,2)
xlim(0,np.max(ellmax)*1.2)
plot(ell,B,lw=3)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')
plot(ellval,specout[3]*ellval*(ellval+1)/(2*np.pi), 'ro')
plot(ellval,specout_o[3]*ellval*(ellval+1)/(2*np.pi), 'go')
ylim(0,0.03)


clf()
plot(ellval, specout[2]/specout_o[2], 'bo-')
plot(ellval, specout[3]/specout_o[3], 'ro-')
ylim([-1,3])




