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
covlim=0.1


####### Create some sampling
center = equ2gal(racenter, deccenter)
sampling = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)

####### Full instrument
instrument = QubicInstrument(filter_nu=150e9,
                    detector_nep=4.7e-17*np.sqrt(len(sampling) * sampling.period / (365 * 86400)))

### Input map
nside_in=64
mapi,mapq,mapu=hp.synfast(spectra[1:],nside_in,new=True)
input_maps=np.array([mapi,mapq,mapu]).T

### Prepare input / output data
nside = 64
scene = QubicScene(nside, kind='IQU')




from Quad import mapmake_jc_lib as mm
reload(mm)

tod_signal, tod_noise, tod = mm.get_tod(instrument, sampling, scene, input_maps, 
                                        withplanck = False, photon_noise=False)


def powspec_inst(ts, tod):
    ps = np.abs(fft(tod, axis=1))**2
    avps = np.mean(ps, axis=0)
    freq = fftfreq(len(avps), ts)
    mask = freq>0
    return(freq[mask], avps[mask])
    
f, psn = powspec_inst(ts, tod_noise)
f, pss = powspec_inst(ts, tod_signal)
f, pstot = powspec_inst(ts, tod)
clf()
plot(f, psn)
plot(f, pss)
plot(f, pstot)
yscale('log')
xscale('log')



##### loop with filtered CMB maps to see the ell frequency correspondance
nside_in=64

def filt_spec(spectra, ell0, deltal):
    theell = spectra[0]
    newspec = [theell]
    mask = np.abs(theell-ell0) < deltal
    for s in spectra[1:]:
        news = s.copy()
        news[~mask]=0
        newspec.append(news)
    return(newspec)

ell0_vals = np.arange(50,200,25)

allps =[]
for l0 in ell0_vals:
    print(l0)
    newspec = filt_spec(spectra, l0,1)
    mapi,mapq,mapu=hp.synfast(newspec[1:],nside_in,new=True)
    input_maps=np.array([mapi,mapq,mapu]).T
    tod_signal, tod_noise, tod = mm.get_tod(instrument, sampling, scene, input_maps, 
                                            withplanck = False, photon_noise=False)
    f, pss = powspec_inst(ts, tod_signal)
    allps.append(pss)

clf()
yscale('log')
xscale('log')
for i in xrange(len(ell0_vals)):
    plot(f, allps[i],label=ell0_vals[i])
legend(loc='upper left')















