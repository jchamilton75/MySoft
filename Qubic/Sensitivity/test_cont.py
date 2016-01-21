from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import pi
import pycamb
from qubic import QubicInstrument
from scipy.constants import c
from Sensitivity import qubic_sensitivity
from Homogeneity import SplineFitting


#### Get input Power spectra
################# Input Power spectrum ###################################
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
r = 0.05
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
         'tensor_ratio':r,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}
lmaxcamb = 1000
T,E,B,X = pycamb.camb(lmaxcamb+1+150,**params)
lll = np.arange(1,lmaxcamb+1)
T=T[:lmaxcamb]
E=E[:lmaxcamb]
B=B[:lmaxcamb]
X=X[:lmaxcamb]
fact = (lll*(lll+1))/(2*np.pi)
spectra = [lll, T/fact, E/fact, B/fact, X/fact]

params_nl = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':r,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False}
lmaxcamb = 1000
Tnl,Enl,Bnl,Xnl = pycamb.camb(lmaxcamb+1+150,**params_nl)
lll = np.arange(1,lmaxcamb+1)
Tnl=Tnl[:lmaxcamb]
Enl=Enl[:lmaxcamb]
Bnl=Bnl[:lmaxcamb]
Xnl=Xnl[:lmaxcamb]
fact = (lll*(lll+1))/(2*np.pi)
spectra_nl = [lll, Tnl/fact, Enl/fact, Bnl/fact, Xnl/fact]

clf()
mp.plot(lll, np.sqrt(spectra[1]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{TT}$')
mp.plot(lll, np.sqrt(abs(spectra[4])*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{TE}$')
mp.plot(lll,np.sqrt(spectra[2]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{EE}$')
mp.plot(lll,np.sqrt(spectra[3]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{BB}$')
mp.yscale('log')
mp.xlim(0,lmaxcamb+1)
#ylim(0.0001,100)
mp.xlabel('$\ell$')
mp.ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
mp.legend(loc='lower right',frameon=False)







########## Qubic Instrument
NET150 = (220+314)/2*sqrt(2)*sqrt(2)  ### Average between witner and summer
inst = QubicInstrument()

### Binning
deltal=100.
lmin=25
ellbins = np.array([25, 45, 65, 95, 140, 200, 300])
ellmin = ellbins[:len(ellbins)-1]
ellmax = ellbins[1:len(ellbins)]


### Get baselines and errors
epsilon = 0.7 * 0.7  #focal plane integration + optical efficiency)

clf()
subplot(2,1,1)
xlim(-300,300)
ylim(-300,300)
xlabel('$u$')
ylabel('$v$')
subplot(2,1,2)
ylim(0,1)
ylabel('$N_{eq}(\ell)/N_h$')
xlabel('$\ell$')

#### initial
ellav, deltacl, noisevar, samplevar, neq_nh, nsig, baselines = qubic_sensitivity.give_qubic_errors(inst, ellbins, lll, spectra[3], nu=150e9, epsilon=epsilon,net_polar=NET150, plot_baselines=True, symplot='ro')

bs, ellbs, bs_unique, ellbs_unique, nbs_unique = baselines

lmax = 1200
dl = 40
ellmin = np.arange(lmax/dl)*dl
ellmax = np.arange(lmax/dl)*dl+dl
ellav = 0.5 * (ellmin + ellmax)

ntot = np.zeros(len(ellav))
for i in np.arange(len(ellmin)):
	msk = (ellbs_unique > ellmin[i]) & (ellbs_unique <= ellmax[i])
	ntot[i] = np.sqrt( np.sum( nbs_unique[msk]**2 ))

nh = len(inst.horn)

clf()
plot(ellav,ntot/nh)
plot(ellav, neq_nh(ellav))











####### now try a continuous method
bs, ellbs, bs_unique, ellbs_unique, nbs_unique = qubic_sensitivity.give_baselines(inst, doplot=True)

nx = 512
ny = 512
uvsigma = 1./(2*np.pi*inst.primary_beam.sigma_rad)
uvcov = np.zeros((nx,ny))
xx = linspace(-200,200, nx)
yy = xx.copy()
x2d, y2d = np.meshgrid(xx,yy)

for i in xrange(len(bs)):
	if i % 100 == 0:
		print(i)
	uvcov += np.nan_to_num(np.exp(-((x2d-bs[i,0])**2 + (y2d-bs[i,1])**2) / (2 * uvsigma**2)))

symcov = (uvcov+uvcov[::-1,::-1])/2
clf()
imshow(symcov, interpolation = 'nearest', extent=[-200,200,-200,200])
colorbar()

ll2d = 2 * np.pi * np.sqrt(x2d**2 + y2d**2)

clf()
plot(ll2d, symcov,',')

clf()
hist(np.ravel(ll2d), range=[0,1500], bins=300, weights = np.ravel(symcov))












