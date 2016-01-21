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
from pysimulators import FitsArray
from Cosmo import interpol_camb as ic

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


### Camblib
ellcamblib = FitsArray('/Users/hamilton/CMB/Interfero/DualBand/camblib600_ell.fits')
rcamblib = FitsArray('/Users/hamilton/CMB/Interfero/DualBand/camblib600_r.fits')
clcamblib = FitsArray('/Users/hamilton/CMB/Interfero/DualBand/camblib600_cl.fits')
camblib = [ellcamblib, rcamblib, clcamblib]

#### models
def get_dl(rvalue, lmaxcamb=1000):
    params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':rvalue,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}
    T,E,B,X = pycamb.camb(lmaxcamb+1+150,**params)
    params_nl = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':rvalue,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False}
    Tnl,Enl,Bnl,Xnl = pycamb.camb(lmaxcamb+1+150,**params_nl)
    lll = np.arange(1,lmaxcamb+1)
    return lll,T[:lmaxcamb],E[:lmaxcamb],B[:lmaxcamb],X[:lmaxcamb], Tnl[:lmaxcamb], Enl[:lmaxcamb], Bnl[:lmaxcamb], Xnl[:lmaxcamb]

clth_0_1 = get_dl(0.1, lmaxcamb=2000)
clth_0_2 = get_dl(0.2, lmaxcamb=2000)
clth_0_05 = get_dl(0.05, lmaxcamb=2000)
clth_0_01 = get_dl(0.01, lmaxcamb=2000)


### uperlimits
uplims = np.loadtxt('/Users/hamilton/CMB/Interfero/PlotsCMB/upperlimits.txt', 
    dtype=[('name', 'S10'), ('lmin', float), ('lmax', float), ('updl', float)], skiprows=3)
experiments = unique(uplims['name'])


col = plt.get_cmap('jet')(np.linspace(0, 1.0, len(experiments)))
clf()
yscale('log')
xscale('log')
i=0
for exp in experiments:
    thedata = uplims[uplims['name']==exp]
    lcenters = (thedata['lmin']+thedata['lmax'])/2
    thexerr = (thedata['lmax']-thedata['lmin'])/2
    errorbar(lcenters, thedata['updl'], xerr=thexerr,fmt=',', label=exp, color=col[i], lw=2)
    i=i+1

dl_lensing = clth_0_1[3]-clth_0_1[7]
plot(clth_0_1[0], dl_lensing, 'k--')
plot(clth_0_1[0], clth_0_1[3],'b', label='r = 0.1')
plot(clth_0_1[0], clth_0_1[3]-dl_lensing,'b:')
plot(clth_0_01[0], clth_0_01[3],'r', label='r = 0.01')
plot(clth_0_01[0], clth_0_01[3]-dl_lensing,'r:')
legend(loc='upper left',fontsize=8)
xlabel('$\ell$')
ylabel('$D_\ell^{BB} [\mu K^2]$')
ylim(1e-4, 1e3)
xlim(1,3000)
savefig('upperlimits.png')

### Measurements
meas = np.loadtxt('/Users/hamilton/CMB/Interfero/PlotsCMB/detections.txt', 
    dtype=[('name', 'S30'), ('lmin', float), ('lcenter', float), ('lmax', float), ('BB', float), ('errBB', float)], skiprows=3)
experiments = unique(meas['name'])
experiments = experiments[experiments != 'BICEP2']

col = plt.get_cmap('jet')(np.linspace(0, 1.0, len(experiments)))
clf()
yscale('log')
xscale('log')
i=0
for exp in experiments:
    thedata = meas[meas['name']==exp]
    lcenters = thedata['lcenter']
    thexerr = (thedata['lmax']-thedata['lmin'])/2
    ups = thedata['BB']<=0
    errorbar(lcenters[ups], np.abs(thedata['errBB'][ups])*2, xerr=thexerr[ups], yerr=np.abs(thedata['errBB'][ups])/2, fmt=',', color=col[i], lw=2, uplims = True)
    ok = thedata['BB']>0
    errorbar(lcenters[ok], thedata['BB'][ok], xerr=thexerr[ok], yerr=thedata['errBB'][ok],fmt=',', label=exp, color=col[i], lw=2, uplims = False)
    i=i+1

dl_lensing = clth_0_1[3]-clth_0_1[7]
plot(clth_0_1[0], dl_lensing, 'k--', label='BB Lensing')
plot(clth_0_1[0], clth_0_1[3],'b', label='r = 0.1')
plot(clth_0_1[0], clth_0_1[3]-dl_lensing,'b:')
plot(clth_0_01[0], clth_0_01[3],'r', label='r = 0.01')
plot(clth_0_01[0], clth_0_01[3]-dl_lensing,'r:')
legend(loc='upper left',fontsize=12)
xlabel('$\ell$')
ylabel('$D_\ell^{BB} [\mu K^2]$')
ylim(1e-4, 1)
xlim(1,3000)
savefig('measurements.png')













