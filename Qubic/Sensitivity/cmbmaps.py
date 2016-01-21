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


spec4synfast = [T, E, B, X]
nside=512
mapi,mapq,mapu=hp.synfast(spectra[1:],nside,pixwin=True,new=True, fwhm=np.radians(1))
hp.mollview(mapi)
savefig('mapi.png')
hp.mollview(mapq)
savefig('mapq.png')
hp.mollview(mapu)
savefig('mapu.png')





