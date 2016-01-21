from __future__ import division
from pylab import *
import healpy as hp
from matplotlib.pyplot import *
import numpy as np
import os
import sys
import pycamb
import string
import random


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

nside=2048

maps = hp.synfast(spectra[1:],nside,fwhm=np.radians(30./60),pixwin=True,new=True)
hp.mollview(maps[1],title='Q map smoothed at 30 arcmin',min=-4,max=4,unit='$\mu K$')
#savefig('Qmap30arcmin.png')
hp.mollview(maps[2],title='U map smoothed at 30 arcmin',min=-4,max=4,unit='$\mu K$')
#savefig('Umap30arcmin.png')

maps = hp.synfast(spectra[1:],nside,fwhm=np.radians(0),pixwin=True,new=True)
hp.mollview(maps[1],title='Q map not smoothed',min=-15,max=15,unit='$\mu K$')
#savefig('Qmap0arcmin.png')
hp.mollview(maps[2],title='U map not smoothed',min=-15,max=15,unit='$\mu K$')
#savefig('Umap0arcmin.png')

maps = hp.synfast(spectra[1:],nside,fwhm=np.radians(1),pixwin=True,new=True)
hp.mollview(maps[1],title='Q map smoothed at 1 degree',unit='$\mu K$')
#savefig('Qmap60arcmin.png')
hp.mollview(maps[2],title='U map smoothed at 1 degree',unit='$\mu K$')
#savefig('Umap60arcmin.png')




######## Systematics
eps = logspace(-2,0,200)
rho = logspace(-2,0,100)
leakage = np.zeros((len(eps), len(rho)))
for i in np.arange(len(eps)):
	for j in np.arange(len(rho)):
		leakage[i,j] = 4*eps[i]**2*(1+rho[j])**2

clf()
imshow(np.log10(leakage.T), origin='upper left', interpolation = 'nearest', extent = [np.log10(eps[0]), 0, np.log10(rho[0]), 0],vmin=-4,vmax=0)
xlabel('$Log(\epsilon)$')
ylabel(r'$Log(\rho)$')
title(r'$Log(4\epsilon^2(1+\rho)^2)$')
colorbar()
savefig('leakage.png')






