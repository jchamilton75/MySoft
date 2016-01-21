from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import sin, cos, pi
from pyoperators import Rotation3dOperator
from pysimulators import FitsArray
from qubic import (
    QubicInstrument, create_random_pointings, equ2gal, gal2equ, map2tod,
    tod2map_all, tod2map_each)
from qubic.utils import progress_bar
import pycamb
from XPol import XPol



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
         'tensor_ratio':0.,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False}

lmaxcamb = 3*512
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
lll = np.arange(1,lmaxcamb+1)
fact = (lll*(lll+1))/(2*np.pi)
spectra = [lll, T/fact, E/fact, B/fact, X/fact]
clf()
plot(lll,np.sqrt(spectra[1]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{TT}$')
plot(lll,np.sqrt(abs(spectra[4])*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{TE}$')
plot(lll,np.sqrt(spectra[2]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{EE}$')
plot(lll,np.sqrt(spectra[3]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{BB}$')
yscale('log')
xlim(0,lmaxcamb+1)
#ylim(0.0001,100)
xlabel('$\ell$')
ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
legend(loc='lower right',frameon=False)








#############################################################################
nside = 256
lmax = 2*nside-1
ell = np.arange(lmax+1)

#### Mask
racenter = 0.0
deccenter = -57.0
maxang = 20.
center = equ2gal(racenter, deccenter)

nsmaskinit = nside

veccenter = hp.ang2vec(pi/2-np.radians(center[1]), np.radians(center[0]))
vecpix = hp.pix2vec(nsmaskinit,np.arange(12*nsmaskinit**2))
cosang = np.dot(veccenter,vecpix)
maskok = np.degrees(np.arccos(cosang)) < maxang

### Make Mask Map
mapang = XPol.map_ang_from_edges(maskok)
maskmap = XPol.apodize_mask(maskok, 2, mapang=mapang)
#hp.gnomview(maskmap,rot=[racenter,deccenter],coord=['G','C'],reso=15)


wl = hp.anafast(maskmap,regression=False)
wl = wl[0:lmax+1]


maps = hp.synfast(spectra[1:],nside,fwhm=0,pixwin=True,new=True)
cls, newl, Mll, MllBinned, MllBinnedInv, p, q, pseudocls = XPol.get_spectra(maps, maskmap, 2*nside-1, 20, 20)
nbins = len(newl)

nbmc = 100
allclsout = np.zeros((nbmc, 6, nbins))
allcls = np.zeros((nbmc, 6, lmax+1))
for i in np.arange(nbmc):
	print(i)
	maps = hp.synfast(spectra[1:],nside,fwhm=0,pixwin=True,new=True)
	res = XPol.get_spectra(maps, maskmap, 2*nside-1, 20, 20, Mllmat=Mll,MllBinnedInv=MllBinnedInv, ellpq=[newl, p,q])
	allclsout[i,:,:] = res[0]
	allcls[i,:,:] = np.array(res[7])

#### Get MC results
mclsout = np.zeros((6, nbins))
sclsout = np.zeros((6, nbins))
for i in np.arange(6):
	for j in np.arange(nbins):
		mclsout[i,j] = np.mean(allclsout[:,i,j])
		sclsout[i,j] = np.std(allclsout[:,i,j])/sqrt(nbmc)

mcls = np.zeros((6, lmax+1))
scls = np.zeros((6, lmax+1))
for i in np.arange(6):
	for j in np.arange(lmax+1):
		mcls[i,j] = np.mean(allcls[:,i,j])
		scls[i,j] = np.std(allcls[:,i,j])/sqrt(nbmc)


pw = hp.pixwin(nside, pol=True)
pw = [pw[0][0:lmax+1], pw[1][0:lmax+1]]
pwb = [np.interp(newl,ell,pw[0]), np.interp(newl,ell,pw[1])]
fact = ell * (ell + 1) / (2 * np.pi) * len(maskmap) / maskok.sum()
factth = lll * (lll + 1) / (2 * np.pi)


clf()
title('BB')
xlim(0,lmax)
errorbar(newl,mclsout[2,:]/pwb[1]**2,yerr=sclsout[2,:],fmt='bo')
plot(ell,fact*mcls[2,:]/pw[1]**2,'g')
plot(spectra[0],spectra[3]*factth,'r')


xlmax=lmax
clf()
subplot(3,2,1)
title('TT')
xlim(0,xlmax)
errorbar(newl,mclsout[0,:]/pwb[0]**2,yerr=sclsout[0,:],fmt='bo',label='XPol')
plot(ell,fact*mcls[0,:]/pw[0]**2,'g',label='Anafast rescaled')
plot(spectra[0],spectra[1]*factth,'r',label='Input')
legend(loc='lower right',frameon=False,fontsize=10)
subplot(3,2,2)
title('TE')
xlim(0,xlmax)
errorbar(newl,mclsout[3,:]/pwb[0]/pwb[1],yerr=sclsout[3,:],fmt='bo')
plot(ell,fact*mcls[3,:]/pw[0]/pw[1],'g')
plot(spectra[0],spectra[4]*factth,'r')
subplot(3,2,3)
title('EE')
xlim(0,xlmax)
ylim(0,2)
errorbar(newl,mclsout[1,:]/pwb[1]**2,yerr=sclsout[1,:],fmt='bo')
plot(ell,fact*mcls[1,:]/pw[1]**2,'g')
plot(spectra[0],spectra[2]*factth,'r')
subplot(3,2,4)
title('BB')
xlim(0,xlmax)
ylim(-0.005,0.01)
errorbar(newl,mclsout[2,:]/pwb[1]**2,yerr=sclsout[2,:],fmt='bo')
plot(ell,fact*mcls[2,:]/pw[1]**2,'g')
plot(spectra[0],spectra[3]*factth,'r')
subplot(3,2,5)
title('TB')
xlim(0,xlmax)
errorbar(newl,mclsout[4,:]/pwb[1]**2,yerr=sclsout[4,:],fmt='bo')
plot(ell,fact*mcls[4,:]/pw[1]**2,'g')
subplot(3,2,6)
title('EB')
xlim(0,xlmax)
errorbar(newl,mclsout[5,:]/pwb[0]/pwb[1],yerr=sclsout[5,:],fmt='bo')
plot(ell,fact*mcls[5,:]/pw[0]/pw[1],'g')



#### Residuals
xlmax=lmax
clf()
subplot(3,2,1)
title('TT')
xlim(0,xlmax)
errorbar(newl,mclsout[0,:]/pwb[0]**2 - np.dot(p,spectra[1][0:lmax+1]),yerr=sclsout[0,:],fmt='bo',label='XPol')
plot(ell,ell*0,'k--')

subplot(3,2,2)
title('TE')
xlim(0,xlmax)
errorbar(newl,mclsout[3,:]/pwb[0]/pwb[1] - np.dot(p,spectra[4][0:lmax+1]),yerr=sclsout[3,:],fmt='bo',label='XPol')
plot(ell,ell*0,'k--')

subplot(3,2,3)
title('EE')
xlim(0,xlmax)
ylim(-0.005,0.1)
errorbar(newl,mclsout[1,:]/pwb[1]**2 - np.dot(p,spectra[2][0:lmax+1]),yerr=sclsout[1,:],fmt='bo',label='XPol')
plot(ell,ell*0,'k--')
plot(spectra[0],spectra[2]*factth,'r')

subplot(3,2,4)
title('BB')
xlim(0,xlmax)
errorbar(newl,mclsout[2,:]/pwb[1]**2 - np.dot(p,spectra[3][0:lmax+1]),yerr=sclsout[2,:],fmt='bo')
plot(ell,ell*0,'k--')
plot(ell,fact*mcls[2,:]/pw[1]**2,'g')
plot(spectra[0],spectra[3]*factth,'r')
ylim(-3*np.max(sclsout[2,:]), 15*np.max(sclsout[2,:]))


subplot(3,2,5)
title('TB')
xlim(0,xlmax)
errorbar(newl,mclsout[4,:]/pwb[1]**2,yerr=sclsout[4,:],fmt='bo')
plot(ell,ell*0,'k--')

subplot(3,2,6)
title('EB')
xlim(0,xlmax)
errorbar(newl,mclsout[5,:]/pwb[0]/pwb[1],yerr=sclsout[5,:],fmt='bo')
plot(ell,ell*0,'k--')




