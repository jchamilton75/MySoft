from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import pi
from qubic import equ2gal
from qubic.xpol import XPol
from qubic.mapmaking import apodize_mask
from qubic.utils import progress_bar
import pycamb




#### Get input Power spectra
################# Input Power spectrum ###################################
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
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
mp.figure()
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




#############################################################################
nside = 256
lmin = 20
lmax = 2*nside-1
delta_ell = 20
ell = np.arange(lmax+1)

#### Mask
racenter = 0.0
deccenter = -57.0
maxang = 20.
center = equ2gal(racenter, deccenter)

nsmaskinit = nside

veccenter = hp.ang2vec(pi/2-np.radians(center[1]), np.radians(center[0]))
vecpix = hp.pix2vec(nsmaskinit, np.arange(12*nsmaskinit**2))
cosang = np.dot(veccenter, vecpix)
maskok = np.degrees(np.arccos(cosang)) < maxang

### Make Mask Map
maskmap = apodize_mask(maskok, 2*2.35)
#hp.gnomview(maskmap,rot=[racenter,deccenter],coord=['G','C'],reso=15)


xpol = XPol(maskmap, lmin, lmax, delta_ell)
newl = xpol.ell_binned
nbins = len(newl)

nbmc = 100
allclsout = np.zeros((nbmc, 6, nbins))
allcls = np.zeros((nbmc, 6, lmax+1))
bar = progress_bar(nbmc)
for i in np.arange(nbmc):
    maps = hp.synfast(spectra[1:], nside, fwhm=0, pixwin=True, new=True,
                      verbose=False)
    allcls[i], allclsout[i] = xpol.get_spectra(maps)
    bar.update()

#### Get MC results
mclsout = np.mean(allclsout, axis=0)
sclsout = np.std(allclsout, axis=0)
mcls = np.mean(allcls, axis=0)
scls = np.std(allcls, axis=0)

pw = hp.pixwin(nside, pol=True)
pw = [pw[0][0:lmax+1], pw[1][0:lmax+1]]
pwb = [np.interp(newl,ell,pw[0]), np.interp(newl,ell,pw[1])]
fact = ell * (ell + 1) / (2 * np.pi) * len(maskmap) / maskok.sum()
factth = lll * (lll + 1) / (2 * np.pi)


mp.figure()
mp.title('BB')
mp.xlim(0,lmax)
mp.errorbar(newl,mclsout[2,:]/pwb[1]**2,yerr=sclsout[2,:],fmt='bo')
mp.plot(ell,fact*mcls[2,:]/pw[1]**2,'g')
mp.plot(spectra[0],spectra[3]*factth,'r')


xlmax=lmax
mp.figure()
mp.subplot(3,2,1)
mp.title('TT')
mp.xlim(0,xlmax)
mp.errorbar(newl,mclsout[0,:]/pwb[0]**2,yerr=sclsout[0,:],fmt='bo',label='XPol')
mp.plot(ell,fact*mcls[0,:]/pw[0]**2,'g',label='Anafast rescaled')
mp.plot(spectra[0],spectra[1]*factth,'r',label='Input')
mp.legend(loc='lower right',frameon=False,fontsize=10)
mp.subplot(3,2,2)
mp.title('TE')
mp.xlim(0,xlmax)
mp.errorbar(newl,mclsout[3,:]/pwb[0]/pwb[1],yerr=sclsout[3,:],fmt='bo')
mp.plot(ell,fact*mcls[3,:]/pw[0]/pw[1],'g')
mp.plot(spectra[0],spectra[4]*factth,'r')
mp.subplot(3,2,3)
mp.title('EE')
mp.xlim(0,xlmax)
mp.ylim(0,2)
mp.errorbar(newl,mclsout[1,:]/pwb[1]**2,yerr=sclsout[1,:],fmt='bo')
mp.plot(ell,fact*mcls[1,:]/pw[1]**2,'g')
mp.plot(spectra[0],spectra[2]*factth,'r')
mp.subplot(3,2,4)
mp.title('BB')
mp.xlim(0,xlmax)
mp.ylim(-0.005,0.01)
mp.errorbar(newl,mclsout[2,:]/pwb[1]**2,yerr=sclsout[2,:],fmt='bo')
mp.plot(ell,fact*mcls[2,:]/pw[1]**2,'g')
mp.plot(spectra[0],spectra[3]*factth,'r')
mp.subplot(3,2,5)
mp.title('TB')
mp.xlim(0,xlmax)
mp.errorbar(newl,mclsout[4,:]/pwb[1]**2,yerr=sclsout[4,:],fmt='bo')
mp.plot(ell,fact*mcls[4,:]/pw[1]**2,'g')
mp.subplot(3,2,6)
mp.title('EB')
mp.xlim(0,xlmax)
mp.errorbar(newl,mclsout[5,:]/pwb[0]/pwb[1],yerr=sclsout[5,:],fmt='bo')
mp.plot(ell,fact*mcls[5,:]/pw[0]/pw[1],'g')



#### Residuals
xlmax=lmax
mp.figure()
mp.subplot(3,2,1)
mp.title('TT')
mp.xlim(0,xlmax)
mp.errorbar(newl,mclsout[0,:]/pwb[0]**2 - np.dot(xpol.p,spectra[1][0:lmax+1]),yerr=sclsout[0,:],fmt='bo',label='XPol')
mp.plot(ell,ell*0,'k--')

mp.subplot(3,2,2)
mp.title('TE')
mp.xlim(0,xlmax)
mp.errorbar(newl,mclsout[3,:]/pwb[0]/pwb[1] - np.dot(xpol.p,spectra[4][0:lmax+1]),yerr=sclsout[3,:],fmt='bo',label='XPol')
mp.plot(ell,ell*0,'k--')

mp.subplot(3,2,3)
mp.title('EE')
mp.xlim(0,xlmax)
mp.ylim(-0.005,0.1)
mp.errorbar(newl,mclsout[1,:]/pwb[1]**2 - np.dot(xpol.p,spectra[2][0:lmax+1]),yerr=sclsout[1,:],fmt='bo',label='XPol')
mp.plot(ell,ell*0,'k--')
mp.plot(spectra[0],spectra[2]*factth,'r')

mp.subplot(3,2,4)
mp.title('BB')
mp.xlim(0,xlmax)
mp.errorbar(newl,mclsout[2,:]/pwb[1]**2 - np.dot(xpol.p,spectra[3][0:lmax+1]),yerr=sclsout[2,:],fmt='bo')
mp.plot(ell,ell*0,'k--')
mp.plot(ell,fact*mcls[2,:]/pw[1]**2,'g')
mp.plot(spectra[0],spectra[3]*factth,'r')
mp.ylim(-3*np.max(sclsout[2,:]), 15*np.max(sclsout[2,:]))


mp.subplot(3,2,5)
mp.title('TB')
mp.xlim(0,xlmax)
mp.errorbar(newl,mclsout[4,:]/pwb[1]**2,yerr=sclsout[4,:],fmt='bo')
mp.plot(ell,ell*0,'k--')

mp.subplot(3,2,6)
mp.title('EB')
mp.xlim(0,xlmax)
mp.errorbar(newl,mclsout[5,:]/pwb[0]/pwb[1],yerr=sclsout[5,:],fmt='bo')
mp.plot(ell,ell*0,'k--')
