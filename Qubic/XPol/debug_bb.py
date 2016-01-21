from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import pycamb



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
         'tensor_ratio':0.1,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False}

lmaxcamb = 3*512
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
lll = np.arange(1,lmaxcamb+1)
fact = (lll*(lll+1))/(2*np.pi)
spectra01 = [lll, T/fact, E/fact, B/fact, X/fact]

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
ylim(0.0001,100)
xlabel('$\ell$')
ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
legend(loc='lower right',frameon=False)


#############################################################################
nside = 128
lmax = 2*nside-1
maps=hp.synfast(spectra[1:],nside,fwhm=0,pixwin=True,new=True)
pw = hp.pixwin(nside, pol=True)

lmax = 1*nside-1
cls1 = hp.anafast(maps,pol=True, lmax=lmax)
ell1= np.arange(lmax+1)
pw1 = [pw[0][0:lmax+1], pw[1][0:lmax+1]]

lmax = 2*nside-1
cls2 = hp.anafast(maps,pol=True, lmax=lmax)
ell2= np.arange(lmax+1)
pw2 = [pw[0][0:lmax+1], pw[1][0:lmax+1]]

lmax = 3*nside-1
cls3 = hp.anafast(maps,pol=True, lmax=lmax)
ell3= np.arange(lmax+1)
pw3 = [pw[0][0:lmax+1], pw[1][0:lmax+1]]


clf()
yscale('log')
ylabel('$\ell(\ell+1)C_\ell/2\pi$')
xlabel('$\ell$')
xlim(0,3*nside)
ylim(1e-13,1e3)
plot(ell1,ell1*(ell1+1)/2/np.pi*cls1[2]/pw1[1]**2,label='$\ell_{max}= n_s-1$')
plot(ell2,ell2*(ell2+1)/2/np.pi*cls2[2]/pw2[1]**2,label='$\ell_{max}= 2n_s-1$')
plot(ell3,ell3*(ell3+1)/2/np.pi*cls3[2]/pw3[1]**2,label='$\ell_{max}= 3n_s-1$')
plot(lll,lll*(lll+1)/2/np.pi*spectra01[3],label='BB $r=0.1$')
title('nside='+str(nside))
legend(loc='lower right')


clf()
yscale('log')
ylabel('$C_\ell$')
xlabel('$\ell$')
xlim(0,3*nside)
plot(ell1,cls1[2]/pw1[1]**2,label='$\ell_{max}= n_s-1$')
plot(ell2,cls2[2]/pw2[1]**2,label='$\ell_{max}= 2n_s-1$')
plot(ell3,cls3[2]/pw3[1]**2,label='$\ell_{max}= 3n_s-1$')
plot(lll,spectra01[3],label='$BB r=0.1$')
legend(loc='lower right')
title('nside='+str(nside))
legend()
savefig('Bresiduals_r0.png')


