from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import pi
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
         'tensor_ratio':0.1,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}

lmaxcamb = 3*256
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
lll = np.arange(1,lmaxcamb+1)
fact = (lll*(lll+1))/(2*np.pi)
spectra = [lll, T/fact, E/fact, B/fact, X/fact]
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


clf()
sm=1

########### Maps
nside = 256
spectra = [lll, T/fact, E/fact, B/fact, X/fact]
numpy.random.seed(1234)
maps = hp.synfast(spectra[1:], nside,fwhm=np.radians(sm))
hp.mollview(maps[1], sub=[2,3,1],title='Q Stokes r=0.1 (1 degree smoothing)')
hp.mollview(maps[2], sub=[2,3,4],title='U Stokes r=0.1 (1 degree smoothing)')


########### Maps
nside = 256
spectra_noB = [lll, T/fact, E/fact, 0*B/fact, X/fact]
numpy.random.seed(1234)
maps = hp.synfast(spectra_noB[1:], nside,fwhm=np.radians(sm))
hp.mollview(maps[1], sub=[2,3,2],title='Q Stokes with r=0.1 & B=0 (1 degree smoothing)')
hp.mollview(maps[2], sub=[2,3,5],title='U Stokes with r=0.1 & B=0 (1 degree smoothing)')



########### Maps
nside = 256
spectra_noE = [lll, T/fact, 0*E/fact, B/fact, 0*X/fact]
numpy.random.seed(1234)
maps = hp.synfast(spectra_noE[1:], nside,fwhm=np.radians(sm))
hp.mollview(maps[1], sub=[2,3,3],title='Q Stokes with r=0.1 & E=0 & TE=0 (1 degree smoothing)')
hp.mollview(maps[2], sub=[2,3,6],title='U Stokes with r=0.1 & E=0 & TE=0 (1 degree smoothing)')

savefig('QUforJim_modifyEB.png')

####################### Other way
clf()
sm=1
res=30
dolensing = True
########### Maps
nside = 256
params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':0.1,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':dolensing}
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
spectra_new = [lll, T/fact, E/fact, B/fact, X/fact]
numpy.random.seed(1234)
maps = hp.synfast(spectra_new[1:], nside,fwhm=np.radians(sm))
hp.mollview(maps[1], sub=[2,3,1],title='Q Stokes r=0.1 (1 degree smoothing)')
hp.mollview(maps[2], sub=[2,3,4],title='U Stokes r=0.1 (1 degree smoothing)')
#hp.gnomview(maps[1], reso=res, sub=[2,3,1],title='Q Stokes r=1 (1 degree smoothing)')
#hp.gnomview(maps[2], reso=res, sub=[2,3,4],title='U Stokes r=1 (1 degree smoothing)')

########### Maps
nside = 256
params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':0.,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':dolensing}
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
spectra_notensor = [lll, T/fact, E/fact, B/fact, X/fact]
numpy.random.seed(1234)
maps = hp.synfast(spectra_notensor[1:], nside,fwhm=np.radians(sm))
hp.mollview(maps[1], sub=[2,3,2],title='Q Stokes r=0. (1 degree smoothing)')
hp.mollview(maps[2], sub=[2,3,5],title='U Stokes r=0. (1 degree smoothing)')
#hp.gnomview(maps[1], reso=res, sub=[2,3,2],title='Q Stokes r=0. (1 degree smoothing)')
#hp.gnomview(maps[2], reso=res, sub=[2,3,5],title='U Stokes r=0. (1 degree smoothing)')


########### Maps
nside = 256
params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':100.,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':dolensing}
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
spectra_tensor = [lll, T/fact, E/fact, B/fact, X/fact]
numpy.random.seed(1234)
maps = hp.synfast(spectra_tensor[1:], nside,fwhm=np.radians(sm))
hp.mollview(maps[1], sub=[2,3,3],title='Q Stokes r=100. (1 degree smoothing)')
hp.mollview(maps[2], sub=[2,3,6],title='U Stokes r=100. (1 degree smoothing)')
#hp.gnomview(maps[1], reso=res, sub=[2,3,3],title='Q Stokes r=10. (1 degree smoothing)')
#hp.gnomview(maps[2], reso=res, sub=[2,3,6],title='U Stokes r=10. (1 degree smoothing)')

savefig('QUforJim_modify_r.png')


clf()
mp.plot(lll, np.sqrt(spectra_tensor[1]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{TT}$')
mp.plot(lll, np.sqrt(abs(spectra_tensor[4])*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{TE}$')
mp.plot(lll,np.sqrt(spectra_tensor[2]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{EE}$')
mp.plot(lll,np.sqrt(spectra_tensor[3]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{BB}$')
mp.yscale('log')
mp.xlim(0,lmaxcamb+1)
#ylim(0.0001,100)
mp.xlabel('$\ell$')
mp.ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
mp.legend(loc='lower right',frameon=False)






####################### Other way with TE=0
clf()
sm=1
res=30
dolensing = True
########### Maps
nside = 256
params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':0.1,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':dolensing}
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
spectra_new = [lll, T/fact, E/fact, B/fact, 0*X/fact]
numpy.random.seed(1234)
maps = hp.synfast(spectra_new[1:], nside,fwhm=np.radians(sm))
hp.mollview(maps[1], sub=[2,3,1],title='Q Stokes r=0.1 (1 degree smoothing)')
hp.mollview(maps[2], sub=[2,3,4],title='U Stokes r=0.1 (1 degree smoothing)')
#hp.gnomview(maps[1], reso=res, sub=[2,3,1],title='Q Stokes r=1 (1 degree smoothing)')
#hp.gnomview(maps[2], reso=res, sub=[2,3,4],title='U Stokes r=1 (1 degree smoothing)')

########### Maps
nside = 256
params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':0.,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':dolensing}
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
spectra_notensor = [lll, T/fact, E/fact, B/fact, 0*X/fact]
numpy.random.seed(1234)
maps = hp.synfast(spectra_notensor[1:], nside,fwhm=np.radians(sm))
hp.mollview(maps[1], sub=[2,3,2],title='Q Stokes r=0. (1 degree smoothing)')
hp.mollview(maps[2], sub=[2,3,5],title='U Stokes r=0. (1 degree smoothing)')
#hp.gnomview(maps[1], reso=res, sub=[2,3,2],title='Q Stokes r=0. (1 degree smoothing)')
#hp.gnomview(maps[2], reso=res, sub=[2,3,5],title='U Stokes r=0. (1 degree smoothing)')


########### Maps
nside = 256
params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':100.,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':dolensing}
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
spectra_tensor = [lll, T/fact, E/fact, B/fact, 0*X/fact]
numpy.random.seed(1234)
maps = hp.synfast(spectra_tensor[1:], nside,fwhm=np.radians(sm))
hp.mollview(maps[1], sub=[2,3,3],title='Q Stokes r=100. (1 degree smoothing)')
hp.mollview(maps[2], sub=[2,3,6],title='U Stokes r=100. (1 degree smoothing)')
#hp.gnomview(maps[1], reso=res, sub=[2,3,3],title='Q Stokes r=10. (1 degree smoothing)')
#hp.gnomview(maps[2], reso=res, sub=[2,3,6],title='U Stokes r=10. (1 degree smoothing)')

savefig('QUforJim_modify_r_TEzero.png')







