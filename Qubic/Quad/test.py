from __future__ import division

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp

from pyquad import pyquad

### read input spectrum
#a=np.loadtxt('/Volumes/Data/Qubic/qubic_v1/cl_r=0.1bis2.txt')
a=np.loadtxt('./cl_r=0.1bis2.txt')
ell=a[:,0]
ctt=np.concatenate([[0,0],a[:,1]*1e12*2*np.pi/(ell*(ell+1))])
cee=np.concatenate([[0,0],a[:,2]*1e12*2*np.pi/(ell*(ell+1))])
cte=np.concatenate([[0,0],a[:,4]*1e12*2*np.pi/(ell*(ell+1))])
cbb=np.concatenate([[0,0],a[:,7]*1e12*2*np.pi/(ell*(ell+1))])
ell=np.concatenate([[0,1],ell])

#wtozero = (ell < 11) | (ell > 30)
wtozero = (ell < 0) | (ell > 30000)
ctt[wtozero]=0
cee[wtozero]=0
cte[wtozero]=0
cbb[wtozero]=0

spectra=[ell,ctt,cte,cee,cbb]

clf()
plot(ell,np.sqrt(spectra[1]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TT}$')
plot(ell,np.sqrt(abs(spectra[2])*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TE}$')
plot(ell,np.sqrt(spectra[3]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{EE}$')
plot(ell,np.sqrt(spectra[4]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{BB} (r=0.1)$')
yscale('log')
xlim(0,250)
ylim(0.01,100)
xlabel('$\ell$')
ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
legend(loc='lower right',frameon=False)


#### Now start the Quadratic Estimator

# define ell bins
minell=35
deltal=25
ellbins=16
maxell=minell+deltal*ellbins

# binned input spectrum
ells=np.linspace(minell,maxell,ellbins+1)
ellmin=ells[0:ellbins]
ellmin[0]=2
ellmax=ells[1:ellbins+1]-1
ellval=(ellmin+ellmax)/2
deltaell=(ellmax+1-ellmin)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)
inputspectrum=binspec


####################### Quadratic estimator
clf()
xlim(0,np.max(maxell))
plot(ell,spectra[4]*(ell*(ell+1))/(2*np.pi),lw=3)
errorbar(ellval,inputspectrum[:,4]*ellval*(ellval+1)/(2*np.pi),xerr=deltaell,fmt='ro')

nside=256
npix=4000
fwhm=30./60*np.pi/180

# calculate ds_dcb
map_orig=hp.synfast(spectra[4],nside,fwhm=fwhm,pixwin=True)
mapin=map_orig.copy()
mapin[npix:mapin.size]=0
mask= mapin != 0
#ds_dcb=pyquad.compute_ds_dcb(mapin,mask,ellmin,ellmax,fwhm)
ds_dcb=pyquad.compute_ds_dcb_line_par(mapin,mask,ellmin,ellmax,fwhm,24)

# Now iterate
nbmc=1
signoise=0.01
guess=inputspectrum[:,4]*0.5
allcl=np.zeros((ellbins,nbmc))
alldcl=np.zeros((ellbins,nbmc))
for i in arange(nbmc):
    print(i)
    map_orig=hp.synfast(spectra[4],nside,fwhm=fwhm,pixwin=True)
    #map_orig=map_orig[0]
    mapin=map_orig.copy()
    mapin[npix:mapin.size]=0
    mask= mapin != 0
    map=mapin.copy()
    map[mask]+=np.random.randn(npix)*signoise
    map[mask]=map[mask]-mean(map[mask])
    covmap=np.diag(np.zeros(npix)+signoise**2)
    thespectrum,err,invfisher,lk,num=pyquad.quadest(map,mask,covmap,ellmin,ellmax,fwhm,guess,spectra[4],ds_dcb,itmax=20,plot=True,cholesky=True)
    allcl[:,i]=thespectrum[:,num]
    alldcl[:,i]=err


clmean=np.zeros(ellbins)
dclmean=np.zeros(ellbins)
dclmean2=np.zeros(ellbins)
clmeancut=np.zeros(ellbins)
dclmeancut=np.zeros(ellbins)
for i in arange(ellbins):
    clmean[i]=np.mean(allcl[i,:])
    dclmean[i]=np.std(allcl[i,:])
    dclmean2[i]=np.std(allcl[i,:])/sqrt(nbmc)
    clmeancut[i],dclmeancut[i]=pyquad.meancut(allcl[i,:],2,5)

clf()
plot(ell,spectra[4]*(ell*(ell+1))/(2*np.pi),lw=3)
ylim(-0.01,0.03)
errorbar(ellval,clmeancut*ellval*(ellval+1)/(2*np.pi),dclmeancut*ellval*(ellval+1)/(2*np.pi),xerr=deltaell,label=str(i),fmt='o')
xlim(0,maxell*1.3)
errorbar(ellval,clmeancut*ellval*(ellval+1)/(2*np.pi),dclmeancut*ellval*(ellval+1)/(2*np.pi)/sqrt(nbmc),xerr=deltaell,label=str(i),fmt='ro')
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$'+'    '+'$[\mu K^2]$ ')
for i in arange(nbmc):
    plot(ellval,allcl[:,i]*ellval*(ellval+1)/(2*np.pi),alpha=0.2)


clf()
errorbar(ellval,clmean*ellval*(ellval+1)/(2*np.pi),dclmean*ellval*(ellval+1)/(2*np.pi),xerr=deltaell,label=str(i),fmt='o')
errorbar(ellval,clmean*ellval*(ellval+1)/(2*np.pi),dclmean2*ellval*(ellval+1)/(2*np.pi),xerr=deltaell,label=str(i),fmt='o')
plot(ell,spectra[4]*(ell*(ell+1))/(2*np.pi),lw=3)
xlim(0,np.max(maxell))
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$'+'    '+'$[\mu K^2]$ ')




#debug
from numpy import *
from pylab import *
clf()
imshow(matsky,interpolation='nearest')
colorbar()

import pickle
out=open('data.pkl', 'wb')
pickle.dump(matsky,out)
pickle.dump(matcov,out)
out.close()

clf()
imshow(ds_dcb[0,:,:])
colorbar()








