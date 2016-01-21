from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
from Quad import qml
from Quad import pyquad
import healpy as hp



#### Get input Power spectra
#a=np.loadtxt('/Volumes/Data/Qubic/qubic_v1/cl_r=0.1bis2.txt')
a=np.loadtxt('./cl_r=0.1bis2.txt')
ell=a[:,0]
ctt=np.concatenate([[0,0],a[:,1]*1e12*2*np.pi/(ell*(ell+1))])
cee=np.concatenate([[0,0],a[:,2]*1e12*2*np.pi/(ell*(ell+1))])
cte=np.concatenate([[0,0],a[:,4]*1e12*2*np.pi/(ell*(ell+1))])
cbb=np.concatenate([[0,0],a[:,7]*1e12*2*np.pi/(ell*(ell+1))])
ell=np.concatenate([[0,1],ell])
spectra=[ell,ctt,cee,cbb,cte]   #### This is the new ordering

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



nside=128
lmax=3*nside
ell=spectra[0]
maskl=ell<(lmax+1)
bl=np.ones(maskl.size)

fwhmrad=0.5*np.pi/180
mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
hp.mollview(mapi)
hp.mollview(mapq)
hp.mollview(mapu)
maps=[mapi,mapq,mapu]


############## T only
nbpixok=100
mask=(np.arange(12*nside**2) >= nbpixok)
maskok=~mask
ip=np.arange(12*nside**2)
ipok=ip[~mask]

ellbins=[0,75,100]
nbins=len(ellbins)-1
ellmin=np.array(ellbins[0:nbins])
ellmax=np.array(ellbins[1:nbins+1])-1
ellval=(ellmin+ellmax)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)

#### Old
reload(pyquad)
fwhm=fwhmrad
#ds_dcb_old=pyquad.compute_ds_dcb_line_par(mapi,maskok,ellmin,ellmax,fwhm,8)
ds_dcb_old=pyquad.compute_ds_dcb(mapi,maskok,ellmin,ellmax,fwhm)



#### New
ll=np.arange(int(np.max(ellbins))+1)
bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)
ds_dcb=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=False,temp=True,EBTB=False)
ds_dcb=ds_dcb[0,:,:,:]


############## Polar
nbpixok=50
mask=(np.arange(12*nside**2) >= nbpixok)
maskok=~mask
ip=np.arange(12*nside**2)
ipok=ip[~mask]

ellbins=[0,75,100]
nbins=len(ellbins)-1
ellmin=np.array(ellbins[0:nbins])
ellmax=np.array(ellbins[1:nbins+1])-1
ellval=(ellmin+ellmax)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)


#### New
ll=np.arange(int(np.max(ellbins))+1)
bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)
ds_dcb=qml.compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=True,temp=False,EBTB=False)
ds_dcb=ds_dcb[0,:,:,:]


