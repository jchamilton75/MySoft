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



nside=64
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
#ds_dcb=0
#nbpixok=1000
#mask=(np.arange(12*nside**2) >= nbpixok)
#maskok=~mask
#ip=np.arange(12*nside**2)
#ipok=ip[~mask]

#mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
#themap=mapi.copy()
#themap[mask]=0
#hp.mollview(themap,rot=[0,90])

#ellbins=[0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,3*nside]
#nbins=len(ellbins)-1
#ellmin=np.array(ellbins[0:nbins])
#ellmax=np.array(ellbins[1:nbins+1])-1
#ellval=(ellmin+ellmax)/2
#binspec=pyquad.binspectrum(spectra,ellmin,ellmax)

#firstguess=binspec[:,1]
#signoise=0.1
#themapnoise=themap.copy()
#themapnoise[~mask]+=np.random.randn(len(themapnoise[~mask]))*signoise
#hp.mollview(themapnoise,rot=[0,90])


#reload(qml)
#covmap=np.identity(len(ipok))*signoise**2
#guess=[0,binspec[:,1],0,0,0]
#specout,error,invfisher,lk,num,ds_dcb=qml.qml([themapnoise],mask,covmap,ellbins,fwhmrad,guess,ds_dcb,spectra,cholesky=False,temp=True,polar=False,plot=True)

#### npix=4000 sig=100 ellbins=[0,75,100,125,150] -> Convergence
#### npix=2000 sig=100 ellbins=[0,75,100,125,150] -> Convergence
#### npix=2000 sig=100 ellbins=[0,75,100,125,150,175,200,225,250,275,300,325,350,375] -> Convergence but noisy
#### npix=2000 sig=10 ellbins=[0,75,100,125,150,175,200,225,250,275,300,325,350,375] -> Convergenc
#### npix=1000 sig=10 ellbins=[0,75,100,125,150,175,200,225,250,275,300,325,350,375] -> Convergenc
#### npix=1000 sig=1 ellbins=[0,75,100,125,150,175,200,225,250,275,300,325,350,375] -> Convergenc mais dernier point foireux
#### npix=1000 sig=0.1 ellbins=[0,75,100,125,150,175,200,225,250,275,300,325,350,375] -> Convergenc mais dernier point foireux
#### quand le rapport signal/bruit du dernier bin est trop bon ca fait tout foirer


#### EB Only
ds_dcb=0
nbpixok=1000
mask=(np.arange(12*nside**2) >= nbpixok)
maskok=~mask
ip=np.arange(12*nside**2)
ipok=ip[~mask]

#ellbins=[0,50,100,150,200,250,300,350,3*nside]
ellbins = [0,50,100,150,200]
nbins=len(ellbins)-1
ellmin=np.array(ellbins[0:nbins])
ellmax=np.array(ellbins[1:nbins+1])-1
ellval=(ellmin+ellmax)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)

reload(qml)
ll=np.arange(int(np.max(ellbins))+1)
bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)
ds_dcb=qml.compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=True,temp=False)

signoise=0.05
mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
themapi=mapi.copy()
themapi[mask]=0
themapi[~mask]+=np.random.randn(len(themapi[~mask]))*signoise
themapi[~mask]-=np.mean(themapi[~mask])
themapq=mapq.copy()
themapq[mask]=0
themapq[~mask]+=np.random.randn(len(themapq[~mask]))*signoise
themapq[~mask]-=np.mean(themapq[~mask])
themapu=mapu.copy()
themapu[mask]=0
themapu[~mask]+=np.random.randn(len(themapu[~mask]))*signoise
themapu[~mask]-=np.mean(themapu[~mask])
themaps=[themapq,themapu]

hp.mollview(mapq,rot=[0,90],title='Q')
hp.mollview(mapu,rot=[0,90],title='U')
hp.gnomview(themaps[0],rot=[0,90],reso=10,title='Q')
hp.gnomview(themaps[1],rot=[0,90],reso=10,title='U')

covmap=np.identity(len(ipok)*len(themaps))*signoise**2
guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],0]
specout,error,invfisher,lk,num,ds_dcb,conv=qml.qml(themaps,mask,covmap,ellbins,fwhmrad,guess,ds_dcb,spectra,cholesky=True,temp=False,polar=True,plot=True,itmax=20)




#### TEB
ds_dcb=0
nbpixok=2000
mask=(np.arange(12*nside**2) >= nbpixok)
maskok=~mask
ip=np.arange(12*nside**2)
ipok=ip[~mask]

ellbins=[0,50,100,150,200,250,300,350,3*nside]
nbins=len(ellbins)-1
ellmin=np.array(ellbins[0:nbins])
ellmax=np.array(ellbins[1:nbins+1])-1
ellval=(ellmin+ellmax)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)

reload(qml)
ll=np.arange(int(np.max(ellbins))+1)
bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)
ds_dcb=qml.compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=True,temp=True)

signoise=0.05
mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
themapi=mapi.copy()
themapi[mask]=0
themapi[~mask]+=np.random.randn(len(themapi[~mask]))*signoise
themapi[~mask]-=np.mean(themapi[~mask])
themapq=mapq.copy()
themapq[mask]=0
themapq[~mask]+=np.random.randn(len(themapq[~mask]))*signoise
themapq[~mask]-=np.mean(themapq[~mask])
themapu=mapu.copy()
themapu[mask]=0
themapu[~mask]+=np.random.randn(len(themapu[~mask]))*signoise
themapu[~mask]-=np.mean(themapu[~mask])
themaps=[themapi,themapq,themapu]


covmap=np.identity(len(ipok)*len(themaps))*signoise**2
guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],0]
specout,error,invfisher,lk,num,ds_dcb=qml.qml(themaps,mask,covmap,ellbins,fwhmrad,guess,ds_dcb,spectra,cholesky=True,temp=True,polar=True,plot=True,itmax=20)









#### TEB with EB and TB
ds_dcb=0
nbpixok=2000
mask=(np.arange(12*nside**2) >= nbpixok)
maskok=~mask
ip=np.arange(12*nside**2)
ipok=ip[~mask]

ellbins=[0,50,100,150,200,250,300,350,3*nside]
nbins=len(ellbins)-1
ellmin=np.array(ellbins[0:nbins])
ellmax=np.array(ellbins[1:nbins+1])-1
ellval=(ellmin+ellmax)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)

reload(qml)
ll=np.arange(int(np.max(ellbins))+1)
bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)
ds_dcb=qml.compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=True,temp=True,EBTB=True)

signoise=0.05
mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
themapi=mapi.copy()
themapi[mask]=0
themapi[~mask]+=np.random.randn(len(themapi[~mask]))*signoise
themapi[~mask]-=np.mean(themapi[~mask])
themapq=mapq.copy()
themapq[mask]=0
themapq[~mask]+=np.random.randn(len(themapq[~mask]))*signoise
themapq[~mask]-=np.mean(themapq[~mask])
themapu=mapu.copy()
themapu[mask]=0
themapu[~mask]+=np.random.randn(len(themapu[~mask]))*signoise
themapu[~mask]-=np.mean(themapu[~mask])
themaps=[themapi,themapq,themapu]

covmap=np.identity(len(ipok)*len(themaps))*signoise**2
guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],0]
specout,error,invfisher,lk,num,ds_dcb=qml.qml(themaps,mask,covmap,ellbins,fwhmrad,guess,ds_dcb,spectra,cholesky=True,temp=True,polar=True,plot=True,EBTB=True,itmax=20)





