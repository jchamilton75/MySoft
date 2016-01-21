from __future__ import division
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



############# Calcul des derivees de la matrice de covariance par bin
nside=128
nbpixok=50
mask=(np.arange(12*nside**2) >= nbpixok)
ip=np.arange(12*nside**2)
ipok=ip[~mask]
lmax=3*nside
ell=spectra[0]
maskl=ell<(lmax+1)
bl=np.ones(maskl.size)
ellbins=[0,10,20]

reload(qml)
cov=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=True,temp=True)
cov=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=True,temp=True,EBTB=True)

cov=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=True,temp=False)
cov=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=True,temp=False,EBTB=True)

cov=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=False,temp=True)
cov=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=False,temp=True,EBTB=True)


#### Verification avec un seul bin en ell
nside=128
nbpixok=50
mask=(np.arange(12*nside**2) >= nbpixok)
ip=np.arange(12*nside**2)
ipok=ip[~mask]
lmax=3*nside
ell=spectra[0]
maskl=ell<(lmax+1)
bl=np.ones(maskl.size)
ellbins=[0,24]

rpix=np.array(hp.pix2vec(nside,ipok))
allcosang=np.dot(np.transpose(rpix),rpix)

##################### cas polar=True, temp=True, EBTB=False
reload(qml)
dsdcb=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=True,temp=True)

spec_dTT=[ell,ctt*0+1,cee*0,cbb*0,cte*0]
cov_dTT=qml.covth_bins(ellbins,nside,ipok,allcosang,bl,spec_dTT,polar=True,temp=True)
spec_dEE=[ell,ctt*0,cee*0+1,cbb*0,cte*0]
cov_dEE=qml.covth_bins(ellbins,nside,ipok,allcosang,bl,spec_dEE,polar=True,temp=True)
spec_dBB=[ell,ctt*0,cee*0,cbb*0+1,cte*0]
cov_dBB=qml.covth_bins(ellbins,nside,ipok,allcosang,bl,spec_dBB,polar=True,temp=True)
spec_dTE=[ell,ctt*0,cee*0,cbb*0,cte*0+1]
cov_dTE=qml.covth_bins(ellbins,nside,ipok,allcosang,bl,spec_dTE,polar=True,temp=True)

clf()
subplot(2,2,1)
imshow(dsdcb[0,0,:,:]-cov_dTT[0,:,:],interpolation='nearest')
subplot(2,2,2)
imshow(dsdcb[1,0,:,:]-cov_dEE[0,:,:],interpolation='nearest')
subplot(2,2,3)
imshow(dsdcb[2,0,:,:]-cov_dBB[0,:,:],interpolation='nearest')
subplot(2,2,4)
imshow(dsdcb[3,0,:,:]-cov_dTE[0,:,:],interpolation='nearest')


##################### cas polar=True, temp=True, EBTB=True
reload(qml)
dsdcb=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=True,temp=True,EBTB=True)

spec_dTT=[ell,ctt*0+1,cee*0,cbb*0,cte*0,cte*0,cte*0]
cov_dTT=qml.covth_bins(ellbins,nside,ipok,bl,spec_dTT,polar=True,temp=True)
spec_dEE=[ell,ctt*0,cee*0+1,cbb*0,cte*0,cte*0,cte*0]
cov_dEE=qml.covth_bins(ellbins,nside,ipok,bl,spec_dEE,polar=True,temp=True)
spec_dBB=[ell,ctt*0,cee*0,cbb*0+1,cte*0,cte*0,cte*0]
cov_dBB=qml.covth_bins(ellbins,nside,ipok,bl,spec_dBB,polar=True,temp=True)
spec_dTE=[ell,ctt*0,cee*0,cbb*0,cte*0+1,cte*0,cte*0]
cov_dTE=qml.covth_bins(ellbins,nside,ipok,bl,spec_dTE,polar=True,temp=True)
spec_dEB=[ell,ctt*0,cee*0,cbb*0,cte*0,cte*0+1,cte*0]
cov_dEB=qml.covth_bins(ellbins,nside,ipok,bl,spec_dEB,polar=True,temp=True)
spec_dTB=[ell,ctt*0,cee*0,cbb*0,cte*0,cte*0,cte*0+1]
cov_dTB=qml.covth_bins(ellbins,nside,ipok,bl,spec_dTB,polar=True,temp=True)

clf()
subplot(3,2,1)
imshow(dsdcb[0,0,:,:]-cov_dTT[0,:,:],interpolation='nearest')
subplot(3,2,2)
imshow(dsdcb[1,0,:,:]-cov_dEE[0,:,:],interpolation='nearest')
subplot(3,2,3)
imshow(dsdcb[2,0,:,:]-cov_dBB[0,:,:],interpolation='nearest')
subplot(3,2,4)
imshow(dsdcb[3,0,:,:]-cov_dTE[0,:,:],interpolation='nearest')
subplot(3,2,5)
imshow(dsdcb[4,0,:,:]-cov_dEB[0,:,:],interpolation='nearest')
subplot(3,2,6)
imshow(dsdcb[5,0,:,:]-cov_dTB[0,:,:],interpolation='nearest')



##################### cas polar=True, temp=False, EBTB=False
reload(qml)
dsdcb=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=True,temp=False)

spec_dEE=[ell,ctt*0,cee*0+1,cbb*0,cte*0]
cov_dEE=qml.covth_bins(ellbins,nside,ipok,bl,spec_dEE,polar=True,temp=False)
spec_dBB=[ell,ctt*0,cee*0,cbb*0+1,cte*0]
cov_dBB=qml.covth_bins(ellbins,nside,ipok,bl,spec_dBB,polar=True,temp=False)

clf()
subplot(1,2,1)
imshow(dsdcb[0,0,:,:]-cov_dEE[0,:,:],interpolation='nearest')
subplot(1,2,2)
imshow(dsdcb[1,0,:,:]-cov_dBB[0,:,:],interpolation='nearest')


##################### cas polar=True, temp=False, EBTB=True
reload(qml)
dsdcb=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=True,temp=False,EBTB=True)

spec_dEE=[ell,ctt*0,cee*0+1,cbb*0,cte*0,cte*0,cte*0]
cov_dEE=qml.covth_bins(ellbins,nside,ipok,bl,spec_dEE,polar=True,temp=False)
spec_dBB=[ell,ctt*0,cee*0,cbb*0+1,cte*0,cte*0,cte*0]
cov_dBB=qml.covth_bins(ellbins,nside,ipok,bl,spec_dBB,polar=True,temp=False)
spec_dEB=[ell,ctt*0,cee*0,cbb*0,cte*0,cte*0+1,cte*0]
cov_dEB=qml.covth_bins(ellbins,nside,ipok,bl,spec_dEB,polar=True,temp=False)

clf()
subplot(1,3,1)
imshow(dsdcb[0,0,:,:]-cov_dEE[0,:,:],interpolation='nearest')
subplot(1,3,2)
imshow(dsdcb[1,0,:,:]-cov_dBB[0,:,:],interpolation='nearest')
subplot(1,3,3)
imshow(dsdcb[2,0,:,:]-cov_dEB[0,:,:],interpolation='nearest')




##################### cas polar=False, temp=True, EBTB=False
reload(qml)
dsdcb=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=False,temp=True)

spec_dTT=[ell,ctt*0+1,cee*0,cbb*0,cte*0]
cov_dTT=qml.covth_bins(ellbins,nside,ipok,bl,spec_dTT,polar=False,temp=True)

clf()
imshow(dsdcb[0,0,:,:]-cov_dTT[0,:,:],interpolation='nearest')


##################### cas polar=False, temp=True, EBTB=True
reload(qml)
dsdcb=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=False,temp=True,EBTB=True)

spec_dTT=[ell,ctt*0+1,cee*0,cbb*0,cte*0,cte*0,cte*0]
cov_dTT=qml.covth_bins(ellbins,nside,ipok,bl,spec_dTT,polar=False,temp=True)

clf()
imshow(dsdcb[0,0,:,:]-cov_dTT[0,:,:],interpolation='nearest')




################### Version parallele : chaque thread fait une derivee
nside=128
nbpixok=130
mask=(np.arange(12*nside**2) >= nbpixok)
ip=np.arange(12*nside**2)
ipok=ip[~mask]
lmax=3*nside
ell=spectra[0]
maskl=ell<(lmax+1)
bl=np.ones(maskl.size)
ellbins=[0,3*nside]

reload(qml)
import time
t0=time.time()
dsdcb=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=True,temp=False,EBTB=False)
t1=time.time()

t0par=time.time()
dsdcbpar=qml.compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=True,temp=False,EBTB=False)
t1par=time.time()

#### loop to check efficiency
allnbpix=np.arange(4)*50+50
dt=np.zeros(len(allnbpix))
dtpar=np.zeros(len(allnbpix))
for i in np.arange(len(allnbpix)):
    nbpixok=allnbpix[i]
    mask=(np.arange(12*nside**2) >= nbpixok)
    ip=np.arange(12*nside**2)
    ipok=ip[~mask]
    t0=time.time()
    dsdcb=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=True,temp=True,EBTB=True)
    t1=time.time()
    t0par=time.time()
    dsdcbpar=qml.compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=True,temp=True,EBTB=True)
    t1par=time.time()
    dt[i]=t1-t0
    dtpar[i]=t1par-t0par

clf()    
plot(allnbpix,dt,'go')
plot(allnbpix,dtpar,'ro')

clf()    
plot(allnbpix,dt/dtpar,'go')

