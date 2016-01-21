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


####### Calcul de la covariance par bins
nside=128
nbpixok=50
mask=(np.arange(12*nside**2) >= nbpixok)

ip=np.arange(12*nside**2)
ipok=ip[~mask]
lmax=3*nside
ell=spectra[0]
maskl=ell<(lmax+1)
bl=np.ones(maskl.size)

reload(qml)
allcovth=qml.covth(nside,ipok,lmax,bl,spectra,polar=False,temp=True)
ellbins=[0,30,60,90,120,384]
allcovth_bins=qml.covth_bins(ellbins,nside,ipok,bl,spectra,polar=False,temp=True)

newcovth=np.zeros_like(allcovth)
for i in np.arange(len(ellbins)-1):
    newcovth += allcovth_bins[i,:,:]


nstokes=allcovth.shape[0]/nbpixok
nn=1
clf()
for i in np.arange(nstokes):
    for j in np.arange(nstokes):
        subplot(nstokes,nstokes,nn)
        imshow(allcovth[i*nbpixok:(i+1)*nbpixok,j*nbpixok:(j+1)*nbpixok]-newcovth[i*nbpixok:(i+1)*nbpixok,j*nbpixok:(j+1)*nbpixok],interpolation='nearest')
        colorbar()
        nn+=1

###### OK ca marche


