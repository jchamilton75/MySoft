from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
from Quad import qml
from Quad import pyquad


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
mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=0,pixwin=True,new=True)
hp.mollview(mapi)
hp.mollview(mapq)
hp.mollview(mapu)

#### coverage cut
nbpixok=50
mask=(np.arange(12*nside**2) >= nbpixok)

#### maps simulation
nbmc=1000
npix=np.size(where(mask==False))
allmapsi1=np.zeros((nbmc,npix))
allmapsq1=np.zeros((nbmc,npix))
allmapsu1=np.zeros((nbmc,npix))
allmapsi2=np.zeros((nbmc,npix))
allmapsq2=np.zeros((nbmc,npix))
allmapsu2=np.zeros((nbmc,npix))
sigI=200.
sigQU=10.
for i in np.arange(nbmc):
    pyquad.progress_bar(i,nbmc)
    mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=0,pixwin=True,new=True)
    noisei1=np.random.randn(12*nside**2)*sigI
    noiseq1=np.random.randn(12*nside**2)*sigQU
    noiseu1=np.random.randn(12*nside**2)*sigQU
    noisei2=np.random.randn(12*nside**2)*sigI
    noiseq2=np.random.randn(12*nside**2)*sigQU
    noiseu2=np.random.randn(12*nside**2)*sigQU
    allmapsi1[i,:]=mapi[~mask]+noisei1[~mask]
    allmapsq1[i,:]=mapq[~mask]+noiseq1[~mask]
    allmapsu1[i,:]=mapu[~mask]+noiseu1[~mask]
    allmapsi2[i,:]=mapi[~mask]+noisei2[~mask]
    allmapsq2[i,:]=mapq[~mask]+noiseq2[~mask]
    allmapsu2[i,:]=mapu[~mask]+noiseu2[~mask]

############ Auto
covii=qml.cov_from_maps(allmapsi1,allmapsi1)
coviq=qml.cov_from_maps(allmapsi1,allmapsq1)
coviu=qml.cov_from_maps(allmapsi1,allmapsu1)
covqi=qml.cov_from_maps(allmapsq1,allmapsi1)
covqq=qml.cov_from_maps(allmapsq1,allmapsq1)
covqu=qml.cov_from_maps(allmapsq1,allmapsu1)
covui=qml.cov_from_maps(allmapsu1,allmapsi1)
covuq=qml.cov_from_maps(allmapsu1,allmapsq1)
covuu=qml.cov_from_maps(allmapsu1,allmapsu1)
bigmatmc=np.array([[covii,coviq,coviu],[covqi,covqq,covqu],[covui,covuq,covuu]])
newmatmc=qml.allmat2bigmat(bigmatmc)
clf()
imshow(np.log10(np.abs(newmatmc)),interpolation='nearest')
colorbar()

############ Cross
coviix=qml.cov_from_maps(allmapsi1,allmapsi2)
coviqx=qml.cov_from_maps(allmapsi1,allmapsq2)
coviux=qml.cov_from_maps(allmapsi1,allmapsu2)
covqix=qml.cov_from_maps(allmapsq1,allmapsi2)
covqqx=qml.cov_from_maps(allmapsq1,allmapsq2)
covqux=qml.cov_from_maps(allmapsq1,allmapsu2)
covuix=qml.cov_from_maps(allmapsu1,allmapsi2)
covuqx=qml.cov_from_maps(allmapsu1,allmapsq2)
covuux=qml.cov_from_maps(allmapsu1,allmapsu2)

bigmatmcx=np.array([[coviix,coviqx,coviux],[covqix,covqqx,covqux],[covuix,covuqx,covuux]])
newmatmcx=qml.allmat2bigmat(bigmatmcx)
clf()
imshow(np.log10(np.abs(newmatmcx)),interpolation='nearest')
colorbar()


#### calcul des matrices de cov theoriques I,Q,U
reload(qml)
ip=np.arange(12*nside**2)
ipok=ip[~mask]
lmax=3*nside
maskl=ell<(lmax+1)
bl=np.ones(maskl.size)
#spectra=[ell,ctt,cee,cbb,cte]   #### This is the new ordering

allcovth=qml.covth(nside,ipok,lmax,bl,spectra,polar=True,temp=True,allinone=False)
newmatth=qml.allmat2bigmat(allcovth)


mini = 0
maxi = 1e4
clf()
subplot(2,3,1)
imshow(covii,interpolation='nearest', vmin=mini, vmax=maxi)
colorbar()
title('Auto')
subplot(2,3,2)
imshow(coviix,interpolation='nearest', vmin=mini, vmax=maxi)
colorbar()
title('Cross')
subplot(2,3,3)
imshow(allcovth[0,0,:,:],interpolation='nearest', vmin=mini, vmax=maxi)
colorbar()
title('Theory')
subplot(2,3,4)
imshow(covii-allcovth[0,0,:,:],interpolation='nearest', vmin=mini, vmax=maxi)
colorbar()
title('Auto-Theory')
subplot(2,3,5)
imshow(coviix-allcovth[0,0,:,:],interpolation='nearest', vmin=mini, vmax=maxi)
colorbar()
title('Cross-Theory')
subplot(2,3,6)
imshow(covii-coviix,interpolation='nearest', vmin=mini, vmax=maxi)
colorbar()
title('Auto-Cross')


########################################
# OK donc la matrice de cov croisee est bien libre de bruit
########################################








