from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
#from pyoperators import pcg, DiagonalOperator, UnpackOperator
#from pysimulators import ProjectionInMemoryOperator
#from qubic import QubicConfiguration, QubicInstrument, create_random_pointings
from Quad import qml
from Quad import pyquad
#path = os.path.dirname('/Users/hamilton/idl/pro/Qubic/People/pierre/qubic/script/script_ga.py')



#### This code tries to recalculate le pixel-pixel covariance matrix from MC and compare it to the expected value from theory




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




#### Generate map for B-modes directly
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
allmapsi=np.zeros((nbmc,npix))
allmapsq=np.zeros((nbmc,npix))
allmapsu=np.zeros((nbmc,npix))
for i in np.arange(nbmc):
    pyquad.progress_bar(i,nbmc)
    mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=0,pixwin=True,new=True)
    allmapsi[i,:]=mapi[~mask]
    allmapsq[i,:]=mapq[~mask]
    allmapsu[i,:]=mapu[~mask]

covii=qml.cov_from_maps(allmapsi,allmapsi)
coviq=qml.cov_from_maps(allmapsi,allmapsq)
coviu=qml.cov_from_maps(allmapsi,allmapsu)

covqi=qml.cov_from_maps(allmapsq,allmapsi)
covqq=qml.cov_from_maps(allmapsq,allmapsq)
covqu=qml.cov_from_maps(allmapsq,allmapsu)

covui=qml.cov_from_maps(allmapsu,allmapsi)
covuq=qml.cov_from_maps(allmapsu,allmapsq)
covuu=qml.cov_from_maps(allmapsu,allmapsu)

clf()
subplot(3,3,1)
imshow(covii,interpolation='nearest')
title('II')
colorbar()
subplot(3,3,2)
imshow(coviq,interpolation='nearest')
title('IQ')
colorbar()
subplot(3,3,3)
imshow(coviu,interpolation='nearest')
title('IU')
colorbar()
subplot(3,3,4)
imshow(covqi,interpolation='nearest')
title('QI')
colorbar()
subplot(3,3,5)
imshow(covqq,interpolation='nearest')
title('QQ')
colorbar()
subplot(3,3,6)
imshow(covqu,interpolation='nearest')
title('QU')
colorbar()
subplot(3,3,7)
imshow(covui,interpolation='nearest')
title('UI')
colorbar()
subplot(3,3,8)
imshow(covuq,interpolation='nearest')
title('UQ')
colorbar()
subplot(3,3,9)
imshow(covuu,interpolation='nearest')
title('UU')
colorbar()

bigmatmc=np.array([[covii,coviq,coviu],[covqi,covqq,covqu],[covui,covuq,covuu]])
newmatmc=qml.allmat2bigmat(bigmatmc)
clf()
imshow(np.log10(np.abs(newmatmc)),interpolation='nearest')
colorbar()




#### calcul des matrices de cov theoriques I,Q,U
reload(qml)
import healpy as hp

ip=np.arange(12*nside**2)
ipok=ip[~mask]
lmax=3*nside
maskl=ell<(lmax+1)
bl=np.ones(maskl.size)
#spectra=[ell,ctt,cee,cbb,cte]   #### This is the new ordering

allcovth=qml.covth(nside,ipok,lmax,bl,spectra,polar=True,temp=True,allinone=False)
newmatth=qml.allmat2bigmat(allcovth)
bigmatmc=np.array([[covii,coviq,coviu],[covqi,covqq,covqu],[covui,covuq,covuu]])
newmatmc=qml.allmat2bigmat(bigmatmc)




clf()
subplot(1,3,1)
title('theory')
imshow(np.log10(np.abs(newmatth)),interpolation='nearest',vmin=-6,vmax=3)
colorbar()
subplot(1,3,2)
title('MC')
imshow(np.log10(np.abs(newmatmc)),interpolation='nearest',vmin=-6,vmax=3)
colorbar()
subplot(1,3,3)
title('MC-Theory')
imshow(newmatmc-newmatth,interpolation='nearest')
colorbar()

diffmat=bigmatmc-allcovth
clf()
subplot(3,3,1)
hist(diffmat[0,0,:,:].flatten(),100)
subplot(3,3,2)
hist(diffmat[0,1,:,:].flatten(),100)
subplot(3,3,3)
hist(diffmat[0,2,:,:].flatten(),100)
subplot(3,3,4)
hist(diffmat[1,0,:,:].flatten(),100)
subplot(3,3,5)
hist(diffmat[1,1,:,:].flatten(),100)
subplot(3,3,6)
hist(diffmat[1,2,:,:].flatten(),100)
subplot(3,3,7)
hist(diffmat[2,0,:,:].flatten(),100)
subplot(3,3,8)
hist(diffmat[2,1,:,:].flatten(),100)
subplot(3,3,9)
hist(diffmat[2,2,:,:].flatten(),100)



#### calcul des matrices de cov theoriques Q,U
reload(qml)
import healpy as hp

ip=np.arange(12*nside**2)
ipok=ip[~mask]
lmax=3*nside
maskl=ell<(lmax+1)
spectra=[ell[maskl],ctt[maskl],cee[maskl],cbb[maskl],cte[maskl]]   #### This is the new ordering

allcovth=qml.covth(nside,ipok,lmax,bl,spectra,polar=True,temp=False,allinone=False)
newmatth=qml.allmat2bigmat(allcovth)
bigmatmc=np.array([[covqq,covqu],[covuq,covuu]])
newmatmc=qml.allmat2bigmat(bigmatmc)

clf()
subplot(1,3,1)
title('theory')
imshow(np.log10(np.abs(newmatth)),interpolation='nearest',vmin=-6,vmax=3)
colorbar()
subplot(1,3,2)
title('MC')
imshow(np.log10(np.abs(newmatmc)),interpolation='nearest',vmin=-6,vmax=3)
colorbar()
subplot(1,3,3)
title('MC-Theory')
imshow(newmatmc-newmatth,interpolation='nearest')
colorbar()

diffmat=bigmatmc-allcovth
clf()
subplot(2,2,1)
hist(diffmat[0,0,:,:].flatten(),100)
subplot(2,2,2)
hist(diffmat[0,1,:,:].flatten(),100)
subplot(2,2,3)
hist(diffmat[1,0,:,:].flatten(),100)
subplot(2,2,4)
hist(diffmat[1,1,:,:].flatten(),100)










