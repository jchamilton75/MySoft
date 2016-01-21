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


#### binned spectrum
nside=128
ellbins=[0,50,100,150,200,250,300,350,3*nside]
nbins=len(ellbins)-1
ellmin=np.array(ellbins[0:nbins])
ellmax=np.array(ellbins[1:nbins+1])-1
ellmax[nbins-1]=np.max(ellbins)
ellval=(ellmin+ellmax)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)

nl=np.size(ell)
newspectra=[ell.copy(),np.zeros(nl),np.zeros(nl),np.zeros(nl),np.zeros(nl)]
for b in np.arange(nbins):
    for j in np.arange(4):
        mask=(ell>=ellmin[b]) & (ell <= ellmax[b])
        newspectra[j+1][mask]=binspec[b,j+1]
        newspectra[j+1][0]=0
        newspectra[j+1][1]=0
        
clf()
plot(ell,np.sqrt(spectra[1]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TT}$')
plot(ell,np.sqrt(abs(spectra[4])*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TE}$')
plot(ell,np.sqrt(spectra[2]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{EE}$')
plot(ell,np.sqrt(spectra[3]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{BB} (r=0.1)$')
plot(ell,np.sqrt(newspectra[1]*(ell*(ell+1))/(2*np.pi)))
plot(ell,np.sqrt(abs(newspectra[4])*(ell*(ell+1))/(2*np.pi)))
plot(ell,np.sqrt(newspectra[2]*(ell*(ell+1))/(2*np.pi)))
plot(ell,np.sqrt(newspectra[3]*(ell*(ell+1))/(2*np.pi)))
yscale('log')
xlim(0,3*nside+100)
ylim(0.01,100)
xlabel('$\ell$')
ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
legend(loc='lower right',frameon=False)


#### Generate map for B-modes directly
mapi,mapq,mapu=hp.synfast(newspectra[1:],nside,fwhm=0,pixwin=True,new=True)
hp.mollview(mapi)
hp.mollview(mapq)
hp.mollview(mapu)

#### coverage cut
nbpixok=100
mask=(np.arange(12*nside**2) >= nbpixok)

#### maps simulation
nbmc=100000
npix=np.size(where(mask==False))
allmapsi=np.zeros((nbmc,npix))
allmapsq=np.zeros((nbmc,npix))
allmapsu=np.zeros((nbmc,npix))
for i in np.arange(nbmc):
    pyquad.progress_bar(i,nbmc)
    mapi,mapq,mapu=hp.synfast(newspectra[1:],nside,fwhm=0,pixwin=True,new=True)
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
rpix=np.array(hp.pix2vec(nside,ipok))
allcosang=np.dot(np.transpose(rpix),rpix)


allcovth=qml.covth(nside,ipok,lmax,bl,newspectra,polar=True,temp=True,allinone=False)
newmatth=qml.allmat2bigmat(allcovth)
bigmatmc=np.array([[covii,coviq,coviu],[covqi,covqq,covqu],[covui,covuq,covuu]])
newmatmc=qml.allmat2bigmat(bigmatmc)

clf()
subplot(2,2,1)
imshow(np.log10(np.abs(newmatth)),interpolation='nearest',vmin=-5,vmax=4)
colorbar()
title('Theory')
subplot(2,2,2)
imshow(np.log10(np.abs(newmatmc)),interpolation='nearest',vmin=-5,vmax=4)
colorbar()
subplot(2,2,3)
imshow(newmatmc-newmatth,interpolation='nearest')
colorbar()
title('Difference')
subplot(2,2,4)
imshow(newmatmc-newmatth,interpolation='nearest',vmin=-0.01,vmax=0.01)
colorbar()
title('Difference')


#### les crois avec I sont OK
clf()
plot(allcovth[0,1,:,:],coviq,'k,')
xx=np.linspace(np.min(coviq),np.max(coviq),100)
plot(xx,xx)

clf()
plot(allcovth[0,2,:,:],coviu,'k,')
xx=np.linspace(np.min(coviu),np.max(coviu),100)
plot(xx,xx)

clf()
plot(allcovth[1,0,:,:],covqi,'k,')
xx=np.linspace(np.min(covqi),np.max(covqi),100)
plot(xx,xx)

clf()
plot(allcovth[2,0,:,:],covui,'k,')
xx=np.linspace(np.min(covui),np.max(covui),100)
plot(xx,xx)


#### les autres non

clf()
plot(allcovth[0,0,:,:],covii,'k,')
xx=np.linspace(np.min(covii),np.max(covii),100)
plot(xx,xx)


clf()
plot(allcovth[1,1,:,:],covqq,'k,')
xx=np.linspace(np.min(covqq),np.max(covqq),100)
plot(xx,xx)

clf()
plot(allcovth[1,2,:,:],covqu,'k,')
xx=np.linspace(np.min(covqu),np.max(covqu),100)
plot(xx,xx)

clf()
plot(allcovth[2,1,:,:],covuq,'k,')
xx=np.linspace(np.min(covuq),np.max(covuq),100)
plot(xx,xx)

clf()
plot(allcovth[2,2,:,:],covuu,'k,')
xx=np.linspace(np.min(covuu),np.max(covuu),100)
plot(xx,xx)


##### dans le cas avec uniquement E et B
reload(qml)
import healpy as hp

ip=np.arange(12*nside**2)
ipok=ip[~mask]
lmax=3*nside
maskl=ell<(lmax+1)
bl=np.ones(maskl.size)
rpix=np.array(hp.pix2vec(nside,ipok))
allcosang=np.dot(np.transpose(rpix),rpix)


allcovth=qml.covth(nside,ipok,lmax,bl,newspectra,polar=True,temp=False,allinone=False)
newmatth=qml.allmat2bigmat(allcovth)
bigmatmc=np.array([[covqq,covqu],[covuq,covuu]])
newmatmc=qml.allmat2bigmat(bigmatmc)

clf()
subplot(2,2,1)
imshow(np.log10(np.abs(newmatth)),interpolation='nearest',vmin=-5,vmax=4)
colorbar()
title('Theory')
subplot(2,2,2)
imshow(np.log10(np.abs(newmatmc)),interpolation='nearest',vmin=-5,vmax=4)
colorbar()
subplot(2,2,3)
imshow(newmatmc-newmatth,interpolation='nearest')
colorbar()
title('Difference')

####
# Donc la matrice de covariance totale est correcte: le bug est ailleurs
####

####
# On regarde par bins
####

lmax=np.max(ellbins)
allcovth=qml.covth(nside,ipok,lmax,bl,spectra,polar=True,temp=False,allinone=False)
newmatth=qml.allmat2bigmat(allcovth)

rpix=np.array(hp.pix2vec(nside,ipok))
allcosang=np.dot(np.transpose(rpix),rpix)
allcovthbins=qml.covth_bins(ellbins,nside,ipok,allcosang,bl,spectra,polar=True,temp=False,allinone=False)

bla=np.zeros_like(allcovthbins[0,:,:,:,:])
for i in np.arange(len(ellbins)-1): bla+=allcovthbins[i,:,:,:,:]
newmatthbins=qml.allmat2bigmat(bla)

clf()
plot(newmatthbins,newmatth,'ko',ms=1)
x=linspace(0,7000,100)
plot(x,x,'r')

clf()
subplot(2,2,1)
imshow(np.log10(np.abs(newmatth)),interpolation='nearest',vmin=-5,vmax=4)
colorbar()
title('Theory')
subplot(2,2,2)
imshow(np.log10(np.abs(newmatthbins)),interpolation='nearest',vmin=-5,vmax=4)
colorbar()
title('theory per bin')
subplot(2,2,3)
imshow(newmatth-newmatthbins,interpolation='nearest')
colorbar()
title('Difference')




#
# On va maitenant regarder la matrice de covariance recalculee a partir des derivees par rapport aux cl
####





###### On se place dans un cas simple
ds_dcb=0
ip=np.arange(12*nside**2)
ipok=ip[~mask]

ellbins=[0,50,100,150,200,250,300,350,3*nside]
#ellbins=[0,50,100]
nbins=len(ellbins)-1
ellmin=np.array(ellbins[0:nbins])
ellmax=np.array(ellbins[1:nbins+1])-1
ellmax[nbins-1]=np.max(ellbins)
ellval=(ellmin+ellmax)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)

binspec[:,1]


reload(qml)
ll=np.arange(int(np.max(ellbins))+1)
fwhmrad=0
bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)
lmax=np.max(ellbins)
maskl=ell<(lmax+1)
bl=np.ones(maskl.size)
ds_dcb=qml.compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=True,temp=True)

mapi,mapq,mapu=hp.synfast(newspectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
themapi=mapi.copy()
themapi[mask]=0
themapq=mapq.copy()
themapq[mask]=0
themapu=mapu.copy()
themapu[mask]=0
themaps=[themapi,themapq,themapu]

covmap=np.identity(len(ipok)*len(themaps))*signoise**2
guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],binspec[:,4]]
matsky=qml.qml(themaps,mask,covmap,ellbins,fwhmrad,guess,ds_dcb,newspectra,cholesky=True,temp=True,polar=True,plot=True,itmax=10,getmat=True)

allcovth=qml.covth(nside,ipok,lmax,bl,newspectra,polar=True,temp=True,allinone=False)
newmatth=qml.allmat2bigmat(allcovth)


clf()
plot(matsky,newmatth,'ro',alpha=0.2)
plot(np.arange(3000))

clf()
subplot(2,2,1)
imshow(np.log10(np.abs(matsky)),interpolation='nearest',vmin=-5,vmax=4)
colorbar()
title('From QML ds/dcb')
subplot(2,2,2)
imshow(np.log10(np.abs(newmatth)),interpolation='nearest',vmin=-5,vmax=4)
title('Th')
colorbar()
subplot(2,2,3)
imshow(newmatth-matsky,interpolation='nearest')
colorbar()
title('Difference')

