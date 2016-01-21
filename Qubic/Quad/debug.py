from __future__ import division

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp

import pyquad

### read input spectrum
#a=np.loadtxt('/Volumes/Data/Qubic/qubic_v1/cl_r=0.1bis2.txt')
a=np.loadtxt('./cl_r=0.1bis2.txt')
ell=a[:,0]
ctt=np.concatenate([[0,0],a[:,1]*1e12*2*np.pi/(ell*(ell+1))])
cee=np.concatenate([[0,0],a[:,2]*1e12*2*np.pi/(ell*(ell+1))])
cte=np.concatenate([[0,0],a[:,4]*1e12*2*np.pi/(ell*(ell+1))])
cbb=np.concatenate([[0,0],a[:,7]*1e12*2*np.pi/(ell*(ell+1))])
ell=np.concatenate([[0,1],a[:,0]])

wtozero = (ell < 50) | (ell > 249)
ctt[wtozero]=0
cee[wtozero]=0
cte[wtozero]=0
cbb[wtozero]=0

spectra=[ell,ctt,cte,cee,cbb]   #old ordering

clf()
plot(ell,np.sqrt(spectra[1]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TT}$')
plot(ell,np.sqrt(abs(spectra[2])*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TE}$')
plot(ell,np.sqrt(spectra[3]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{EE}$')
plot(ell,np.sqrt(spectra[4]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{BB} (r=0.1)$')
yscale('log')
xlim(0,300)
ylim(0.01,100)
xlabel('$\ell$')
ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
legend(loc='lower right',frameon=False)



nside=128
fwhm=30./60*np.pi/180
signoise=0.001
nbmc=10000
npix=1000

allmaps=np.zeros((npix,nbmc))
allmaps_nonoise=np.zeros((npix,nbmc))
allmaps_nosig=np.zeros((npix,nbmc))
allcl=np.zeros((251,nbmc))
for i in arange(nbmc):
    print(i)
    #### Create input map and covariance matrix
    map_orig=hp.synfast(spectra[4],nside,fwhm=fwhm,pixwin=True)
    allcl[:,i]=hp.anafast(map_orig,lmax=250)
    # input map
    mapin=map_orig.copy()
    mapin[npix:mapin.size]=0
    mask= mapin != 0
    allmaps_nonoise[:,i]=mapin[mask]
    # add noise
    map=mapin.copy()
    map[mask]+=np.random.randn(npix)*signoise
    allmaps[:,i]=map[mask]
    allmaps_nosig[:,i]=map[mask]-mapin[mask]
    
mcl=np.zeros(251)
for i in arange(251):
    mcl[i]=np.mean(allcl[i,:])

clf()
plot(ell,spectra[1]*(ell*(ell+1))/(2*np.pi),'r')
xlim(0,251)
thel=np.arange(251)
thebell=np.exp(-0.5*(thel**2)*((fwhm/2.35)**2))
pixwin=hp.pixwin(hp.npix2nside(len(mapin)))[0:251]
plot(thel,mcl*thel*(thel+1)/(2*np.pi)/thebell**2/pixwin**2)



covmc=np.zeros((npix,npix))
covmc_nonoise=np.zeros((npix,npix))
covmc_nosig=np.zeros((npix,npix))
for i in np.arange(npix):
    print(i)
    for j in np.arange(i,npix):
        covmc[i,j]=mean( (allmaps[i,:]-mean(allmaps[i,:]))*(allmaps[j,:]-mean(allmaps[j,:])))
        covmc_nonoise[i,j]=mean( (allmaps_nonoise[i,:]-mean(allmaps_nonoise[i,:]))*(allmaps_nonoise[j,:]-mean(allmaps_nonoise[j,:])))
        covmc_nosig[i,j]=mean( (allmaps_nosig[i,:]-mean(allmaps_nosig[i,:]))*(allmaps_nosig[j,:]-mean(allmaps_nosig[j,:])))
        covmc[j,i]=covmc[i,j]
        covmc_nosig[j,i]=covmc_nosig[i,j]
        covmc_nonoise[j,i]=covmc_nonoise[i,j]



clf()
imshow(covmc,interpolation='nearest')
colorbar()
clf()
imshow(covmc_nonoise,interpolation='nearest')
colorbar()


#### calcul théorique de la matrice de covariance
import scipy
import scipy.interpolate
pixwin=hp.pixwin(hp.npix2nside(len(map)))
iprings=np.arange(12*nside**2)
vecs=hp.pix2vec(int(nside),iprings[mask])
cosangles=np.dot(np.transpose(vecs),vecs)
lll=arange(251)
bl=np.exp(-0.5*(lll**2)*((fwhm/2.35)**2))
clvals=np.zeros(251)
clvals[2:]=(spectra[1])[:249]
pixw=pixwin[0:251]

covth=np.zeros((npix,npix))
for i in arange(npix):
    print(i)
    for j in arange(i,npix):
        pl=scipy.special.lpn(250,cosangles[i,j])[0]
        bla=(2*lll+1)/(4*np.pi)*clvals*(bl**2)*(pixw**2)*pl
        bla=np.sum(bla)
        covth[i,j]=bla
        covth[j,i]=bla


clf()
plot(covth,covmc_nonoise,'r.')
plot(arange(4000))

clf()
plot(covth[500,:])
plot(covmc_nonoise[500,:])

clf()
plot(covth[500,:]-covmc_nonoise[500,:])

clf()
plot(covth[500,:],covth[500,:]-covmc_nonoise[500,:],'r.')
from pyquad import pyquad
pyquad.profile(covth[500,:],covth[500,:]-covmc_nonoise[500,:],500,3500,10)
pyquad.profile(covth[500,:],covth[500,:]-covmc_nonoise[500,:],-500,500,20,dispersion=False)

clf()
xlim(-500,500)
plot(covth[500,:],covth[500,:]-covmc_nonoise[500,:],'r.')
pyquad.profile(covth[500,:],covth[500,:]-covmc_nonoise[500,:],-500,500,50,dispersion=False)



#### celle qui vient de pyquad
import pickle
infile=open('data.pkl', 'rb')
matsky=pickle.load(infile)
infile.close()

clf()
plot(covmc_nonoise[500,:],label='MC')
plot(matsky[500,:],label='PyQuad')
plot(covth[500,:],'--',label='Th')
legend()


clf()
plot(covth,covmc_nonoise,'bo',alpha=0.01)
plot(covth,matsky,'ro',alpha=0.7)
plot(arange(1200))

clf()
plot(matsky,covmc_nonoise,'b.')
plot(matsky,covth,'r.')
plot(arange(1200))



clf()
imshow(covth,interpolation='nearest')
colorbar()

clf()
imshow(covmc_nonoise,interpolation='nearest')
colorbar()

clf()
imshow(matsky,interpolation='nearest')
colorbar()

clf()
plot(covth,matsky,'r,')
plot(arange(1200))

clf()
plot(matsky,covmc_nonoise,'r,')
plot(arange(1200))




clf()
imshow(covmc_nonoise/covth,interpolation='nearest',vmin=0,vmax=2)
colorbar()


clf()
imshow(covmc_nosig,inyerpolation=nearest)
colorbar()









