from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
from Quad import qml
from Quad import pyquad
import healpy as hp
from pysimulators import FitsArray


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


#### EB Only
ds_dcb=0
nbpixok=1000    #1% sky coverage at nside=128
mask=(np.arange(12*nside**2) >= nbpixok)
maskok=~mask
ip=np.arange(12*nside**2)
ipok=ip[~mask]

#ellbins = [0, 50, 100, 150, 200]
ellbins = [0, 25, 50, 75, 100, 125, 150, 175, 200]
nbins=len(ellbins)-1
ellmin=np.array(ellbins[0:nbins])
ellmax=np.array(ellbins[1:nbins+1])-1
ellval=(ellmin+ellmax)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)

reload(qml)
ll=np.arange(int(np.max(ellbins))+1)
bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)

### Compute ds/dCb
#ds_dcb=qml.compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=True,temp=False)
#FitsArray(ds_dcb, copy=False).save('myds_dcb64_dl25.fits')
ds_dcb = FitsArray('myds_dcb64_dl25.fits')

# mi = 0
# ma = 7
# ellbins = ellbins[mi:ma+1]
# ellmin=np.array(ellbins[0:len(ellbins)-1])
# ellmax=np.array(ellbins[1:len(ellbins)+1])-1
# ellval=(ellmin+ellmax)/2
# binspec=pyquad.binspectrum(spectra,ellmin,ellmax)
# ll=np.arange(int(np.max(ellbins))+1)
# bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)
#
# ds_dcb = ds_dcb[:,mi:ma,:,:]


#### One realization
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

#hp.mollview(mapq,rot=[0,90],title='Q')
#hp.mollview(mapu,rot=[0,90],title='U')
#hp.gnomview(themaps[0],rot=[0,90],reso=10,title='Q')
#hp.gnomview(themaps[1],rot=[0,90],reso=10,title='U')

covmap=np.identity(len(ipok)*len(themaps))*signoise**2
guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],0]
specout,error,invfisher,lk,num,ds_dcb, convergence=qml.qml(themaps,mask,covmap,ellbins,fwhmrad,guess,ds_dcb,spectra,cholesky=True,temp=False,polar=True,plot=True,itmax=30)



#### Now a MC
import string
import random
def random_string(nchars):
    lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(nchars)]
    str = "".join(lst)
    return(str)

nbmc = 1000
signoise=0.05
covmap=np.identity(len(ipok)*len(themaps))*signoise**2
guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],0]
allBBout = []
allBBerror = []
allEEout = []
allEEerror = []
for i in xrange(nbmc):
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
    specout,error,invfisher,lk,num,ds_dcb, convergence=qml.qml(themaps,mask,covmap,ellbins,fwhmrad,guess,ds_dcb,
        spectra,cholesky=True,temp=False,polar=True,plot=True,itmax=40)
    if convergence ==   1:
        allBBout.append(specout[3])
        allBBerror.append(error[3])
        allEEout.append(specout[2])
        allEEerror.append(error[2])
    title('MC num: {0:3} - Kept: {1:3}'.format(i, len(allBBout)))
    draw()
    strname = random_string(10)
    FitsArray([specout[3], error[3], specout[2], error[2]], 
        copy=False).save('/Users/hamilton/Qubic/Quad/QuadMC/dl25_whitenoise/cl_'+strname+'.fits')



### retrieve MC
import glob

allBBout = []
allBBerror = []
allEEout = []
allEEerror = []
for file in glob.glob('/Users/hamilton/Qubic/Quad/QuadMC/dl25_whitenoise/*.fits'):
    bla = FitsArray(file)
    allBBout.append(bla[0])
    allBBerror.append(bla[1])
    allEEout.append(bla[2])
    allEEerror.append(bla[3])

theallBBout = np.array(allBBout)
theallBBerror = np.array(allBBerror)
theallEEout = np.array(allEEout)
theallEEerror = np.array(allEEerror)

### cut outliers
def clip(arr, nbs, niter):
    good = np.isfinite(arr)
    print(np.sum(good))
    for i in xrange(niter):
        mm = np.mean(arr[good])
        ss = np.std(arr[good])
        good *= np.abs(arr-mm) <= (nbs * ss)
        print(i,mm,ss,np.sum(good))
    return good

nbs = 4 
a=clip(np.mean(theallBBout, axis=1), nbs, 20) * clip(np.mean(theallEEout, axis=1), nbs, 20)
theallBBout = theallBBout[a,:]
theallBBerror = theallBBerror[a,:]
theallEEout = theallEEout[a,:]
theallEEerror = theallEEerror[a,:]


     
rec_cee = np.mean(theallEEout, axis=0)
rec_cbb = np.mean(theallBBout, axis=0)
err_cee = np.std(theallEEout, axis=0)
err_cbb = np.std(theallBBout, axis=0)


clf()
ellb = ellval
subplot(1,2,1)
xlim(0, np.max(ellmax))
ylim(0, 1.5)
plot(ell,cee*(ell*(ell+1))/(2*np.pi),lw=3)
for i in xrange(theallEEout.shape[0]):
    plot(ellb, theallEEout[i,:]*ellb*(ellb+1)/(2*np.pi),'k.')
    plot(ellb, binspec[:,2]*ellb*(ellb+1)/(2*np.pi),'go')
errorbar(ellb,rec_cee*ellb*(ellb+1)/(2*np.pi), err_cee*ellb*(ellb+1)/(2*np.pi),xerr=(ellmax+1-ellmin)/2,
    label=str(i),fmt='ro', lw=6)
mp.xlabel('$\ell$')
mp.ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')

subplot(1,2,2)
xlim(0, np.max(ellmax))
ylim(0, 0.05)
plot(ell,cbb*(ell*(ell+1))/(2*np.pi),lw=3)
for i in xrange(theallBBout.shape[0]):
    plot(ellb, theallBBout[i,:]*ellb*(ellb+1)/(2*np.pi),'k.')
    plot(ellb, binspec[:,3]*ellb*(ellb+1)/(2*np.pi),'go')
errorbar(ellb,rec_cbb*ellb*(ellb+1)/(2*np.pi), err_cbb*ellb*(ellb+1)/(2*np.pi),xerr=(ellmax+1-ellmin)/2,
    label=str(i),fmt='ro', lw=6)
mp.xlabel('$\ell$')
mp.ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')

def statstr(arr):
    return '{0:5.2f} +/- {1:5.2f}'.format(np.mean(arr), np.std(arr))

clf()
for bin in xrange(8):
    subplot(2,4,bin+1)
    rel_error = (theallBBout[:,bin]-binspec[bin,3])/binspec[bin,3]
    hist(rel_error, bins=20, range = [-3*std(rel_error), 3*std(rel_error)])

clf()
for bin in xrange(8):
    subplot(2,4,bin+1)
    rel_error = (theallEEout[:,bin]-binspec[bin,2])/binspec[bin,2]
    hist(rel_error, bins=20, range = [-3*std(rel_error), 3*std(rel_error)])


clf()
for bin in xrange(8):
    subplot(2,4,bin+1)
    pull2 = (theallBBout[:,bin]-binspec[bin,3])/theallBBerror[:,bin]
    hist(pull2, bins=20, range = [-3*std(pull2), 3*std(pull2)], alpha=0.3, label=statstr(pull2))
    legend(fontsize=9)
    title('BB[{0:}-{1:}]'.format(ellmin[bin], ellmax[bin]))

clf()
for bin in xrange(8):
    subplot(2,4,bin+1)
    pull2 = (theallEEout[:,bin]-binspec[bin,2])/theallEEerror[:,bin]
    hist(pull2, bins=20, range = [-3*std(pull2), 3*std(pull2)], alpha=0.3, label=statstr(pull2))
    legend(fontsize=9)
    title('EE[{0:}-{1:}]'.format(ellmin[bin], ellmax[bin]))




