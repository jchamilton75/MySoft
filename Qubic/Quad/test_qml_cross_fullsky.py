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



nside=8
lmax=2*nside
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
nbpixok=12*nside**2
mask=(np.arange(12*nside**2) >= nbpixok)
maskok=~mask
ip=np.arange(12*nside**2)
ipok=ip[~mask]

deltal=2
ellbins = np.arange(2*nside/deltal)*deltal
nbins=len(ellbins)-1
ellmin=np.array(ellbins[0:nbins])
ellmax = ellmin+1
ellval=(ellmin+ellmax)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)

reload(qml)
ll=np.arange(int(np.max(ellbins))+1)
bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)

#ds_dcb=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=True,temp=False)
#ds_dcb=qml.compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=True,temp=False)
ds_dcb=qml.compute_ds_dcb_parpix(ellbins,nside,ipok,bl,polar=True,temp=False,nprocs=6)


signoise=1e-5
mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
themapi=mapi.copy()
themapi[mask]=0
themapi[~mask]+=np.random.randn(len(themapi[~mask]))*signoise
themapi[~mask]-=np.mean(themapi[~mask])
themapq=mapq.copy()
themapq[mask]=0
themapq[~mask]+=np.random.randn(len(themapq[~mask]))*signoise
# do not remove mean from QU !!! #themapq[~mask]-=np.mean(themapq[~mask])
themapu=mapu.copy()
themapu[mask]=0
themapu[~mask]+=np.random.randn(len(themapu[~mask]))*signoise
# do not remove mean from QU !!! #themapu[~mask]-=np.mean(themapu[~mask])
themaps=[themapq,themapu]


themapi2=mapi.copy()
themapi2[mask]=0
themapi2[~mask]+=np.random.randn(len(themapi2[~mask]))*signoise
themapi2[~mask]-=np.mean(themapi2[~mask])
themapq2=mapq.copy()
themapq2[mask]=0
themapq2[~mask]+=np.random.randn(len(themapq2[~mask]))*signoise
# do not remove mean from QU !!! #themapq2[~mask]-=np.mean(themapq2[~mask])
themapu2=mapu.copy()
themapu2[mask]=0
themapu2[~mask]+=np.random.randn(len(themapu2[~mask]))*signoise
# do not remove mean from QU !!! #themapu2[~mask]-=np.mean(themapu2[~mask])
themaps2=[themapq2,themapu2]

#hp.mollview(mapq,rot=[0,90],title='Q')
#hp.mollview(mapu,rot=[0,90],title='U')
#hp.gnomview(themaps[0],rot=[0,90],reso=10,title='Q')
#hp.gnomview(themaps[1],rot=[0,90],reso=10,title='U')

#hp.gnomview(themaps2[0],rot=[0,90],reso=10,title='Q')
#hp.gnomview(themaps2[1],rot=[0,90],reso=10,title='U')

covmap=np.identity(len(ipok)*len(themaps))*signoise**2*0
guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],0]
specout,error,invfisher,ds_dcb=qml.qml_cross_noiter(themaps, themaps2,mask,covmap,ellbins,fwhmrad,guess,
    ds_dcb,spectra,cholesky=False,temp=False,polar=True,plot=True)


#### MC
nbmc = 100
signoise=0.2
allspec = []
allerr = []
for i in xrange(nbmc):
    mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
    themapi=mapi.copy()
    themapi[mask]=0
    themapi[~mask]+=np.random.randn(len(themapi[~mask]))*signoise
    themapi[~mask]-=np.mean(themapi[~mask])
    themapq=mapq.copy()
    themapq[mask]=0
    themapq[~mask]+=np.random.randn(len(themapq[~mask]))*signoise
    # do not remove mean from QU !!! #themapq[~mask]-=np.mean(themapq[~mask])
    themapu=mapu.copy()
    themapu[mask]=0
    themapu[~mask]+=np.random.randn(len(themapu[~mask]))*signoise
    # do not remove mean from QU !!! #themapu[~mask]-=np.mean(themapu[~mask])
    themaps=[themapq,themapu]


    themapi2=mapi.copy()
    themapi2[mask]=0
    themapi2[~mask]+=np.random.randn(len(themapi2[~mask]))*signoise
    themapi2[~mask]-=np.mean(themapi2[~mask])
    themapq2=mapq.copy()
    themapq2[mask]=0
    themapq2[~mask]+=np.random.randn(len(themapq2[~mask]))*signoise
    # do not remove mean from QU !!! #themapq2[~mask]-=np.mean(themapq2[~mask])
    themapu2=mapu.copy()
    themapu2[mask]=0
    themapu2[~mask]+=np.random.randn(len(themapu2[~mask]))*signoise
    # do not remove mean from QU !!! #themapu2[~mask]-=np.mean(themapu2[~mask])
    themaps2=[themapq2,themapu2]

    covmap=np.identity(len(ipok)*len(themaps))*signoise**2*0
    guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],0]
    specout,error,invfisher,ds_dcb=qml.qml_cross_noiter(themaps, themaps2,mask,covmap,ellbins,fwhmrad,guess,ds_dcb,spectra,cholesky=True,temp=False,polar=True,plot=True)
    allspec.append(specout)
    allerr.append(error)


ell=spectra[0]
clE=spectra[2]
clB=spectra[3]
ellb=specout[0]

thecee = np.zeros((nbmc, nbins))
theerrcee = np.zeros((nbmc, nbins))
thecbb = np.zeros((nbmc, nbins))
theerrcbb = np.zeros((nbmc, nbins))
clf()
for i in xrange(nbmc):
    specout = allspec[i]
    err = allerr[i]
    thecee[i,:] = specout[2]
    thecbb[i,:] = specout[3]
    theerrcee[i,:] = err[2]
    theerrcbb[i,:] = err[3]
    subplot(1,2,1)
    mp.plot(ell,clE*(ell*(ell+1))/(2*np.pi),lw=3)
    errorbar(specout[0], specout[2]*ellb*(ellb+1)/(2*np.pi), yerr=err[2]*ellb*(ellb+1)/(2*np.pi),fmt='ro')
    xlim(0,200)
    ylim(-1,2)
    subplot(1,2,2)
    errorbar(specout[0], specout[3]*ellb*(ellb+1)/(2*np.pi), yerr=err[3]*ellb*(ellb+1)/(2*np.pi),fmt='bo')
    mp.plot(ell,clB*(ell*(ell+1))/(2*np.pi),lw=3)
    xlim(0,200)
    ylim(-0.01,0.03)


clf()
subplot(1,2,1)
mp.plot(ell,clE*(ell*(ell+1))/(2*np.pi),lw=3)
errorbar(specout[0], np.mean(thecee, axis=0)*ellb*(ellb+1)/(2*np.pi), yerr=np.std(thecee, axis=0)*ellb*(ellb+1)/(2*np.pi), fmt='ro')
xlim(0,200)
ylim(-1,2)
subplot(1,2,2)
mp.plot(ell,clB*(ell*(ell+1))/(2*np.pi),lw=3)
errorbar(specout[0], np.mean(thecbb, axis=0)*ellb*(ellb+1)/(2*np.pi), yerr=np.std(thecbb, axis=0)*ellb*(ellb+1)/(2*np.pi), fmt='ro')
xlim(0,200)
ylim(-0.01,0.03)



def statstr(vec):
    m=np.mean(vec)
    s=np.std(vec)
    return '{0:.4f} +/- {1:.4f}'.format(m,s)


clf()
for i in xrange(nbins):
    subplot(3,3,i+1)
    pullsE = (thecee[:,i]-guess[2][i])/theerrcee[:,i]
    hist(pullsE, bins=10, range=[mean(pullsE) - 4*std(pullsE), mean(pullsE)+4*std(pullsE)], label='EE: '+statstr(pullsE), alpha=0.5)
    pullsB = (thecbb[:,i]-guess[3][i])/theerrcbb[:,i]
    hist(pullsB, bins=10, range=[mean(pullsB) - 4*std(pullsB), mean(pullsB)+4*std(pullsB)], label='BB: '+statstr(pullsB), alpha=0.5)
    legend()





######## Now with TEB
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
# do not remove mean from QU !!! #themapq[~mask]-=np.mean(themapq[~mask])
themapu=mapu.copy()
themapu[mask]=0
themapu[~mask]+=np.random.randn(len(themapu[~mask]))*signoise
# do not remove mean from QU !!! #themapu[~mask]-=np.mean(themapu[~mask])
themaps=[themapi, themapq,themapu]


themapi2=mapi.copy()
themapi2[mask]=0
themapi2[~mask]+=np.random.randn(len(themapi2[~mask]))*signoise
themapi2[~mask]-=np.mean(themapi2[~mask])
themapq2=mapq.copy()
themapq2[mask]=0
themapq2[~mask]+=np.random.randn(len(themapq2[~mask]))*signoise
# do not remove mean from QU !!! #themapq2[~mask]-=np.mean(themapq2[~mask])
themapu2=mapu.copy()
themapu2[mask]=0
themapu2[~mask]+=np.random.randn(len(themapu2[~mask]))*signoise
# do not remove mean from QU !!! #themapu2[~mask]-=np.mean(themapu2[~mask])
themaps2=[themapi2, themapq2,themapu2]

#hp.mollview(mapq,rot=[0,90],title='Q')
#hp.mollview(mapu,rot=[0,90],title='U')
#hp.gnomview(themaps[0],rot=[0,90],reso=10,title='Q')
#hp.gnomview(themaps[1],rot=[0,90],reso=10,title='U')

#hp.gnomview(themaps2[0],rot=[0,90],reso=10,title='Q')
#hp.gnomview(themaps2[1],rot=[0,90],reso=10,title='U')

covmap=np.identity(len(ipok)*len(themaps))*signoise**2
guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],0]
specout,error,invfisher,ds_dcb=qml.qml_cross_noiter(themaps, themaps2,mask,covmap,ellbins,fwhmrad,guess,
    ds_dcb,spectra,cholesky=True,temp=True,polar=True,plot=True)















