from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
from Quad import qml
from Quad import pyquad
import healpy as hp
import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

#### Get input Power spectra
################# Input Power spectrum ###################################
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
import pycamb
H0 = 67.04
omegab = 0.022032
omegac = 0.12038
h2 = (H0/100.)**2
scalar_amp = np.exp(3.098)/1.E10
omegav = h2 - omegab - omegac
Omegab = omegab/h2
Omegac = omegac/h2
print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',omegav/h2

params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':0.05,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}

lmaxcamb = 3*512
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
ell = np.arange(1,lmaxcamb+1)
fact = (ell*(ell+1))/(2*np.pi)
spectra = [ell, T/fact, E/fact, B/fact, X/fact]


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

fwhmrad=0.006853589624526168
mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
hp.mollview(mapi)
hp.mollview(mapq)
hp.mollview(mapu)
maps=[mapi,mapq,mapu]






#### EB Only
ds_dcb=0

### Case QUBIC coverage
#okpix = np.loadtxt('ipok.txt')
#mask=(np.arange(12*nside**2) >=0)
#for px in okpix: mask[px] = False
#maskok=~mask


### case first pixels
nbpixok=1000
mask=(np.arange(12*nside**2) >= nbpixok)
maskok=~mask



ip=np.arange(12*nside**2)
ipok=ip[~mask]

#ellbins=[0,50,100,150,200,3*nside]
#ellbins = [0,50,70, 90,110,130,150, 170, 190, 210,4*nside]
ellbins = [0,50,75,100,125,150,175,200,225]
nbins=len(ellbins)-1
ellmin=np.array(ellbins[0:nbins])
ellmax=np.array(ellbins[1:nbins+1])-1
ellval=(ellmin+ellmax)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)

reload(qml)
ll=np.arange(int(np.max(ellbins))+1)
bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)
import time

#ds_dcb=qml.compute_ds_dcb(ellbins,nside,ipok,bl,polar=True,temp=False)
#ds_dcb=qml.compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=True,temp=False)
init = time.time()
ds_dcb=qml.compute_ds_dcb_parpix(ellbins,nside,ipok,bl,polar=True,temp=False,nprocs=8)
fin = time.time()
print(fin-init)

# init1 = time.time()
# ds_dcb_new=qml.compute_ds_dcb_parpix_direct(ellbins,nside,ipok,bl,polar=True,temp=False,nprocs=8)
# fin1 = time.time()
# print(fin1-init1)

# clf()
# ider=1
# for i in np.arange(len(ellbins)-1):
#     subplot(3,3,i+1)
#     imshow(ds_dcb[ider,i,:,:]-ds_dcb_new[ider,i,:,:],interpolation='nearest')
#     colorbar()

# np.min(ds_dcb-ds_dcb_new)
# np.max(ds_dcb-ds_dcb_new)




signoise=0.15
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

#hp.write_map('/Users/hamilton/CMB/Interfero/Quad/mapq1_ok.fits', themapq)
#hp.write_map('/Users/hamilton/CMB/Interfero/Quad/mapu1_ok.fits', themapu)
#hp.write_map('/Users/hamilton/CMB/Interfero/Quad/mapq2_ok.fits', themapq2)
#hp.write_map('/Users/hamilton/CMB/Interfero/Quad/mapu2_ok.fits', themapu2)



# covmap=np.identity(len(ipok)*len(themaps))*signoise**2
# guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],0]
# specout,error,invfisher,lk,num,ds_dcb=qml.qml(themaps,mask,covmap,ellbins,fwhmrad,guess,ds_dcb,spectra,
# 	cholesky=True,temp=False,polar=True,plot=True,itmax=20)


ion()
covmap=np.identity(len(ipok)*len(themaps))*signoise**2
guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],0]
specout,error,invfisher,lk, num,ds_dcb, conv=qml.qml_cross(themaps, themaps,mask,covmap,ellbins,fwhmrad,guess,
    ds_dcb,spectra,cholesky=False,temp=False,polar=True,plot=True)

ion()
covmap=np.identity(len(ipok)*len(themaps))*signoise**2
guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],0]
specout,error,invfisher,ds_dcb=qml.qml_cross(themaps, themaps2,mask,covmap,ellbins,fwhmrad,guess,
    ds_dcb,spectra,cholesky=False,temp=False,polar=True,plot=True)


#### MC
import string
import random
def random_string(nchars):
    lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(nchars)]
    str = "".join(lst)
    return(str)

rep = '/Volumes/Data/Qubic/Quad/QuadMC/MC08012016/'
allvalnoise = [0., 0.05, 0.1, 0.015, 0.2]
nbmc = 1000
allspec = []
allerr = []
for i in xrange(nbmc):
	mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
	thename = random_string(10)
	for signoise in allvalnoise:
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
		specout[1] = specout[0]*0
		specout[4] = specout[0]*0
		specout[5] = specout[0]*0
		specout[6] = specout[0]*0
		error[1] = specout[0]*0
		error[4] = specout[0]*0
		error[5] = specout[0]*0
		error[6] = specout[0]*0 
		savetxt(rep+'cl_noise{}_{}.txt'.format(signoise,thename), np.array(specout).T,fmt='%10.5g')
		savetxt(rep+'errcl_noise{}_{}.txt'.format(signoise,thename), np.array(error).T,fmt='%10.5g')

#### rewrite them
# for s in xrange(len(allvalnoise)):
#     for i in xrange(nbmc):
#         specout = allspec[s][i]
#         error = allerr[s][i]
#         specout[1] = specout[0]*0
#         specout[4] = specout[0]*0
#         specout[5] = specout[0]*0
#         specout[6] = specout[0]*0
#         error[1] = specout[0]*0
#         error[4] = specout[0]*0
#         error[5] = specout[0]*0
#         error[6] = specout[0]*0
#         savetxt(rep+'cl_noise{}_mc{}.txt'.format(allvalnoise[s],i), np.array(specout).T,fmt='%10.5g')
#         savetxt(rep+'errcl_noise{}_mc{}.txt'.format(allvalnoise[s],i), np.array(error).T,fmt='%10.5g')


#### read them back
theell = np.zeros(nbins)
thecee = np.zeros((len(allvalnoise), nbmc, nbins))
theerrcee = np.zeros((len(allvalnoise), nbmc, nbins))
thecbb = np.zeros((len(allvalnoise), nbmc, nbins))
theerrcbb = np.zeros((len(allvalnoise), nbmc, nbins))
thedee = np.zeros((len(allvalnoise), nbmc, nbins))
thedbb = np.zeros((len(allvalnoise), nbmc, nbins))
for s in xrange(len(allvalnoise)):
    signoise = allvalnoise[s]
    for i in xrange(nbmc):
        specout = loadtxt(rep+'cl_noise{}_mc{}.txt'.format(signoise,i))
        error = loadtxt(rep+'errcl_noise{}_mc{}.txt'.format(signoise,i))
        theell = specout[:,0]
        thecee[s, i, :] = specout[:,2]
        thecbb[s, i, :] = specout[:,3]
        thedee[s, i, :] = specout[:,2]*theell*(theell+1)/(2*np.pi)
        thedbb[s, i, :] = specout[:,3]*theell*(theell+1)/(2*np.pi)
        theerrcee[s, i, :] = error[:,2]
        theerrcbb[s, i, :] = error[:,3]


### covariance matrices
alldata = np.append(thedee,thedbb,axis=2)
allcovmat = np.zeros((len(allvalnoise), nbins*2, nbins*2))
allcorrmat = np.zeros((len(allvalnoise), nbins*2, nbins*2))
sigdee = np.zeros((len(allvalnoise), nbins)) 
sigdbb = np.zeros((len(allvalnoise), nbins)) 
for i in xrange(len(allvalnoise)):
    allcovmat[i,:,:] = np.cov(alldata[i,:,:].T)
    allcorrmat[i,:,:] = np.corrcoef(alldata[i,:,:].T)
    sigdee[i,:] = np.std(thedee[i, :,:], axis=0)
    sigdbb[i,:] = np.std(thedbb[i, :,:], axis=0)

num=3
clf()
subplot(2,2,1)
mp.plot(ell,cee*(ell*(ell+1))/(2*np.pi),lw=3)
errorbar(theell, np.mean(thedee[num, :,:], axis=0), yerr=np.std(thedee[num, :,:], axis=0), fmt='ro')
xlim(0,200)
ylim(-1,2)
subplot(2,2,2)
mp.plot(ell,cbb*(ell*(ell+1))/(2*np.pi),lw=3)
errorbar(theell, np.mean(thedbb[num, :,:], axis=0), yerr=np.std(thedbb[num, :,:], axis=0), fmt='ro')
xlim(0,200)
ylim(-0.01,0.03)
subplot(2,2,3)
imshow(np.log(np.abs(allcovmat[num,:,:])), interpolation='nearest')
plot([0-0.5,2*nbins-0.5], [nbins-0.5,nbins-0.5], 'k')
plot([nbins-0.5,nbins-0.5], [0-0.5,2*nbins-0.5], 'k')
xlim(-0.5, 2*nbins-0.5)
ylim(-0.5, 2*nbins-0.5)
colorbar()
subplot(2,2,4)
imshow(allcorrmat[num,:,:], interpolation='nearest')
plot([0-0.5,2*nbins-0.5], [nbins-0.5,nbins-0.5], 'k')
plot([nbins-0.5,nbins-0.5], [0-0.5,2*nbins-0.5], 'k')
xlim(-0.5, 2*nbins-0.5)
ylim(-0.5, 2*nbins-0.5)
colorbar()


clf()
subplot(1,2,1)
title('EE: SigCl - SigClSample')
x,y = np.meshgrid(theell, allvalnoise)
pcolor(x,y, sigdee-sigdee[0,:]) 
xlabel('$\ell$')
ylabel('Noise RMS')
colorbar()
subplot(1,2,2)
title('BB: SigCl - SigClSample')
x,y = np.meshgrid(theell, allvalnoise)
pcolor(x,y, sigdbb-sigdbb[0,:]) 
xlabel('$\ell$')
ylabel('Noise RMS')
colorbar()

clf()
subplot(1,2,1)
for i in xrange(len(allvalnoise)):
    plot(theell, sigdee[i,:], label='$\sigma$ = {}'.format(allvalnoise[i]), color=cm.jet(allvalnoise[i]/max(allvalnoise)))
legend(fontsize=8)
title('EE Total RMS')
xlabel('$\ell$')
ylabel('$\Delta \cal{D}_\ell$')
subplot(1,2,2)
for i in xrange(len(allvalnoise)):
    plot(theell, sigdbb[i,:], label='$\sigma$ = {}'.format(allvalnoise[i]), color=cm.jet(allvalnoise[i]/max(allvalnoise)))
legend(fontsize=8)
title('BB Total RMS')
xlabel('$\ell$')
ylabel('$\Delta \cal{D}_\ell$')


##### Sample Variance
sigcosvardee = sigdee[0,:]
sigcosvardbb = sigdbb[0,:]
deltacl_ee_like = theerrcee[0,0,:] * theell * (theell +1) / (2*pi)
deltacl_bb_like = theerrcbb[0,0,:] * theell * (theell +1) / (2*pi)
fsky = np.sum(maskok) / len(mask)
samplevar_th_ee = np.interp(theell, ell, cee*(ell*(ell+1))/(2*np.pi) * np.sqrt(2./((2*ell+1)*fsky))) / np.sqrt(ellmax-ellmin)
samplevar_th_bb = np.interp(theell, ell, cbb*(ell*(ell+1))/(2*np.pi) * np.sqrt(2./((2*ell+1)*fsky))) / np.sqrt(ellmax-ellmin)
clf()
plot(theell, sigcosvardee, 'ro', label = 'BB: MC RMS')
plot(theell, samplevar_th_ee, 'r', label = 'EE: Knox')
plot(theell, deltacl_ee_like, 'r--', label = 'EE: Likelihood')
plot(theell, sigcosvardbb, 'bo', label = 'BB: MC RMS')
plot(theell, samplevar_th_bb, 'b', label = 'BB: Knox')
plot(theell, deltacl_bb_like, 'b--', label = 'BB: Likelihood')
yscale('log')
legend(numpoints=1)

clf()
plot(theell, sigcosvardee / samplevar_th_ee, 'r', label = 'EE: MC / Knox')
plot(theell, sigcosvardee / deltacl_ee_like, 'r--', label = 'EE: MC / Likelihood')
plot(theell, sigcosvardbb / samplevar_th_bb, 'b', label = 'BB: MC / Knox')
plot(theell, sigcosvardbb / deltacl_bb_like, 'b--', label = 'BB: MC / Likelihood')
ylim(0,2)
legend(numpoints=1)




##### Noise variance
signoisedee = sigdee-sigdee[0,:]
signoisedbb = sigdbb-sigdbb[0,:]

pixwin=hp.pixwin(nside)[0:max(ll)+1]
omega = 4* pi * fsky

fact = 2 * np.interp(theell, ell, np.sqrt(2./((2*ell+1)*fsky))) / np.sqrt(ellmax-ellmin) * theell * (theell +1) / (2*pi) / np.interp(theell, ll,bl**2*pixwin**2) * omega

clf()
subplot(1,2,1)
for i in xrange(4, len(allvalnoise)):
    plot(theell, signoisedee[i,:], label='$\sigma$ = {}'.format(allvalnoise[i]), color=cm.jet(allvalnoise[i]/max(allvalnoise)))
    plot(theell, (fact * allvalnoise[i]**2)/100, '--', color=cm.jet(allvalnoise[i]/max(allvalnoise)))
legend(fontsize=8, loc='upper left')
yscale('log')
subplot(1,2,2)
for i in xrange(4, len(allvalnoise)):
    plot(theell, signoisedbb[i,:], label='$\sigma$ = {}'.format(allvalnoise[i]), color=cm.jet(allvalnoise[i]/max(allvalnoise)))
    plot(theell, (fact * allvalnoise[i]**2)/1000, '--', color=cm.jet(allvalnoise[i]/max(allvalnoise)))
yscale('log')
legend(fontsize=8, loc='upper left')












