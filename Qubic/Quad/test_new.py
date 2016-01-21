from __future__ import division

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os

from pyquad import pyquad

from pyoperators import pcg, DiagonalOperator, UnpackOperator
from pysimulators import ProjectionInMemoryOperator
from qubic import QubicConfiguration, QubicInstrument, create_random_pointings

path = os.path.dirname('/Users/hamilton/idl/pro/Qubic/People/pierre/qubic/script/script_ga.py')


#### Get Power spectra
#a=np.loadtxt('/Volumes/Data/Qubic/qubic_v1/cl_r=0.1bis2.txt')
a=np.loadtxt('./cl_r=0.1bis2.txt')
ell=a[:,0]
ctt=np.concatenate([[0,0],a[:,1]*1e12*2*np.pi/(ell*(ell+1))])
cee=np.concatenate([[0,0],a[:,2]*1e12*2*np.pi/(ell*(ell+1))])
cte=np.concatenate([[0,0],a[:,4]*1e12*2*np.pi/(ell*(ell+1))])
cbb=np.concatenate([[0,0],a[:,7]*1e12*2*np.pi/(ell*(ell+1))])
ell=np.concatenate([[0,1],ell])
spectra=[ell,ctt,cte,cee,cbb]

clf()
plot(ell,np.sqrt(spectra[1]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TT}$')
plot(ell,np.sqrt(abs(spectra[2])*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TE}$')
plot(ell,np.sqrt(spectra[3]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{EE}$')
plot(ell,np.sqrt(spectra[4]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{BB} (r=0.1)$')
yscale('log')
xlim(0,250)
ylim(0.01,100)
xlabel('$\ell$')
ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
legend(loc='lower right',frameon=False)


#### Generate map for B-modes directly
nside=128
map_orig=hp.synfast(spectra[4],nside,fwhm=0,pixwin=True)
input_map=map_orig.copy()
hp.mollview(input_map)
# we know that the map has fwhm 0.6488deg (from Pierre)
#fwhm=0.6488*np.pi/180
# but fitting a point source map after mapmaking leads FWHM=0.64810
fwhm=0.64810*np.pi/180

#### QUBIC Instrument
kmax = 2
qubic = QubicInstrument('monochromatic,nopol',nside=128)

#### Pointing
#pointings = create_random_pointings(1000, 10)
import pickle
from pysimulators import FitsArray
infile=open('saved_ptg.dat', 'rb')
data=pickle.load(infile)
infile.close()
pointings=data['pointings']




#### configure observation
obs = QubicConfiguration(qubic, pointings)
C = obs.get_convolution_peak_operator()
P = obs.get_projection_peak_operator(kmax=kmax)
H = P * C

# Produce the Time-Ordered data
tod = H(input_map)

# Add noise
signoise=0.1
ndet=tod.shape[0]
nsamples=tod.shape[1]

tod = tod + np.random.randn(ndet,nsamples)*signoise

# map-making
coverage = P.T(np.ones_like(tod))
mask = coverage < 10
P.matrix.pack(mask)
P_packed = ProjectionInMemoryOperator(P.matrix)
unpack = UnpackOperator(mask)
solution = pcg(P_packed.T * P_packed, P_packed.T(tod), M=DiagonalOperator(1/coverage[~mask]), disp=True)
output_map = unpack(solution['x'])



# some display
orig = input_map.copy()
orig[mask] = np.nan
hp.gnomview(orig, rot=[0,90], reso=5, xsize=600, min=-0.5, max=0.5, title='Original map')
cmap = C(input_map)
cmap[mask] = np.nan
hp.gnomview(cmap, rot=[0,90], reso=5, xsize=600, min=-0.5, max=0.5, title='Convolved original map')
output_map[mask] = np.nan
hp.gnomview(output_map, rot=[0,90], reso=5, xsize=600, min=-0.5, max=0.5, title='Reconstructed map (simulpeak)')

#### select the covered area
maskok = np.isfinite(output_map)
npix=output_map[maskok].size
print(npix)

#### noise covariance matrix
# naive
covmap = np.diag(signoise**2/coverage[maskok])

# get it through MC
nbmc=10000
allnoisemaps=np.zeros((npix,nbmc))
for i in np.arange(nbmc):
    print(i,nbmc)
    todnoise = tod*0 + np.random.randn(ndet,nsamples)*signoise
    solution = pcg(P_packed.T * P_packed, P_packed.T(todnoise), M=DiagonalOperator(1/coverage[~mask]))
    allnoisemaps[:,i]=unpack(solution['x'])[~mask]

covmc=np.zeros((npix,npix))
mm=np.mean(allnoisemaps,axis=-1)
for i in np.arange(npix):
    print(i)
    mm0=allnoisemaps[i,:]-mm[i]
    for j in np.arange(i,npix):
        covmc[i,j]=mean( mm0*(allnoisemaps[j,:]-mm[j]))
        covmc[j,i]=covmc[i,j]

covmap=covmc

######### Linear Bins
# define ell bins
minell=35
deltal=25
ellbins=14
maxell=minell+deltal*ellbins

# binned input spectrum
ells=np.linspace(minell,maxell,ellbins+1)
ellmin=ells[0:ellbins]
ellmin[0]=2
ellmax=ells[1:ellbins+1]-1
ellval=(ellmin+ellmax)/2
deltaell=(ellmax+1-ellmin)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)
inputspectrum=binspec
######################



############ Log Bins
# define ell bins
minell=12
deltal=37
ellbins=8
maxell=minell+deltal*ellbins

# binned input spectrum
ells=np.floor(np.logspace(np.log10(minell),np.log10(maxell),ellbins+1))
ellmin=ells[0:ellbins]
ellmin[0]=2
ellmax=ells[1:ellbins+1]-1
ellval=(ellmin+ellmax)/2
deltaell=(ellmax+1-ellmin)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)
inputspectrum=binspec
######################




########## New bins half loh and half lin
ells=np.array([0.,9,17,27,40,60,85.,110.,135.,160.,185.,210.,235.,260.,285.,310.,335.,360.,385])
ellbins=ells.size-1
minell=ells[0]
maxell=max(ells)
ellmin=ells[0:ellbins]
ellmax=ells[1:ellbins+1]-1
ellval=(ellmin+ellmax)/2
deltaell=(ellmax+1-ellmin)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)
inputspectrum=binspec


mp.clf()
mp.plot(spectra[0],spectra[4]*(spectra[0]*(spectra[0]+1))/(2*np.pi),lw=3)
mp.xlabel('$\ell$')
mp.xscale('log')
mp.ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')
mp.xlim(0,np.max(ellmax)*1.2)
errorbar(binspec[:,0],binspec[:,4]*binspec[:,0]*(binspec[:,0]+1)/(2*np.pi),xerr=deltaell,fmt='ro')

# window functions
ds_dcb=pyquad.compute_ds_dcb_line_par(output_map,maskok,ellmin,ellmax,fwhm,24)

####### Save results #######################
from pysimulators import FitsArray
import pickle
## Covariance matrix
#FitsArray(covmc,copy=False).save('covmc.dat')
## Pointing
#out=open('saved_ptg.dat','wb')
    #cPickle.dump({'pointings':pointings, 'cmap':cmap, 'mask':mask,
#             'signoise':signoise},out)
#out.close()
## dS/dCb
FitsArray(ds_dcb,copy=False).save('ds_dcb_linlog.dat')
## ell range
out=open('saved_ellrange_linlog.dat','wb')
pickle.dump({'minell':minell, 'deltal':deltal,
            'ellbins':ellbins, 'maxell':maxell,'ells':ells,
            'ellmin':ellmin, 'ellmax':ellmax, 'ellval':ellval, 'deltaell':deltaell},out)
out.close()
############################################

# quadratic estimator
guess=inputspectrum[:,4]*0.5
thespectrum,err,invfisher,lk,num=pyquad.quadest(output_map,maskok,covmap,ellmin,ellmax,fwhm,guess,spectra[4],ds_dcb,itmax=100,plot=True,cholesky=True)

clf()
ell=np.arange(spectra[0])+2
plot(ell,spectra[4]*(ell*(ell+1))/(2*np.pi),lw=3)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')
xlim(0,np.max(ellmax)*1.2)
errorbar(ellval,thespectrum[:,num+1]*ellval*(ellval+1)/(2*np.pi),err*ellval*(ellval+1)/(2*np.pi),xerr=(ellmax+1-ellmin)/2,fmt='ro')


figure()
guess=inputspectrum[:,4]*0.
thespectrum,err,invfisher,lk,num=pyquad.quadest(output_map,maskok,covmap,ellmin,ellmax,fwhm,guess,spectra[4],ds_dcb,itmax=100,plot=True,cholesky=True)





