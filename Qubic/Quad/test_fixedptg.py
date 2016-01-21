from __future__ import division

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os


from pyoperators import pcg, DiagonalOperator, UnpackOperator
from pysimulators import ProjectionInMemoryOperator
from qubic import QubicConfiguration, QubicInstrument, create_random_pointings
from pyquad import pyquad

path = os.path.dirname('/Users/hamilton/idl/pro/Qubic/People/pierre/qubic/script/script_ga.py')


signoise=0.1
nbmc=10000
correc=True

####################### restore stuff ###################
import pickle
from pysimulators import FitsArray
## Pointing
infile=open('saved_ptg.dat', 'rb')
data=pickle.load(infile)
infile.close()
pointings=data['pointings']
oldmask=data['mask']
#signoise=data['signoise']
data=0
## Covariance matrix
covmc=FitsArray('covmc'+str(signoise)+'_'+str(nbmc)+'.dat')
## ell range
infile=open('saved_ellrange_linlog.dat', 'rb')
data=pickle.load(infile)
infile.close()
minell=data['minell']
deltal=data['deltal']
ellbins=data['ellbins']
maxell=data['maxell']
ells=data['ells']
ellmin=data['ellmin']
ellmax=data['ellmax']
ellval=data['ellval']
deltaell=data['deltaell']
data=0
## dS/dCb
ds_dcb=FitsArray('ds_dcb_linlog.dat')
###########################################################




#### Get Power spectra
#a=np.loadtxt('/Volumes/Data/Qubic/qubic_v1/cl_r=0.1bis2.txt')
a=np.loadtxt('./cl_r=0.1bis2.txt')
ell=a[:,0]
ctt=np.concatenate([[0,0],a[:,1]*1e12*2*np.pi/(ell*(ell+1))])
cee=np.concatenate([[0,0],a[:,2]*1e12*2*np.pi/(ell*(ell+1))])
cte=np.concatenate([[0,0],a[:,4]*1e12*2*np.pi/(ell*(ell+1))])
cbb=np.concatenate([[0,0],a[:,7]*1e12*2*np.pi/(ell*(ell+1))])
ell=np.concatenate([[0,1],ell])

#cbb[ell>384]=0

spectra=[ell,ctt,cte,cee,cbb]

#clf()
#plot(ell,np.sqrt(spectra[1]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TT}$')
#plot(ell,np.sqrt(abs(spectra[2])*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TE}$')
#plot(ell,np.sqrt(spectra[3]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{EE}$')
#plot(ell,np.sqrt(spectra[4]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{BB} (r=0.1)$')
#yscale('log')
#xlim(0,600)
#ylim(0.01,100)
#xlabel('$\ell$')
#ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
#legend(loc='lower right',frameon=False)




#### Generate map for B-modes directly
nside=128
map_orig=hp.synfast(spectra[4],nside,fwhm=0,pixwin=True)
input_map=map_orig.copy()
#hp.mollview(input_map)
# we know that the map has fwhm 0.6488deg (from Pierre)
fwhm=0.6488*np.pi/180
# but fitting a point source map after mapmaking leads FWHM=0.64810
#fwhm=0.64810*np.pi/180

#### QUBIC Instrument
kmax = 2
qubic = QubicInstrument('monochromatic,nopol',nside=128)

#### configure observation
obs = QubicConfiguration(qubic, pointings)
C = obs.get_convolution_peak_operator()
P = obs.get_projection_peak_operator(kmax=kmax)
H = P * C

# Produce the Time-Ordered data
tod = H(input_map)

# Add noise
ndet=tod.shape[0]
nsamples=tod.shape[1]

tod = tod + np.random.randn(ndet,nsamples)*signoise

# map-making
coverage = P.T(np.ones_like(tod))
covmin=10
mask=coverage < covmin

P.matrix.pack(mask)
P_packed = ProjectionInMemoryOperator(P.matrix)
unpack = UnpackOperator(mask)
solution = pcg(P_packed.T * P_packed, P_packed.T(tod), M=DiagonalOperator(1/coverage[~mask]), disp=True)
output_map = unpack(solution['x'])

# some display
orig = input_map.copy()
orig[mask] = np.nan
#hp.gnomview(orig, rot=[0,90], reso=5, xsize=600, min=-0.5, max=0.5, title='Original map')
cmap = C(input_map)
cmap[mask] = np.nan
#hp.gnomview(cmap, rot=[0,90], reso=5, xsize=600, min=-0.5, max=0.5, title='Convolved original map')
output_map[mask] = np.nan
#hp.gnomview(output_map, rot=[0,90], reso=5, xsize=600, min=-0.5, max=0.5, title='Reconstructed map (simulpeak)')

#### select the covered area
maskok = np.isfinite(output_map)
npix=output_map[maskok].size
print(npix)


# binned input spectrum
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)
inputspectrum=binspec

#### build submatrix for covariance accounting for the new mask
covpixinit=coverage[~oldmask]
newindices=covpixinit >= covmin
covmc=covmc[newindices,:]
covmc=covmc[:,newindices]
npixnew=coverage[~mask].size
newds_dcb=np.zeros((ellbins,npixnew,npixnew))
for i in np.arange(ellbins):
    bla=ds_dcb[i,newindices,:]
    bla=bla[:,newindices]
    newds_dcb[i,:,:]=bla

ds_dcb=newds_dcb.copy()


# quadratic estimator
covmap=covmc
extent='_nocorr'
if correc:
    ns=nbmc
    nd=npixnew
    correction=(ns-1)*1./(ns-nd-2)
    print('Applying correction '+str(correction)+' on covariance matrix')
    covmap=covmc*correction
    extent='_corr'

guess=inputspectrum[:,4]*0.5
thespectrum,err,invfisher,lk,num=pyquad.quadest(output_map,maskok,covmap,ellmin,ellmax,fwhm,guess,spectra[4],ds_dcb,itmax=10,plot=True,cholesky=True)

result=np.zeros((4,ellbins))
result[0,:]=ellval
result[1,:]=thespectrum[:,num+1]
result[2,:]=(ellmax+1-ellmin)/2
result[3,:]=err

mp.clf()
ell=np.arange(spectra[0].size)+2
mp.plot(spectra[0],spectra[4]*(spectra[0]*(spectra[0]+1))/(2*np.pi),lw=3)
mp.xlabel('$\ell$')
mp.ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')
mp.xlim(0,np.max(ellmax)*1.2)
mp.errorbar(result[0,:],result[1,:]*result[0,:]*(result[0,:]+1)/(2*np.pi),result[3,:]*result[0,:]*(result[0,:]+1)/(2*np.pi),xerr=result[2,:],fmt='ro')


from pysimulators import FitsArray
import os
rep='/Volumes/Data/Qubic/PyQuad/ClRes/Cl_linlog_FullMap_mc'+str(nbmc)+'_noise'+str(signoise)+extent+'/'
if not os.path.isdir(rep): os.mkdir(rep)
filename=rep+'clresults_'+pyquad.random_string(10)+'.fits'
FitsArray(result,copy=False).save(filename)

