from __future__ import division

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os

from Quad import pyquad

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

#### Generate map for B-modes directly
nside=128
map_orig=hp.synfast(spectra[4],nside,fwhm=0,pixwin=True)
input_map=map_orig.copy()

#### QUBIC Instrument
kmax = 2
qubic = QubicInstrument('monochromatic,nopol',nside=128)

#### restore pointing
import pickle
## Pointing
infile=open('saved_ptg.dat', 'rb')
data=pickle.load(infile)
infile.close()
pointings=data['pointings']
oldmask=data['mask']
signoise=data['signoise']
data=0


#### configure observation
obs = QubicConfiguration(qubic, pointings)
C = obs.get_convolution_peak_operator()
P = obs.get_projection_peak_operator(kmax=kmax)
H = P * C

# Produce the Time-Ordered data
tod = H(input_map)

# Add noise
signoise=0.02
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

#### noise covariance matrix
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
    pyquad.progress_bar(i,npix)
    mm0=allnoisemaps[i,:]-mm[i]
    for j in np.arange(i,npix):
        covmc[i,j]=np.mean( mm0*(allnoisemaps[j,:]-mm[j]))
        covmc[j,i]=covmc[i,j]

cormc=pyquad.cov2cor(covmc)

from pysimulators import FitsArray 
FitsArray(covmc,copy=False).save('covmc'+str(signoise)+'_'+str(nbmc)+'.dat')
FitsArray(cormc,copy=False).save('cormc'+str(signoise)+'_'+str(nbmc)+'.dat')






