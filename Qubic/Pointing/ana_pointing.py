from __future__ import division
from pylab import *
import healpy as hp
from matplotlib.pyplot import *
import numpy as np
import os
import sys
import qubic
import pycamb
from pyoperators import DenseBlockDiagonalOperator, Rotation3dOperator
from pysimulators import FitsArray
from pyoperators import MPI
import glob
from qubic import (
    QubicAcquisition, create_sweeping_pointings, equ2gal, map2tod, tod2map_all,
    tod2map_each, create_random_pointings, QubicInstrument)




############# Input Power spectrum ##############################################
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
         'tensor_ratio':0.1,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}

lmaxcamb = 3*512
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
lll = np.arange(1,lmaxcamb+1)
fact = (lll*(lll+1))/(2*np.pi)
spectra = [lll, T/fact, E/fact, B/fact, X/fact]
##################################################################################



ns = 256
noise = 1.0
sigptgvals = np.array([1., 3., 10., 30., 100.])
rep = 'Nersc/'
#noI = 'noI_'
noI = ''

from XPol import XPol

mapang = None
apodize_fwhm = 1.
maskmap = None
wl = None
Mllmat = None
MllBinned = None
ellpq = None
MllBinnedInv = None
nbins = np.int((2*ns-25)/25)


clinit = np.zeros((len(sigptgvals),6,nbins))
clnoiseless = np.zeros((len(sigptgvals),6,nbins))
clnoisy = np.zeros((len(sigptgvals),6,nbins))
clres = np.zeros((len(sigptgvals),6,nbins))
clspoiled_noiseless = np.zeros((len(sigptgvals),6,nbins))
clspoiled_noisy = np.zeros((len(sigptgvals),6,nbins))
clspoiled_res = np.zeros((len(sigptgvals),6,nbins))
dclinit = np.zeros((len(sigptgvals),6,nbins))
dclnoiseless = np.zeros((len(sigptgvals),6,nbins))
dclnoisy = np.zeros((len(sigptgvals),6,nbins))
dclres = np.zeros((len(sigptgvals),6,nbins))
dclspoiled_noiseless = np.zeros((len(sigptgvals),6,nbins))
dclspoiled_noisy = np.zeros((len(sigptgvals),6,nbins))
dclspoiled_res = np.zeros((len(sigptgvals),6,nbins))

for j in np.arange(len(sigptgvals)):
	print('doing PointingSigma = '+np.str(sigptgvals[j]))
	basestr = 'ns'+str(ns)+'_noise'+np.str(noise)+'_sigptg'+np.str(sigptgvals[j])+'_'
	allreal = glob.glob(rep+'maps_'+basestr+noI+'input_*.fits')
	strrnd = []
	for tt in allreal: 
		blo = tt.split('_')[-1].split('.')[0]
		toto = glob.glob(rep+'*'+blo+'*.fits')
		if len(toto)==12: strrnd.append(blo)
	allclinit = np.zeros((6, nbins, len(strrnd)))
	allclnoiseless = np.zeros((6, nbins, len(strrnd)))
	allclnoisy = np.zeros((6, nbins, len(strrnd)))
	allclres = np.zeros((6, nbins, len(strrnd)))
	allclspoiled_noiseless = np.zeros((6, nbins, len(strrnd)))
	allclspoiled_noisy = np.zeros((6, nbins, len(strrnd)))
	allclspoiled_res = np.zeros((6, nbins, len(strrnd)))
	for kk in np.arange(len(strrnd)):
		print('    - Doing '+strrnd[kk])
		### Read the maps
		covmap = qubic.io.read_map(rep+'cov_'+basestr+strrnd[kk]+'.fits')
		spoiled_covmap = qubic.io.read_map(rep+'cov_'+basestr+'spoiled_'+strrnd[kk]+'.fits')
		initmap = qubic.io.read_map(rep+'maps_'+basestr+noI+'input_'+strrnd[kk]+'.fits')
		noiselessmap = qubic.io.read_map(rep+'maps_'+basestr+noI+'noiseless_'+strrnd[kk]+'.fits')
		noisymap = qubic.io.read_map(rep+'maps_'+basestr+noI+'noisy_'+strrnd[kk]+'.fits')
		spoiled_noiselessmap = qubic.io.read_map(rep+'maps_'+basestr+noI+'spoiled_noiseless_'+strrnd[kk]+'.fits')
		spoiled_noisymap = qubic.io.read_map(rep+'maps_'+basestr+noI+'spoiled_noisy_'+strrnd[kk]+'.fits')
		### Get the apodization mask if needed
		if maskmap is None:
			maskok = covmap != 0
			maskmap = XPol.apodize_mask(maskok,apodize_fwhm,mapang=mapang)
		### Get the Cls
		allclinit[:,:,kk], newl, Mllmat, MllBinned, MllBinnedInv, p, q, allcls = XPol.get_spectra(initmap.T, maskmap, 2*ns, 25, 25, wl=wl, Mllmat=Mllmat, MllBinned=MllBinned, ellpq=ellpq, MllBinnedInv=MllBinnedInv)
		allclnoiseless[:,:,kk], newl, Mllmat, MllBinned, MllBinnedInv, p, q, allcls = XPol.get_spectra(noiselessmap.T, maskmap, 2*ns, 25, 25, wl=wl, Mllmat=Mllmat, MllBinned=MllBinned, ellpq=ellpq, MllBinnedInv=MllBinnedInv)
		allclnoisy[:,:,kk], newl, Mllmat, MllBinned, MllBinnedInv, p, q, allcls = XPol.get_spectra(noisymap.T, maskmap, 2*ns, 25, 25, wl=wl, Mllmat=Mllmat, MllBinned=MllBinned, ellpq=ellpq, MllBinnedInv=MllBinnedInv)
		allclres[:,:,kk], newl, Mllmat, MllBinned, MllBinnedInv, p, q, allcls = XPol.get_spectra(noisymap.T-noiselessmap.T, maskmap, 2*ns, 25, 25, wl=wl, Mllmat=Mllmat, MllBinned=MllBinned, ellpq=ellpq, MllBinnedInv=MllBinnedInv)
		allclspoiled_noiseless[:,:,kk], newl, Mllmat, MllBinned, MllBinnedInv, p, q, allcls = XPol.get_spectra(spoiled_noiselessmap.T, maskmap, 2*ns, 25, 25, wl=wl, Mllmat=Mllmat, MllBinned=MllBinned, ellpq=ellpq, MllBinnedInv=MllBinnedInv)
		allclspoiled_noisy[:,:,kk], newl, Mllmat, MllBinned, MllBinnedInv, p, q, allcls = XPol.get_spectra(spoiled_noisymap.T, maskmap, 2*ns, 25, 25, wl=wl, Mllmat=Mllmat, MllBinned=MllBinned, ellpq=ellpq, MllBinnedInv=MllBinnedInv)
		allclspoiled_res[:,:,kk], newl, Mllmat, MllBinned, MllBinnedInv, p, q, allcls = XPol.get_spectra(spoiled_noisymap.T - spoiled_noiselessmap.T, maskmap, 2*ns, 25, 25, wl=wl, Mllmat=Mllmat, MllBinned=MllBinned, ellpq=ellpq, MllBinnedInv=MllBinnedInv)
	########### Make averages and rms
	clinit[j,:,:] = np.mean(allclinit, axis = 2)
	dclinit[j,:,:] = np.std(allclinit, axis = 2)
	clnoiseless[j,:,:] = np.mean(allclnoiseless, axis = 2)
	dclnoiseless[j,:,:] = np.std(allclnoiseless, axis = 2)
	clnoisy[j,:,:] = np.mean(allclnoisy, axis = 2)
	dclnoisy[j,:,:] = np.std(allclnoisy, axis = 2)
	clres[j,:,:] = np.mean(allclres, axis = 2)
	dclres[j,:,:] = np.std(allclres, axis = 2)
	clspoiled_noiseless[j,:,:] = np.mean(allclspoiled_noiseless, axis = 2)
	dclspoiled_noiseless[j,:,:] = np.std(allclspoiled_noiseless, axis = 2)
	clspoiled_noisy[j,:,:] = np.mean(allclspoiled_noisy, axis = 2)
	dclspoiled_noisy[j,:,:] = np.std(allclspoiled_noisy, axis = 2)
	clspoiled_res[j,:,:] = np.mean(allclspoiled_res, axis = 2)
	dclspoiled_res[j,:,:] = np.std(allclspoiled_res, axis = 2)


#### Pixel Window function
wpix = hp.pixwin(ns,pol=True)
#### Synthesized Beam window function
fwhm = 0.47
ll = np.arange(len(wpix[0]))
bl = exp(-ll**2*(np.radians(fwhm/2.35))**2/2)
#### Correction
corrT = np.interp(newl, ll, wpix[0]**2*bl**2)
corrP = np.interp(newl, ll, wpix[1]**2*bl**2)
corr=corrT

cl = clinit
col = ['black','red', 'blue', 'green', 'magenta']
clf()
for i in np.arange(len(sigptgvals)):
	for j in np.arange(6):
		subplot(2,3,j+1)	
		xlim(0,2*ns)
		plot(newl, cl[i,j,:]/corr**2,color=col[i])
		if j < 4: plot(lll, lll*(lll+1)/(2*np.pi)*spectra[j+1],'k--')







		
