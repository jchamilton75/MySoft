import os
import qubic
import healpy as hp
import numpy as np
import pylab as plt
from pylab import *
import matplotlib as mpl
import sys
import glob

import healpy as hp
from pysimulators import FitsArray
from qubic import gal2equ, equ2gal
from Tools import QubicToolsJCH as qt
from Tools import ReadMC as rmc

from SpectroImager import SpectroImLib as si




###################################################################################################
#### MC on CORI VARYNsubin: Nsubin=4000, 40000 nsub=1,2,3,4,5,6 tol=5e-4, 1e-4, 5e-5, 1e-5
## RA_center= 0.
## DEC_center=-57.
center = equ2gal(0., -57.)
rep_sim = '/Users/hamilton/Qubic/SpectroImager/McCori/VaryTolNsubNptg'
nsubvals_sim = np.array([1,2,3,4,5,6])
tolvals = np.array(['5e-4', '1e-4', '5e-5'])

#### Get Input maps and frequencies
inmaps = []
freqs = []
for i in xrange(len(nsubvals_sim)):
	nsout = nsubvals_sim[i]
	ff = glob.glob(rep_sim+'/*_nf{}_maps_convolved.fits'.format(nsout))
	mm = FitsArray(ff[0])
	inmaps.append(mm)
	ffile = glob.glob(rep_sim+'/*_nf{}_nus.fits'.format(nsout))
	nus = FitsArray(ffile[0])
	freqs.append(nus)

#### Get Output maps files list
allarch = []
for k in xrange(len(tolvals)):
    arch = []
    for i in xrange(len(nsubvals_sim)):
        arch.append('mpiQ_Nodes_*_Ptg_40000_Noutmax_6_Tol_{}*_nf{}_maps_recon.fits'.format(tolvals[k],nsubvals_sim[i]))
    allarch.append(arch)


#### Get output maps and (pixel averaged nu-nu cov) averaged output maps
allmaps = []
allmeanrecombinedmaps = []
allmeanmaps = []
for k in xrange(len(tolvals)):
    mm, seen = rmc.get_all_maps(rep_sim, allarch[k], nsubvals_sim)
    thrmsmap, allmeanmat, allstdmat = rmc.get_rms_covar(nsubvals_sim, seen, mm)
    meanrecombinedmaps, rmsrecombinedmaps = rmc.get_rms_covarmean(nsubvals_sim, seen, mm, allmeanmat)
    allmaps.append(mm)
    allmeanrecombinedmaps.append(meanrecombinedmaps)
    themeanmm = []
    for i in xrange(len(nsubvals_sim)):
        themeanmm.append(np.mean(mm[i], axis=0))
    allmeanmaps.append(themeanmm)



#### Run XPol ###############################################
## Apodization
apodization_mask = qubic.apodize_mask(seen, 5)
hp.gnomview(apodization_mask, rot=center, reso=15)

## Xpol initialization
nside = hp.npix2nside(len(seen))
lmin = 35
lmax = 2*nside-1
delta_ell = 20
xpol = qubic.Xpol(apodization_mask, lmin, lmax, delta_ell)
ell_binned = xpol.ell_binned
nbins = len(ell_binned)

def getXpol(xpol, maps_in, nside, seen):
	maps = np.zeros((3, 12*nside**2))+np.nan
	for iqu in [0,1,2]:
		maps[iqu,:] = qt.smallhpmap(maps_in[:,iqu],seen)
		unseen = maps == hp.UNSEEN
		maps[unseen]=0
	return xpol.get_spectra(maps)


## Xpol on input maps
allcls_input = []
allclsX_input = []
for i in xrange(len(nsubvals_sim)):
	maps_in = inmaps[i]
	jcls = np.zeros((nsubvals_sim[i], 6, 2*nside))
	jclsX = np.zeros((nsubvals_sim[i], 6, nbins))
	for j in xrange(nsubvals_sim[i]):
		print(i,j)
		jcls[j,:,:], jclsX[j,:,:] = getXpol(xpol, maps_in[j,:,:], nside, seen)
	print(jclsX[:,2,:])
	allcls_input.append(jcls)
	allclsX_input.append(jclsX)


figure()
clf()
for i in xrange(len(nsubvals_sim)):
	subplot(2,3,nsubvals_sim[i])
	yscale('log')
	ylim(0.1,1000)
	clsXin = allclsX_input[i]
	for j in xrange(nsubvals_sim[i]):
		plot(ell_binned, clsXin[j,2,:], label = j+1)
	legend()


### Input Cls
# Parameters used for input maps
skypars = {'dust_coeff':1.39e-2, 'r':0}
coef = skypars['dust_coeff']
ell = np.arange(1, 2*nside)
fact = (ell * (ell + 1)) / (2 * np.pi)
spectra_dust = [np.zeros(len(ell)), 
                  coef * (ell / 80.)**(-0.42) / (fact * 0.52), 
                  coef * (ell / 80.)**(-0.42) / fact, 
                  np.zeros(len(ell))]
spcmb = qubic.read_spectra(skypars['r'])

spec_input = []
for i in xrange(len(nsubvals_sim)):
	spec = np.zeros((len(ell),nsubvals_sim[i],3))
	for j in xrange(nsubvals_sim[i]):
		spec[:,j,0] = spectra_dust[0] * si.scaling_dust(150, freqs[i][j], 1.59) + spcmb[0][0:len(ell)]
		spec[:,j,1] = spectra_dust[1] * si.scaling_dust(150, freqs[i][j], 1.59) + spcmb[1][0:len(ell)]
		spec[:,j,2] = spectra_dust[2] * si.scaling_dust(150, freqs[i][j], 1.59) + spcmb[2][0:len(ell)]
	spec_input.append(spec)


figure()
clf()
teb = 1
for i in xrange(len(nsubvals_sim)):
	subplot(2,3,nsubvals_sim[i])
	yscale('log')
	#ylim(0.1,1000)
	clsXin = allclsX_input[i]
	for j in xrange(nsubvals_sim[i]):
		p = plot(ell_binned, clsXin[j,teb,:],'.-', label = j+1)
		cc = p[0].get_color()
		plot(ell, spec_input[i][:,j,teb]*1e6, ':', color=cc)
	legend(fontsize=6)






## Xpol on recombined maps
allcls_rec = []
allclsX_rec = []
for i in xrange(len(nsubvals_sim)):
	maps_rec = allmeanrecombinedmaps[2][i]
	jcls = np.zeros((6, 2*nside))
	jclsX = np.zeros((6, nbins))
	print(i)
	jcls, jclsX = getXpol(xpol, maps_rec.T, nside, seen)
	print(jclsX[2,:])
	allcls_rec.append(jcls)
	allclsX_rec.append(jclsX)


figure()
clf()
clsXin = allclsX_input[0]
yscale('log')
ylim(0.1,1000)
teb = 2
plot(ell_binned, clsXin[0,2,:], label='Input', color='k')
for i in xrange(len(nsubvals_sim)):
	p = plot(ell_binned, allclsX_rec[i][teb,:], 'o', label = i+1)
	cc = p[0].get_color()
	plot(ell, spec_input[0][:,0,teb]*1e6, ':', color=cc)
legend()

figure()
clf()
clsXin = allclsX_input[0]
for i in xrange(len(nsubvals_sim)):
	plot(ell_binned, allclsX_rec[i][2,:]-clsXin[0,2,:], label = i+1)
legend()
## en principe j'ai 10 realisations donc je devrais pouvoire trouver des barres d'erreur...




#### XPol on MC averaged maps per sub-frequency
allcls_meanout = []
allclsX_meanout = []
for k in xrange(len(tolvals)):
	the_allcls_meanout = []
	the_allclsX_meanout = []
	maps_m = allmeanmaps[k]
	for i in xrange(len(nsubvals_sim)):
		maps = maps_m[i]
		jcls = np.zeros((nsubvals_sim[i], 6, 2*nside))
		jclsX = np.zeros((nsubvals_sim[i], 6, nbins))
		for j in xrange(nsubvals_sim[i]):
			print(k,i,j)
			jcls[j,:,:], jclsX[j,:,:] = getXpol(xpol, maps[j,:,:], nside, seen)
		print(jclsX[:,2,:])
		the_allcls_meanout.append(jcls)
		the_allclsX_meanout.append(jclsX)
	allcls_meanout.append(the_allcls_meanout)
	allclsX_meanout.append(the_allclsX_meanout)

figure()
clf()
teb = 1
for i in xrange(len(nsubvals_sim)):
	subplot(2,3,nsubvals_sim[i])
	yscale('log')
	#ylim(0.1,1000)
	clsXin = allclsX_input[i]
	for j in xrange(nsubvals_sim[i]):
		p = plot(ell_binned, clsXin[j,teb,:], label = j+1)
		cc=p[0].get_color()
		for itol in [0,1,2]:
			clsXout = allclsX_meanout[itol][i]
			plot(ell_binned, clsXout[j,teb,:], 'o', color=cc, alpha=0.5+itol*0.5/3)
			plot(ell, spec_input[i][:,j,teb]*1e6, ':', color=cc)
	legend(fontsize=6)
































