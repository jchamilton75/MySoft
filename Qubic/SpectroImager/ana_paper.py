import os
import qubic
import healpy as hp
import numpy as np
import pylab as plt
import matplotlib as mpl
import sys
import glob

import healpy as hp
from pysimulators import FitsArray
from qubic import gal2equ, equ2gal

from Tools import QubicToolsJCH as qt
from Tools import ReadMC as rmc


center = equ2gal(0., -57.)

#rep_sim = '/Users/hamilton/Qubic/SpectroImager/McCori/VaryTolNsubNptg'
rep_sim = '/Users/hamilton/Qubic/SpectroImager/McCori/Duration20'
nsubvals_sim = np.array([1,2,3,4])
tolvals = np.array(['1e-4'])
# nsubvals_sim = np.array([7,8,9,10])
# tolvals = np.array(['2e-4'])

allarch = []
allarch_in = []
for k in xrange(len(tolvals)):
    arch = []
    arch_in = []
    for i in xrange(len(nsubvals_sim)):
        arch.append('mpiQ_Nodes_*_Ptg_40000_Noutmax_*_Tol_{}*_nf{}_maps_recon.fits'.format(tolvals[k],nsubvals_sim[i]))
        arch_in.append('mpiQ_Nodes_*_Ptg_40000_Noutmax_*_Tol_{}*_nf{}_maps_convolved.fits'.format(tolvals[k],nsubvals_sim[i]))
    allarch.append(arch)
    allarch_in.append(arch_in)

allmeanmat_sim = []
ally_sim = []
ally_cov_sim = []
ally_th_sim = []
allmrms_sim = []
allmrms_cov_sim = []
allmrms_th_sim = []
for k in xrange(len(tolvals)):
    theallmeanmat, xsim, theally, theally_cov, theally_th, mean_rms, mean_rms_cov, mean_rms_covpix = rmc.do_all_profiles(rep_sim, 
        allarch[k], nsubvals_sim, rot=center, nbins=10)
    allmeanmat_sim.append(theallmeanmat)
    ally_sim.append(theally)
    ally_cov_sim.append(theally_cov)
    ally_th_sim.append(theally_th)
    allmrms_sim.append(mean_rms)
    allmrms_cov_sim.append(mean_rms_cov)
    allmrms_th_sim.append(mean_rms_covpix)


figure('Freq-Freq Pixel Averaged Correlation Matrices 2')
clf()
stokes = ['I', 'Q', 'U']
for irec in xrange(len(nsubvals_sim)):
    for t in [0,1,2]:
        subplot(3,len(nsubvals_sim),len(nsubvals_sim)*t+irec+1)
        imshow(qt.cov2corr(allmeanmat_sim[0][irec][:,:,t]), interpolation='nearest',vmin=-1,vmax=1)
        colorbar()
        title(stokes[t])



clf()
plot(nsubvals_sim, nsubvals_sim*0+1,'k--',label='Optimal $\sqrt{N}$',lw=2)
for istokes in arange(2)+1:
    k=0
    plot(nsubvals_sim, allmrms_cov_sim[k][:,istokes]/allmrms_cov_sim[k][0,istokes],label=stokes[istokes],lw=2)
xlabel('Number of sub-frequencies')
ylabel('Noise increase on maps')
ylim(0.95,1.3)
legend()
#savefig('noise_increase.png')


clf()
plot(nsubvals_sim, sqrt(nsubvals_sim),'k--',label='Optimal $\sqrt{N}$',lw=2)
for istokes in arange(2)+1:
    k=0
    ls = '-'
    if istokes==2: 
        ls ='--'
    plot(nsubvals_sim, allmrms_cov_sim[k][:,istokes]/allmrms_cov_sim[k][0,istokes]*sqrt(nsubvals_sim),label=stokes[istokes],lw=2,ls=ls)
xlabel('Number of sub-frequencies')
ylabel('Relative maps RMS')
legend()
title('QUBIC Spectro-Imaging')
# savefig('rms_spectroim.png')


#############################
# get the Cls with XPol
allmapsout, seenmap = rmc.get_all_maps(rep_sim, allarch[0], nsubvals_sim)
allmapsin, seenmap_in = rmc.get_all_maps(rep_sim, allarch_in[0], nsubvals_sim)


ns=int(np.sqrt(len(seenmap)/12))
#mymask = np.zeros(12*ns**2)
#mymask[seenmap] = 1
from qubic import apodize_mask
mymask = apodize_mask(seenmap, 5)



from qubic import Xpol
lmin = 20
lmax = 2*ns
delta_ell = 20
xpol = Xpol(mymask, lmin, lmax, delta_ell)

ell_binned = xpol.ell_binned
nbins = len(ell_binned)



mcls = []
mcls_in = []
scls = []
scls_in = []
for i in xrange(len(nsubvals_sim)):
	sh = allmapsout[i].shape
	nbmc = sh[0]
	nbsub = sh[1]
	cells = np.zeros((6,nbins, nbsub, nbmc))
	cells_in = np.zeros((6,nbins, nbsub, nbmc))
	for k in xrange(nbsub):
		for l in xrange(nbmc):
			print('NsubTot: {} over {} - nsub: {} over {} - Nbmc: {} over {}'.format(i+1,len(nsubvals_sim),k+1,nbsub,l+1,nbmc))
			mymaps = np.zeros((12*ns**2,3))
			mymaps_in = np.zeros((12*ns**2,3))
			for istokes in xrange(3): 
				mymaps[seenmap,istokes] = allmapsout[i][l,k,:,istokes]*mymask[seenmap]
				mymaps_in[seenmap,istokes] = allmapsin[i][l,k,:,istokes]*mymask[seenmap]
			cells[:,:, k, l] = xpol.get_spectra(mymaps)[1]
			cells_in[:,:, k, l] = xpol.get_spectra(mymaps_in)[1]
	mcls.append(np.mean(cells, axis = 3))
	mcls_in.append(np.mean(cells_in, axis = 3))
	scls.append(np.std(cells, axis = 3))
	scls_in.append(np.std(cells_in, axis = 3))

clf()
thespec = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
isub = 0
for s in xrange(3):
	subplot(3,1,s+1)
	ylabel(thespec[s])
	print(isub,shape(mcls[isub]))
	for i in arange(isub+1):
		plot(ell_binned, ell_binned*(ell_binned+1)*mcls[isub][s,:,i])
		plot(ell_binned, ell_binned*(ell_binned+1)*mcls_in[isub][s,:,i],'--')



clf()
thespec = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
isub =1
for s in xrange(3):
	subplot(3,1,s+1)
	ylabel(thespec[s])
	for i in arange(isub+1):
		p=plot(ell_binned, ell_binned*(ell_binned+1)*mcls_in[isub][s,:,i],'--')
		errorbar(ell_binned, ell_binned*(ell_binned+1)*mcls[isub][s,:,i], 
			yerr=ell_binned*(ell_binned+1)*scls[isub][s,:,i],fmt='o', color=p[0].get_color(),
			label=str(i))
		if s==0: legend()














