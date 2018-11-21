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

p=clf()
p=imshow(qt.cov2corr(allmeanmat_sim[0][3][:,:,1]), interpolation='nearest',vmin=-1,vmax=1)
colorbar()
title('Corr. Matrix Q')
#xlabel('Sub-Frequency')
#ylabel('Sub-Frequency')
savefig('corrQ.png')

clf()
subplot(1,2,1)
imshow(qt.cov2corr(allmeanmat_sim[0][3][:,:,1]), interpolation='nearest',vmin=-1,vmax=1)
colorbar()
title('Corr. Matrix Q')
subplot(1,2,2)
imshow(qt.cov2corr(allmeanmat_sim[0][3][:,:,2]), interpolation='nearest',vmin=-1,vmax=1)
colorbar()
title('Corr. Matrix U')




correction = np.zeros((3, len(nsubvals_sim)))
clf()
plot(nsubvals_sim, nsubvals_sim*0+1,'k--',label='Optimal $\sqrt{N}$',lw=2)
for istokes in arange(2)+1:
    k=0
    correction[istokes,:] = allmrms_cov_sim[k][:,istokes]/allmrms_cov_sim[k][0,istokes]
    plot(nsubvals_sim, allmrms_cov_sim[k][:,istokes]/allmrms_cov_sim[k][0,istokes],'o-',label=stokes[istokes],lw=2)
xlabel('Number of sub-frequencies')
ylabel('Noise increase on maps')
ylim(0.,1.3)
legend()
savefig('noise_increase_new.png')


clf()
plot(nsubvals_sim, sqrt(nsubvals_sim),'k--',label='Optimal $\sqrt{N}$',lw=2)
for istokes in arange(3):
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










####################################################################################################
# get the Cls with XPol
allmapsout, seenmap = rmc.get_all_maps(rep_sim, allarch[0], nsubvals_sim)
allmapsin, seenmap_in = rmc.get_all_maps(rep_sim, allarch_in[0], nsubvals_sim)

#### Residual maps w.r.t. input maps convolved at QUBIC resolution
allres_in = []
for i in xrange(len(allmapsout)):
	allres_in.append(allmapsout[i] - allmapsin[i])

#### Residual maps w.r.t. average of realizations
avmaps = []
sigmaps = []
allres_av = []
for i in xrange(len(allmapsout)):
	avmaps.append(np.mean(allmapsout[i], axis=0))
	sigmaps.append(np.std(allmapsout[i], axis=0))
	allres_av.append(allmapsout[i] - avmaps[i])



nus = []
nus_edges = []
for sub in nsubvals_sim:
	bla = glob.glob(rep_sim+'/*nf{}*nus_edges.fits'.format(sub))[0]
	nus_edges.append(FitsArray(bla))
	bla = glob.glob(rep_sim+'/*nf{}*nus.fits'.format(sub))[0]
	nus.append(FitsArray(bla))

ns=int(np.sqrt(len(seenmap)/12))
#mymask = np.zeros(12*ns**2)
#mymask[seenmap] = 1
from qubic import apodize_mask
mymask = apodize_mask(seenmap, 5)



sp = qubic.read_spectra(0)
ll = np.arange(1, 2*ns+2)
dl = []
for i in xrange(4): dl.append(sp[i][0:2*ns+1]*ll*(ll+1)/2/np.pi)

fact = (ll * (ll + 1)) / (2 * np.pi)
coef = 1.39e-2
spectra_dust = [np.zeros(len(ll)), 
              coef * (ll / 80.)**(-0.42) / (fact * 0.52), 
              coef * (ll / 80.)**(-0.42) / fact, 
              np.zeros(len(ll))]

from SpectroImager import SpectroImLib as si
sc_dust = []
for isub in xrange(len(nsubvals_sim)):
	thescdust = np.zeros(isub+1)
	for inu in arange(len(nus[isub])):
		thescdust[inu] = si.scaling_dust(150, nus[isub][inu])
	sc_dust.append(thescdust)



from qubic import Xpol
lmin = 20
lmax = 2*ns
delta_ell = 40
xpol = Xpol(mymask, lmin, lmax, delta_ell)

ell_binned = xpol.ell_binned
nbins = len(ell_binned)

# Pixel window function
pw = hp.pixwin(ns)
pwb = xpol.bin_spectra(pw[:lmax+1])


# FWHM in each sub-band
fwhm_150 = 0.39268176
fwhm = []
bls = []
for n in nus:
	thefwhm = fwhm_150 * (150 / n)
	fwhm.append(thefwhm)	
	allbl = np.zeros((len(n), nbins))
	for i in xrange(len(n)):
		thebl = exp(-0.5 * ll * (ll+1) * np.radians(thefwhm[i]/2.35)**2)
		allbl[i,:] = xpol.bin_spectra(thebl)
	bls.append(allbl)



################################# Just Auto Spectra - Not so good ####################################
# mcls = []
# mcls_in = []
# scls = []
# scls_in = []
# avmapscl = []
# for i in xrange(len(nsubvals_sim)):
# 	sh = allmapsout[i].shape
# 	nbmc = sh[0]
# 	nbsub = sh[1]
# 	cells = np.zeros((6,nbins, nbsub, nbmc))
# 	cells_in = np.zeros((6,nbins, nbsub, nbmc))
# 	avmaps = np.zeros((nsubvals_sim[i], 12*ns**2, 3))
# 	thecl_avmaps = np.zeros((6,nbins, nbsub))
# 	for k in xrange(nbsub):
# 		for l in xrange(nbmc):
# 			print('NsubTot: {} over {} - nsub: {} over {} - Nbmc: {} over {}'.format(i+1,len(nsubvals_sim),k+1,nbsub,l+1,nbmc))
# 			mymaps = np.zeros((12*ns**2,3))
# 			mymaps_in = np.zeros((12*ns**2,3))
# 			for istokes in xrange(3): 
# 				avmaps[k, seenmap, istokes] += allmapsout[i][l,k,:,istokes]*mymask[seenmap]/nbmc
# 				mymaps[seenmap,istokes] = allmapsout[i][l,k,:,istokes]*mymask[seenmap]
# 				mymaps_in[seenmap,istokes] = allmapsin[i][l,k,:,istokes]*mymask[seenmap]
# 			cells[:,:, k, l] = xpol.get_spectra(mymaps)[1]
# 			cells_in[:,:, k, l] = xpol.get_spectra(mymaps_in)[1]
# 		thecl_avmaps[:,:,k] = xpol.get_spectra(avmaps[k,:,:])[1]
# 	avmapscl.append(thecl_avmaps)
# 	mcls.append(np.mean(cells, axis = 3))
# 	mcls_in.append(np.mean(cells_in, axis = 3))
# 	scls.append(np.std(cells, axis = 3)/np.sqrt(nbmc))
# 	scls_in.append(np.std(cells_in, axis = 3)/np.sqrt(nbmc))
#############################################################################################################




############################### Cross Spectra - but not all of them... Dirty way #############################
# mcls = []
# mcls_in = []
# scls = []
# scls_in = []
# avmapscl = []
# for i in xrange(len(nsubvals_sim)):
# 	sh = allmapsout[i].shape
# 	nbmc = sh[0]
# 	nbsub = sh[1]
# 	cells = np.zeros((6,nbins, nbsub, nbmc/2))
# 	cells_in = np.zeros((6,nbins, nbsub, nbmc/2))
# 	avmaps1 = np.zeros((nsubvals_sim[i], 12*ns**2, 3))
# 	avmaps2 = np.zeros((nsubvals_sim[i], 12*ns**2, 3))
# 	thecl_avmaps = np.zeros((6,nbins, nbsub))
# 	for k in xrange(nbsub):
# 		for l in xrange(nbmc/2):
# 			print('NsubTot: {} over {} - nsub: {} over {} - Nbmc: {} over {}'.format(i+1,len(nsubvals_sim),k+1,nbsub,l+1,nbmc))
# 			mymaps1 = np.zeros((12*ns**2,3))
# 			mymaps2 = np.zeros((12*ns**2,3))
# 			mymaps_in = np.zeros((12*ns**2,3))
# 			for istokes in xrange(3): 
# 				avmaps1[k, seenmap, istokes] += allmapsout[i][l,k,:,istokes]*mymask[seenmap]/(nbmc/2)
# 				avmaps2[k, seenmap, istokes] += allmapsout[i][l+nbmc/2,k,:,istokes]*mymask[seenmap]/(nbmc/2)
# 				mymaps1[seenmap,istokes] = allmapsout[i][l,k,:,istokes]*mymask[seenmap]
# 				mymaps2[seenmap,istokes] = allmapsout[i][l+nbmc/2,k,:,istokes]*mymask[seenmap]
# 				mymaps_in[seenmap,istokes] = allmapsin[i][l,k,:,istokes]*mymask[seenmap]
# 			cells[:,:, k, l] = xpol.get_spectra(mymaps1, mymaps2)[1]
# 			cells_in[:,:, k, l] = xpol.get_spectra(mymaps_in)[1]
# 		thecl_avmaps[:,:,k] = xpol.get_spectra(avmaps1[k,:,:],avmaps2[k,:,:])[1]
# 	avmapscl.append(thecl_avmaps)
# 	mcls.append(np.mean(cells, axis = 3))
# 	mcls_in.append(np.mean(cells_in, axis = 3))
# 	scls.append(np.std(cells, axis = 3)/np.sqrt(nbmc/2))
# 	scls_in.append(np.std(cells_in, axis = 3)/np.sqrt(nbmc/2))
#############################################################################################################





#################################### Now all cross and auto Not Parallel => SLow ############################
# def allcross(xpol, allmaps, silent=False):
# 	nmaps = len(allmaps)
# 	nbl = len(xpol.ell_binned)
# 	autos = np.zeros((nmaps,6,nbl))
# 	ncross = nmaps*(nmaps-1)/2
# 	cross = np.zeros((ncross, 6, nbl))
# 	jcross = 0
# 	if not silent: 
# 		print('Computing spectra:')
# 	for i in xrange(nmaps):
# 		#if not silent: print('  Auto: {} over {}'.format(i,nmaps))
# 		autos[i,:,:] = xpol.get_spectra(allmaps[i])[1]
# 		for j in xrange(i+1, nmaps):
# 			cross[jcross,:,:] = xpol.get_spectra(allmaps[i], allmaps[j])[1]
# 			#if not silent: print('  Cross: {} over {}'.format(jcross,ncross))
# 			message = '\r  Auto: {} over {} - Cross {} over {}'.format(i,nmaps,jcross,ncross)
# 			if not silent: 
# 				sys.stdout.write(message)
#         		sys.stdout.flush()
# 			jcross += 1
# 	if not silent: 
# 		sys.stdout.write(' Done \n')
#         sys.stdout.flush()
# 	m_autos = np.mean(autos, axis = 0)
# 	s_autos = np.std(autos, axis = 0)# / np.sqrt(nmaps)
# 	m_cross = np.mean(cross, axis = 0)
# 	s_cross = np.std(cross, axis = 0)# / np.sqrt(ncross)
# 	return m_autos, s_autos, m_cross, s_cross


#################################### Now all cross and auto #############################################
from joblib import Parallel, delayed
import multiprocessing

def allcross_par(xpol, allmaps, silent=False, verbose=1):
	num_cores = multiprocessing.cpu_count()
	nmaps = len(allmaps)
	nbl = len(xpol.ell_binned)
	autos = np.zeros((nmaps,6,nbl))
	ncross = nmaps*(nmaps-1)/2
	cross = np.zeros((ncross, 6, nbl))
	jcross = 0
	if not silent: 
		print('Computing spectra:')

	#### Auto spectra ran in //
	if not silent: print('  Doing All Autos:')
	results_auto = Parallel(n_jobs=num_cores,verbose=verbose)(delayed(xpol.get_spectra)(allmaps[i]) for i in xrange(nmaps))
	for i in xrange(nmaps): autos[i,:,:] = results_auto[i][1]

	#### Cross Spectra ran in // - need to prepare indices in a global variable
	if not silent: print('  Doing All Cross:')
	global cross_indices 
	cross_indices = np.zeros((2, ncross), dtype=int)
	for i in xrange(nmaps):
		for j in xrange(i+1, nmaps):
			cross_indices[:,jcross] = np.array([i,j])
			jcross += 1
	results_cross = Parallel(n_jobs=num_cores,verbose=verbose)(delayed(xpol.get_spectra)(allmaps[cross_indices[0,i]], allmaps[cross_indices[1,i]]) for i in xrange(ncross))
	for i in xrange(ncross): cross[i,:,:] = results_cross[i][1]

	if not silent: 
		sys.stdout.write(' Done \n')
        sys.stdout.flush()

	#### The error-bars are absolutely incorrect if calculated as the following... There is an analytical estimate in Xpol paper. See if implemented in the gitlab xpol from Tristram instead of in qubic.xpol...
	m_autos = np.mean(autos, axis = 0)
	s_autos = np.std(autos, axis = 0) / np.sqrt(nmaps)
	m_cross = np.mean(cross, axis = 0)
	s_cross = np.std(cross, axis = 0) / np.sqrt(ncross)
	return m_autos, s_autos, m_cross, s_cross


mapstoanalyse = allmapsout
#mapstoanalyse = allres_in
#mapstoanalyse = allres_av


mcls = []
mcls_auto = []
mcls_in = []
scls = []
scls_auto = []
scls_in = []
avmapscl = []
for i in xrange(len(nsubvals_sim)):
	sh = mapstoanalyse[i].shape
	nbmc = sh[0]
	nbsub = sh[1]
	cells = np.zeros((6,nbins, nbsub))
	cells_auto = np.zeros((6,nbins, nbsub))
	cells_in = np.zeros((6,nbins, nbsub))
	s_cells = np.zeros((6,nbins, nbsub))
	s_cells_auto = np.zeros((6,nbins, nbsub))
	s_cells_in = np.zeros((6,nbins, nbsub))
	avmaps = np.zeros((nsubvals_sim[i], 12*ns**2, 3))
	thecl_avmaps = np.zeros((6,nbins, nbsub))
	for k in xrange(nbsub):
		allmaps = []
		allmaps_in = []
		print('NsubTot: {} over {} - nsub: {} over {}'.format(i+1,len(nsubvals_sim),k+1,nbsub))
		for l in xrange(nbmc):
			mymaps = np.zeros((12*ns**2,3))
			mymaps_in = np.zeros((12*ns**2,3))
			for istokes in xrange(3): 
				avmaps[k, seenmap, istokes] += mapstoanalyse[i][l,k,:,istokes]/nbmc
				mymaps[seenmap,istokes] = mapstoanalyse[i][l,k,:,istokes]
				mymaps_in[seenmap,istokes] = allmapsin[i][l,k,:,istokes]
			mymaps[:,0]=0
			mymaps_in[:,0]=0
			avmaps[:,:,0]=0
			allmaps.append(mymaps)
			allmaps_in.append(mymaps_in)
		thecl_avmaps[:,:,k] = xpol.get_spectra(avmaps[k,:,:])[1]
		print(np.shape(allmaps))
		m_autos, s_autos, m_cross, s_cross = allcross_par(xpol, allmaps)
		cells[:,:,k] = m_cross
		s_cells[:,:,k] = s_cross
		cells_auto[:,:,k] = m_autos
		s_cells_auto[:,:,k] = s_autos

		cells_in[:,:,k] = xpol.get_spectra(allmaps_in[0])[1]
		s_cells_in[:,:,k] = 0
		# m_autos, s_autos, m_cross, s_cross = allcross(xpol, allmaps_in)
		# cells_in[:,:,k] = m_cross
		# s_cells_in[:,:,k] = s_cross
	
	avmapscl.append(thecl_avmaps)
	mcls.append(cells)
	mcls_auto.append(cells_auto)
	mcls_in.append(cells_in)
	scls.append(s_cells)
	scls_auto.append(s_cells_auto)
	scls_in.append(s_cells_in)
#############################################################################################################



thespec = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
isub = 3

clf()
for s in xrange(3):
	subplot(2,2,s+1)
	if s==2: subplot(2,1,2)
	ylabel(thespec[s])
	for i in arange(isub+1):
		p=plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_in[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,'--', label='Input Convolved '+str(i))
		errorbar(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2, 
			yerr=ell_binned*(ell_binned+1)/2/np.pi*scls[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,fmt='o-', color=p[0].get_color(),
			label='Cross sub-band '+str(i))
		plot(ll, dl[s] + (ll * (ll + 1)) / (2 * np.pi) * spectra_dust[s]*sc_dust[isub][i], color=p[0].get_color(), label='Theory '+str(i), alpha=0.5)
		#if s==0: legend(fontsize=8)
#savefig('cross_isub{}.png'.format(isub))


clf()
for s in xrange(3):
	subplot(2,2,s+1)
	if s==2: subplot(2,1,2)
	ylabel(thespec[s])
	for i in arange(isub+1):
		p=plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_in[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,'--', label='Input Convolved '+str(i))
		errorbar(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_auto[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2, 
			yerr=ell_binned*(ell_binned+1)/2/np.pi*scls_auto[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,fmt='o-', color=p[0].get_color(),
			label='Auto sub-band '+str(i))
		plot(ll, dl[s] + (ll * (ll + 1)) / (2 * np.pi) * spectra_dust[s]*sc_dust[isub][i], color=p[0].get_color(), label='Theory '+str(i), alpha=0.5)
		if s==0: legend(fontsize=8)
#savefig('auto_isub{}.png'.format(isub))



clf()
for s in xrange(3):
	subplot(2,2,s+1)
	if s==2: subplot(2,1,2)
	ylabel(thespec[s])
	for i in arange(isub+1):
		p=plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_in[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,'--', label='Input Convolved '+str(i))
		plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*avmapscl[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,'o-', color=p[0].get_color(), label='Auto Av. Map '+str(i))
		plot(ll, dl[s] + (ll * (ll + 1)) / (2 * np.pi) * spectra_dust[s]*sc_dust[isub][i], color=p[0].get_color(), label='Theory '+str(i), alpha=0.5)
		if s==0: legend(fontsize=8)
#savefig('avmap_auto_isub{}.png'.format(isub))



############# Test Compare with test_guess.py
isub=3
s=2

clf()
ylabel(thespec[s])
title('Average of Cross-Spectra')
for i in arange(isub+1):
	p=plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_in[isub][s,:,i] / pwb**2 ,'--', label='Input Convolved '+str(i))
	errorbar(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls[isub][s,:,i] / pwb**2 , 
		yerr=ell_binned*(ell_binned+1)/2/np.pi*scls[isub][s,:,i] / pwb**2 ,fmt='o-', color=p[0].get_color(),
		label='Cross sub-band '+str(i))
	plot(ll, dl[s] + (ll * (ll + 1)) / (2 * np.pi) * spectra_dust[s]*sc_dust[isub][i], color=p[0].get_color(), label='Theory '+str(i), alpha=0.5)
	ylim(-0.02,0.12)
	xlim(0,250)


clf()
ylabel(thespec[s])
title('Average of Auto-Spectra')
for i in arange(isub+1):
	p=plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_in[isub][s,:,i] / pwb**2,'--', label='Input Convolved '+str(i))
	errorbar(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_auto[isub][s,:,i] / pwb**2, 
		yerr=ell_binned*(ell_binned+1)/2/np.pi*scls_auto[isub][s,:,i] / pwb**2,fmt='o-', color=p[0].get_color(),
		label='Cross sub-band '+str(i))
	plot(ll, dl[s] + (ll * (ll + 1)) / (2 * np.pi) * spectra_dust[s]*sc_dust[isub][i], color=p[0].get_color(), label='Theory '+str(i), alpha=0.5)
	ylim(-0.02,0.5)
	xlim(0,250)


#### compare auto spec for nbsub=1
ell_binned*(ell_binned+1)/2/np.pi*mcls_auto[0][2,:,:].T/pwb**2
isub = 0
mcls_auto[isub][2,:,:]
themap=np.zeros((12*128**2,3))
themap[seenmap,:] = allmapsout[isub][0,0,:,:]
res = xpol.get_spectra(themap)[1]


############# Just BB
clf()
isub=3
s=2


subplot(3,1,1)
ylabel(thespec[s])
title('Average of Cross-Spectra')
for i in arange(isub+1):
	p=plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_in[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,'--', label='Input Convolved '+str(i))
	errorbar(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2, 
		yerr=ell_binned*(ell_binned+1)/2/np.pi*scls[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,fmt='o-', color=p[0].get_color(),
		label='Cross sub-band '+str(i))
	plot(ll, dl[s] + (ll * (ll + 1)) / (2 * np.pi) * spectra_dust[s]*sc_dust[isub][i], color=p[0].get_color(), label='Theory '+str(i), alpha=0.5)
	ylim(-0.02,0.12)
	xlim(0,250)

subplot(3,1,2)
ylabel(thespec[s])
title('Average of Auto-Spectra')
for i in arange(isub+1):
	p=plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_in[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,'--', label='Input Convolved '+str(i))
	errorbar(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_auto[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2, 
		yerr=ell_binned*(ell_binned+1)/2/np.pi*scls_auto[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,fmt='o-', color=p[0].get_color(),
		label='Cross sub-band '+str(i))
	plot(ll, dl[s] + (ll * (ll + 1)) / (2 * np.pi) * spectra_dust[s]*sc_dust[isub][i], color=p[0].get_color(), label='Theory '+str(i), alpha=0.5)

subplot(3,1,3)
ylabel(thespec[s])
title('Auto-Spectrum of Average map')
for i in arange(isub+1):
	p=plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_in[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,'--', label='Input Convolved '+str(i))
	errorbar(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*avmapscl[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,fmt='o-', color=p[0].get_color(),
		label='Cross sub-band '+str(i))
	plot(ll, dl[s] + (ll * (ll + 1)) / (2 * np.pi) * spectra_dust[s]*sc_dust[isub][i], color=p[0].get_color(), label='Theory '+str(i), alpha=0.5)



#### Plot them all
clf()
s=2
for isub in xrange(4):
	subplot(4,3,1+3*isub)
	ylabel(thespec[s])
	if isub==0: title('Average of Cross-Spectra')
	for i in arange(isub+1):
		p=plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_in[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,'--', label='Input Convolved '+str(i))
		errorbar(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2, 
			yerr=ell_binned*(ell_binned+1)/2/np.pi*scls[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,fmt='o-', color=p[0].get_color(),
			label='Cross sub-band '+str(i))
		#plot(ll, dl[s] + (ll * (ll + 1)) / (2 * np.pi) * spectra_dust[s]*sc_dust[isub][i], color=p[0].get_color(), label='Theory '+str(i), alpha=0.5)

		#plot(ell_binned, -(1./(nbmc-1))*ell_binned*(ell_binned+1)/2/np.pi*mcls_auto[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2, 'k--', lw=3)

	subplot(4,3,2+3*isub)
	ylabel(thespec[s])
	if isub==0: title('Average of Auto-Spectra')
	for i in arange(isub+1):
		p=plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_in[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,'--', label='Input Convolved '+str(i))
		errorbar(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_auto[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2, 
			yerr=ell_binned*(ell_binned+1)/2/np.pi*scls_auto[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,fmt='o-', color=p[0].get_color(),
			label='Cross sub-band '+str(i))
		#plot(ll, dl[s] + (ll * (ll + 1)) / (2 * np.pi) * spectra_dust[s]*sc_dust[isub][i], color=p[0].get_color(), label='Theory '+str(i), alpha=0.5)

	subplot(4,3,3+3*isub)
	ylabel(thespec[s])
	if isub==0: title('Auto-Spectrum of Average map')
	for i in arange(isub+1):
		p=plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_in[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,'--', label='Input Convolved '+str(i))
		errorbar(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*avmapscl[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,fmt='o-', color=p[0].get_color(),
			label='Cross sub-band '+str(i))
		#plot(ll, dl[s] + (ll * (ll + 1)) / (2 * np.pi) * spectra_dust[s]*sc_dust[isub][i], color=p[0].get_color(), label='Theory '+str(i), alpha=0.5)







############ Error-Bar analysis => give results that are very unpleasant... 
# My guess is that it is due to the variance of the undesired correlated noise...
col = ['b','m','g','r']
lt = ['-',':','--','-.']
s=2

clf()
ylabel('$\Delta C_\ell$'+thespec[s])
yscale('log')
#ylim(1e-5,1e-1)
for isub in xrange(4):
	for i in arange(isub+1):
		p = plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*scls[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2 ,lt[isub],color=col[i],label='Cross sub-band '+str(i))
		if isub > 0: plot(ell_binned, (isub+1) * ell_binned*(ell_binned+1)/2/np.pi*scls[0][s,:,0] / pwb**2 / bls[0][0,:]**2 ,lt[isub],color='k')
#legend(fontsize=8)




#### Look IQU Covariance matrices
clf()
for isub in xrange(len(nsubvals_sim)):
	themaps = allres_in[isub].copy()

	# we need to correct for the coverage rms effect
	rmspix = np.std(themaps, axis=0)
	themapsnorm = themaps / rmspix
	sh = np.shape(themaps)
	covmats = np.zeros((sh[1], sh[3], sh[3]))
	for i in xrange(sh[1]):
		print(i,sh[1])
		covmats[i,:,:] = np.cov(np.reshape(themapsnorm[:,i,:,:], (sh[0]*sh[2], sh[3])).T)
		subplot(len(nsubvals_sim), len(nsubvals_sim), isub*len(nsubvals_sim)+i+1)
		imshow((qt.cov2corr(covmats[i,:,:])), vmin=-0.03, vmax=0.03)
		#imshow(covmats[i,:,:])
		title('Nsub = {} -  sub={}'.format(isub,i))
		colorbar()




















#### RMS residus
toto = allmapsout[0] - allmapsin[0]
sh = toto.shape

clf()
for i in xrange(3):
	subplot(1,3,i+1)
	hist(np.ravel(toto[:,0,:,i]), range=[-2,2], bins=15)


pixstd = np.std(toto[:,0,:,:], axis=0)
pixmean = np.mean(toto[:,0,:,:], axis=0)

mapsstd = np.zeros((12*ns**2, 3))
mapsmean = np.zeros((12*ns**2, 3))
for i in xrange(3):
	mapsstd[seenmap, i] = pixstd[:,i]
	mapsmean[seenmap, i] = pixmean[:,i]

hp.mollview(mapsstd[:,0])

hp.gnomview(mapsstd[:,0], rot=center, reso=12)
hp.gnomview(mapsstd[:,1], rot=center, reso=12)
hp.gnomview(mapsstd[:,2], rot=center, reso=12)

clf()
for i in xrange(3):
	hp.gnomview(mapsmean[:,i], rot=center, reso=12, sub=(1,3,i+1))


######### small MC
allmapsmc = []

nbmcnoise = 10
npixok = np.sum(seenmap)
mcls_mc = []
scls_mc = []
for i in xrange(len(nsubvals_sim)):
	sh = allmapsout[i].shape
	nbmc = sh[0]
	nbsub = sh[1]
	cells = np.zeros((6,nbins, nbsub, nbmcnoise))
	themapsmc = np.zeros((nbmcnoise, nbsub, npixok, 3))
	for k in xrange(nbsub):
		for l in xrange(nbmcnoise):
			print('NsubTot: {} over {} - nsub: {} over {} - Nbmc: {} over {}'.format(i+1,len(nsubvals_sim),k+1,nbsub,l+1,nbmcnoise))
			mymaps = np.zeros((12*ns**2,3))
			for istokes in xrange(3):
				### NB all maps are identical for this MC so we only take the first one... 
				bla = allmapsin[i][0,k,:,istokes] + np.random.randn(npixok) * pixstd[:,istokes] * np.sqrt(nbsub) *correction[istokes,nbsub-1]
				themapsmc[l,k,:,istokes] = bla
				mymaps[seenmap,istokes] = bla*mymask[seenmap]
			cells[:,:, k, l] = xpol.get_spectra(mymaps)[1]
	allmapsmc.append(themapsmc)
	mcls_mc.append(np.mean(cells, axis = 3))
	scls_mc.append(np.std(cells, axis = 3))



clf()
thespec = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
isub = 2
for s in xrange(3):
	subplot(3,1,s+1)
	ylabel(thespec[s])
	xlim(0,2*ns)
	#ylim(0,np.max(dl[s]))
	for i in arange(isub+1):
		p=plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_in[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,'--')
		errorbar(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2, 
			yerr=ell_binned*(ell_binned+1)/4/np.pi*scls[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,fmt='o', color=p[0].get_color(),
			label=str(i))
		plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_mc[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,':', color=p[0].get_color())
		fill_between(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*(mcls_mc[isub][s,:,i] + scls_mc[isub][s,:,i]) / pwb**2 / bls[isub][i,:]**2,
			y2=ell_binned*(ell_binned+1)/2/np.pi*(mcls_mc[isub][s,:,i] - scls_mc[isub][s,:,i]) / pwb**2 / bls[isub][i,:]**2, color=p[0].get_color(), alpha=0.4)
		plot(ll, dl[s] + (ll * (ll + 1)) / (2 * np.pi) * spectra_dust[s]*sc_dust[isub][i], color=p[0].get_color())
		if s==0: legend()



######### small MC with iqu-nu-iqu-nu covariance matrices

### These are the nu-nu covariance matrices and they are not singular
cov_matrices = allmeanmat_sim[0]
for i in xrange(len(cov_matrices)):
	iqucovmat = cov_matrices[i]
	icovmat = iqucovmat[:,:,0]
	qcovmat = iqucovmat[:,:,1]
	ucovmat = iqucovmat[:,:,2]
	print('')
	print(i)
	print(np.linalg.inv(icovmat))
	print(np.linalg.det(icovmat))
	print(np.linalg.inv(qcovmat))
	print(np.linalg.det(qcovmat))
	print(np.linalg.inv(ucovmat))
	print(np.linalg.det(ucovmat))

### We need covariance matrices between frequencies as well as stokes parameters
residuals = []
for i in xrange(len(nsubvals_sim)):
	residuals.append(allmapsout[i] - allmapsin[i])


def get_covariances(residuals):
	allcoviqunu = []
	for i in xrange(len(residuals)):
		print(i)
		resid = residuals[i]
		sh = np.shape(resid)
		nstokes = 3
		nsubs = sh[1]
		cov = np.zeros((sh[1]*sh[3], sh[1]*sh[3]))
		for isub in xrange(nsubs):
			for istokes in xrange(nstokes):
				ii = isub*nstokes+istokes
				#print('')
				print(isub,istokes,ii)
				dataii = resid[:,isub,:,istokes] / pixstd[:,istokes] * np.mean(pixstd[:,istokes])
				dataii -= np.mean(dataii)
				for jsub in xrange(nsubs):
					for jstokes in xrange(nstokes):
						jj = jsub*nstokes+jstokes
						#print(jsub,jstokes,jj)
						datajj = resid[:,jsub,:,jstokes] / pixstd[:,istokes] * np.mean(pixstd[:,istokes])
						datajj -= np.mean(datajj)
						cov[ii,jj] = np.mean(dataii*datajj)
		allcoviqunu.append(cov)
	return allcoviqunu

allcoviqunu = get_covariances(residuals)


clf()
for i in xrange(len(allmapsout)):
	subplot(1,4,i+1)
	imshow(allcoviqunu[i], interpolation='nearest')
	colorbar()

clf()
for i in xrange(len(allmapsout)):
	subplot(1,4,i+1)
	imshow(qt.cov2corr(allcoviqunu[i]), interpolation='nearest',vmin=-1,vmax=1)
	colorbar()



def gencorrnoise(thepixstd, thecovmat):
	npix = len(thepixstd)
	sh = np.shape(thecovmat)
	bla = np.random.multivariate_normal(np.zeros(sh[0]), thecovmat, npix)
	nbands = sh[0]/3
	simnoise = np.zeros((nbands, npix, 3))
	for inu in xrange(nbands):
		for istokes in xrange(3):
			index = istokes + inu*3
			simnoise[inu, :, istokes] = bla[:, index] * thepixstd[:,istokes] / np.mean(thepixstd[:,istokes])
	return simnoise


nbmc = 10
maps_sim = []
for i in xrange(len(nsubvals_sim)):
	themaps = np.zeros((nbmc, nsubvals_sim[i], npixok, 3))
	for n in xrange(nbmc):
		themaps[n,:,:,:] = gencorrnoise(pixstd, allcoviqunu[i])
	maps_sim.append(themaps)

clf()
stokes = ['I','Q', 'U']
nsub = 3
for istokes in xrange(3):
	for isub in xrange(nsubvals_sim[nsub]):
		subplot(nsubvals_sim[nsub], 3, 1 + istokes + isub*3)
		title(stokes[istokes])
		data = np.ravel(residuals[nsub][:,isub,:,istokes])
		hist(data, bins=30, range=[-5,5], alpha=0.5, label = 'Data : $\sigma$={0:4.2f}'.format(np.std(data)))
		sim = np.ravel(maps_sim[nsub][:,isub,:,istokes])
		hist(sim, bins=30, range=[-5,5], alpha=0.5, label = 'Sim : $\sigma$={0:4.2f}'.format(np.std(sim)))
		legend(fontsize=8,frameon=False)


allcoviqunu = get_covariances(residuals)
allcoviqunu_sim = get_covariances(maps_sim)

figure()
clf()
for i in xrange(len(allmapsout)):
	subplot(1,4,i+1)
	imshow(allcoviqunu[i], interpolation='nearest')
	colorbar()

figure()
clf()
for i in xrange(len(allmapsout)):
	subplot(1,4,i+1)
	imshow(allcoviqunu_sim[i], interpolation='nearest')
	colorbar()


nbmcnoise = 10
npixok = np.sum(seenmap)
mcls_mc = []
scls_mc = []
for i in xrange(len(nsubvals_sim)):
	sh = allmapsout[i].shape
	nbmc = sh[0]
	nbsub = sh[1]
	cells = np.zeros((6,nbins, nbsub, nbmcnoise))
	for k in xrange(nbsub):
		for l in xrange(nbmcnoise):
			print('NsubTot: {} over {} - nsub: {} over {} - Nbmc: {} over {}'.format(i+1,len(nsubvals_sim),k+1,nbsub,l+1,nbmcnoise))
			mymaps = np.zeros((12*ns**2,3))
			for istokes in xrange(3):
				### NB all maps are identical for this MC so we only take the first one... 
				bla = allmapsin[i][0,k,:,istokes] + maps_sim[i][l,k,:,istokes]
				mymaps[seenmap,istokes] = bla*mymask[seenmap]
			cells[:,:, k, l] = xpol.get_spectra(mymaps)[1]
	mcls_mc.append(np.mean(cells, axis = 3))
	scls_mc.append(np.std(cells, axis = 3))



clf()
thespec = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
isub = 3
for s in xrange(3):
	subplot(3,1,s+1)
	ylabel(thespec[s])
	xlim(0,2*ns)
	#ylim(0,np.max(dl[s]))
	for i in arange(isub+1):
		p=plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_in[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,'--')
		errorbar(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2, 
			yerr=ell_binned*(ell_binned+1)/4/np.pi*scls[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,fmt='o', color=p[0].get_color(),
			label=str(i))
		plot(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*mcls_mc[isub][s,:,i] / pwb**2 / bls[isub][i,:]**2,':', color=p[0].get_color())
		fill_between(ell_binned, ell_binned*(ell_binned+1)/2/np.pi*(mcls_mc[isub][s,:,i] + scls_mc[isub][s,:,i]) / pwb**2 / bls[isub][i,:]**2,
			y2=ell_binned*(ell_binned+1)/2/np.pi*(mcls_mc[isub][s,:,i] - scls_mc[isub][s,:,i]) / pwb**2 / bls[isub][i,:]**2, color=p[0].get_color(), alpha=0.4)
		plot(ll, dl[s] + (ll * (ll + 1)) / (2 * np.pi) * spectra_dust[s]*sc_dust[isub][i], color=p[0].get_color())
		if s==0: legend()


