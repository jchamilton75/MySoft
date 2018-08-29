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
from Tools.ReadMC import *

nsub=2
maps = FitsArray('/Users/hamilton/Qubic/SpectroImager/McCori/VaryTolNsubNptg/mpiQ_Nodes_1_Ptg_40000_Noutmax_6_Tol_5e-5_9350825_nf{}_maps_recon.fits'.format(nsub))
maps_in = FitsArray('/Users/hamilton/Qubic/SpectroImager/McCori/VaryTolNsubNptg/mpiQ_Nodes_1_Ptg_40000_Noutmax_6_Tol_5e-5_9350825_nf{}_maps_convolved.fits'.format(nsub))

# clf()
# for i in xrange(nsub):
#     hp.gnomview(maps[i][:,2], rot=center, reso=12, sub=(3,2,i+1))

# clf()
# for i in xrange(nsub):
#     hp.gnomview(maps_in[i][:,2], rot=center, reso=12, sub=(3,2,i+1), min=-3, max=3)

ns=128
maskok = maps[0][:,0] != hp.UNSEEN
mask = np.zeros(12*ns**2)
mask[maskok] = 1

from qubic import Xpol
lmin = 20
lmax = 2*ns
delta_ell = 20
xpol = Xpol(mask, lmin, lmax, delta_ell)

ell_binned = xpol.ell_binned
nbins = len(ell_binned)

allcls = []
allclsin = []
allclsres = []
for i in xrange(nsub):
    allcls.append(xpol.get_spectra(maps[i])[1])
    allclsin.append(xpol.get_spectra(maps_in[i])[1])
    allclsres.append(xpol.get_spectra(maps[i]-maps_in[i])[1])


thespec = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
clf()
for ss in xrange(3):
    subplot(3,1,ss+1)
    title(thespec[ss])
    for k in xrange(nsub):
        a=plot(ell_binned, ell_binned*(ell_binned+1)*allclsin[k][ss,:],label=k)
        plot(ell_binned, ell_binned*(ell_binned+1)*allcls[k][ss,:],'--', color=a[0].get_color())
        plot(ell_binned, ell_binned*(ell_binned+1)*allclsres[k][ss,:],':', color=a[0].get_color())
legend()











###################################################################################################
#### MC on CORI VARYNsubin: Nsubin=4000, 40000 nsub=1,2,3,4,5,6 tol=5e-4, 1e-4, 5e-5, 1e-5
## RA_center= 0.
## DEC_center=-57.
center = equ2gal(0., -57.)


rep_sim = '/Users/hamilton/Qubic/SpectroImager/McCori/VaryTolNsubNptg'
#nsubvals_sim = np.array([1,2,3,4])
#tolvals = np.array(['5e-4', '1e-4', '5e-5', '1e-5'])
nsubvals_sim = np.array([1,2,3,4,5,6])
tolvals = np.array(['5e-4', '1e-4', '5e-5'])
allarch4000 = []
allarch40000 = []
for k in xrange(len(tolvals)):
    arch4000 = []
    arch40000 = []
    for i in xrange(len(nsubvals_sim)):
        arch4000.append('mpiQ_Nodes_*_Ptg_4000_Noutmax_6_Tol_{}*_nf{}_maps_recon.fits'.format(tolvals[k],nsubvals_sim[i]))
        arch40000.append('mpiQ_Nodes_*_Ptg_40000_Noutmax_6_Tol_{}*_nf{}_maps_recon.fits'.format(tolvals[k],nsubvals_sim[i]))
    allarch4000.append(arch4000)
    allarch40000.append(arch40000)

allmeanmat_sim4000 = []
ally_sim4000 = []
ally_cov_sim4000 = []
ally_th_sim4000 = []
allmrms_sim4000 = []
allmrms_cov_sim4000 = []
allmrms_th_sim4000 = []
allmeanmat_sim40000 = []
ally_sim40000 = []
ally_cov_sim40000 = []
ally_th_sim40000 = []
allmrms_sim40000 = []
allmrms_cov_sim40000 = []
allmrms_th_sim40000 = []
for k in xrange(len(tolvals)):
    theallmeanmat, xsim, theally, theally_cov, theally_th, mean_rms, mean_rms_cov, mean_rms_covpix = do_all_profiles(rep_sim, 
        allarch4000[k], nsubvals_sim, rot=center, nbins=10)
    allmeanmat_sim4000.append(theallmeanmat)
    ally_sim4000.append(theally)
    ally_cov_sim4000.append(theally_cov)
    ally_th_sim4000.append(theally_th)
    allmrms_sim4000.append(mean_rms)
    allmrms_cov_sim4000.append(mean_rms_cov)
    allmrms_th_sim4000.append(mean_rms_covpix)

    theallmeanmat, xsim, theally, theally_cov, theally_th, mean_rms, mean_rms_cov, mean_rms_covpix = do_all_profiles(rep_sim, 
        allarch40000[k], nsubvals_sim, rot=center, nbins=10)
    allmeanmat_sim40000.append(theallmeanmat)
    ally_sim40000.append(theally)
    ally_cov_sim40000.append(theally_cov)
    ally_th_sim40000.append(theally_th)
    allmrms_sim40000.append(mean_rms)
    allmrms_cov_sim40000.append(mean_rms_cov)
    allmrms_th_sim40000.append(mean_rms_covpix)


figure()
clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals_sim)):
    subplot(2,3,isub+1)
    title('Stokes '+stokes[iqu]+r' - PixAv $\nu\nu$ CovMat - Nsub={}'.format(nsubvals_sim[isub]))
    for k in xrange(len(tolvals)):
        p=plot(xsim, ally_cov_sim4000[k][:,isub,iqu], label='4000 Tol = {}'.format(tolvals[k]))
        cc=p[0].get_color()
        plot(xsim, ally_cov_sim40000[k][:,isub,iqu], color=cc, label='40000 Tol = {}'.format(tolvals[k]))
    xlabel('Deg.')
    ylabel('RMS')
    if isub==0: legend()


figure()
clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals_sim)):
    subplot(2,3,isub+1)
    title('Stokes '+stokes[iqu]+r' - PixAv $\nu\nu$ CovMat - Nsub={}'.format(nsubvals_sim[isub]))
    for k in xrange(len(tolvals)):
        p=plot(xsim, ally_cov_sim4000[k][:,isub,iqu]/ally_cov_sim40000[-1][:,isub,iqu], '--', label='4000 Tol = {}'.format(tolvals[k]))
        cc=p[0].get_color()
        plot(xsim, ally_cov_sim40000[k][:,isub,iqu]/ally_cov_sim40000[-1][:,isub,iqu], color=cc, label='40000 Tol = {}'.format(tolvals[k]))
    xlabel('Deg.')
    ylabel('RMS')
    if isub==0: legend()





figure()

clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals_sim)):
    p=plot(xsim, ally_cov_sim4000[1][:,isub,iqu], '--', lw=2, label='Ptg = 4000 - Nsub = {}'.format(nsubvals_sim[isub]))
    cc = p[0].get_color()
    print(cc)
    plot(xsim, ally_cov_sim40000[1][:,isub,iqu], color=cc, lw=2, label='Ptg = 40000 - Nsub = {}'.format(nsubvals_sim[isub]))
xlabel('Deg.')
ylabel('RMS')
legend()


figure()
clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals_sim)):
    p=plot(xsim, ally_cov_sim4000[1][:,isub,iqu]/ally_cov_sim4000[1][:,0,iqu], '--', lw=2, label='Ptg = 4000 - Nsub = {}'.format(nsubvals_sim[isub]))
    cc = p[0].get_color()
    print(cc)
    plot(xsim, ally_cov_sim40000[1][:,isub,iqu]/ally_cov_sim40000[1][:,0,iqu], color=cc, lw=2, label='Ptg = 40000 - Nsub = {}'.format(nsubvals_sim[isub]))
xlabel('Deg.')
ylabel('RMS')
legend()

figure('Freq-Freq Pixel Averaged Correlation Matrices')
clf()
stokes = ['I', 'Q', 'U']
for irec in xrange(len(nsubvals_sim)):
    for t in [0,1,2]:
        subplot(3,len(nsubvals_sim),len(nsubvals_sim)*t+irec+1)
        imshow(qt.cov2corr(allmeanmat_sim4000[1][irec][:,:,t]), interpolation='nearest',vmin=-1,vmax=1)
        colorbar()
        title(stokes[t])

figure('Freq-Freq Pixel Averaged Correlation Matrices 2')
clf()
stokes = ['I', 'Q', 'U']
for irec in xrange(len(nsubvals_sim)):
    for t in [0,1,2]:
        subplot(3,len(nsubvals_sim),len(nsubvals_sim)*t+irec+1)
        imshow(qt.cov2corr(allmeanmat_sim40000[1][irec][:,:,t]), interpolation='nearest',vmin=-1,vmax=1)
        colorbar()
        title(stokes[t])



figure()

clf()
for istokes in xrange(3):
    subplot(1,3,istokes+1)
    title(stokes[istokes])
    for k in xrange(len(tolvals)):
        p=plot(nsubvals_sim, allmrms_cov_sim4000[k][:,istokes],'--',label='Nptg=4000 - tol={}'.format(tolvals[k]))
        cc=p[0].get_color()
        plot(nsubvals_sim, allmrms_cov_sim40000[k][:,istokes], color=cc,label='Nptg=40000 - tol={}'.format(tolvals[k]))
        xlabel('Deg.')
        ylabel('Average map RMS')
        ylim(0.7,1.5)
    legend()



clf()
for istokes in xrange(3):
    k=1
    plot(nsubvals_sim, allmrms_cov_sim40000[k][:,istokes]/allmrms_cov_sim40000[k][0,istokes],label=stokes[istokes],lw=2)
xlabel('Number of sub-frequencies')
ylabel('Noise increase on maps')
ylim(0.95,1.3)
legend()
#savefig('noise_increase.png')









#### MC by Matt. 
rep_matt = '/Users/hamilton/Qubic/SpectroImager/McMatt/mc_128/'
filearchetypes = ['*nrec1_outmaps.fits', '*nrec2_outmaps.fits', '*nrec4_outmaps.fits']
nsubvals_matt = np.array([1,2,4])
rc('font',size=8)
allmeanmat, x, ally, ally_cov, ally_th, mrms, mrms_cov, mrms_covpix = do_all_profiles(rep_matt, filearchetypes, nsubvals_matt)


figure()
clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals_matt)):
    plot(x, ally_cov[:,isub,iqu], label=stokes[iqu]+' - Nsub={}'.format(nsubvals_matt[isub]))
    xlabel('Deg.')
    ylabel('RMS')
    legend()
title('MC files from Matt. Tristram')


figure()
clf()
ylim(0.98,1.25)
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals_matt)):
    plot(x, ally_cov[:,isub,iqu]/ally_cov[:,0,iqu], label=stokes[iqu]+' - Nsub={}'.format(nsubvals_matt[isub]))
    xlabel('Deg.')
    ylabel('RMS/RMS[sub=1]')
    legend()
title('MC files from Matt. Tristram')








#### MC on Tycho: Nptg=4000 nsub=1,2,3,4,5,6 tol=5e-4, 1e-4, 5e-5, 1e-5
## RA_center= 0.
## DEC_center=-57.
center = equ2gal(0., -57.)


rep = '/Users/hamilton/Qubic/SpectroImager/McTycho/'
nsubvals = np.array([1,2,3,4,5,6])
arch5em4 = []
arch1em4 = []
arch5em5 = []
arch1em5 = []
for i in xrange(len(nsubvals)):
    arch5em4.append('mpiQ_Nodes_2_Ptg_4000_Noutmax_6_Tol_5e-4_*_nf{}_maps_recon.fits'.format(nsubvals[i]))
    arch1em4.append('mpiQ_Nodes_2_Ptg_4000_Noutmax_6_Tol_1e-4_*_nf{}_maps_recon.fits'.format(nsubvals[i]))
    arch5em5.append('mpiQ_Nodes_2_Ptg_4000_Noutmax_6_Tol_5e-5_*_nf{}_maps_recon.fits'.format(nsubvals[i]))
    arch1em5.append('mpiQ_Nodes_2_Ptg_4000_Noutmax_6_Tol_1e-5_*_nf{}_maps_recon.fits'.format(nsubvals[i]))


allmeanmat5em4, xx, ally5em4, ally_cov5em4, ally_th5em4, mrms5em4, mrms_cov5em4, mrms_covpix5em4 = do_all_profiles(rep, arch5em4, nsubvals, rot=center, nbins=100)
allmeanmat1em4, xx, ally1em4, ally_cov1em4, ally_th1em4, mrms1em4, mrms_cov1em4, mrms_covpix1em4 = do_all_profiles(rep, arch1em4, nsubvals, rot=center, nbins=100)
allmeanmat5em5, xx, ally5em5, ally_cov5em5, ally_th5em5, mrms5em5, mrms_cov5em5, mrms_covpix5em5 = do_all_profiles(rep, arch5em5, nsubvals, rot=center, nbins=100)
#allmeanmat1em5, x, ally1em5, ally_cov1em5, ally_th1em5 = do_all_profiles(rep, arch1em5, nsubvals, rot=center)



figure()
clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals)):
    plot(xx, ally_cov1em4[:,isub,iqu], label=stokes[iqu]+' - Nsub={}'.format(nsubvals[isub]))
xlabel('Deg. from center of field')
ylabel('RMS')
legend()



figure()
clf()
ylim(0.98,1.25)
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals)):
    plot(xx, ally_cov1em4[:,isub,iqu]/ally_cov1em4[:,0,iqu], label=stokes[iqu]+' - Nsub={}'.format(nsubvals[isub]))
xlabel('Deg.')
ylabel('RMS/RMS[sub=1]')
legend()


figure()
clf()
plot(nsubvals, mrms_cov5em4[:,0]/mrms_cov5em4[0,0], 'b:', label='I')
plot(nsubvals, mrms_cov5em4[:,1]/mrms_cov5em4[0,1], 'g:', label='Q')
plot(nsubvals, mrms_cov5em4[:,2]/mrms_cov5em4[0,2], 'r:', label='U')

plot(nsubvals, mrms_cov1em4[:,0]/mrms_cov1em4[0,0], 'b--', label='I')
plot(nsubvals, mrms_cov1em4[:,1]/mrms_cov1em4[0,1], 'g--', label='Q')
plot(nsubvals, mrms_cov1em4[:,2]/mrms_cov1em4[0,2], 'r--', label='U')

plot(nsubvals, mrms_cov5em5[:,0]/mrms_cov5em5[0,0], 'b', label='I')
plot(nsubvals, mrms_cov5em5[:,1]/mrms_cov5em5[0,1], 'g', label='Q')
plot(nsubvals, mrms_cov5em5[:,2]/mrms_cov5em5[0,2], 'r', label='U')
ylim(0.9,1.4)
xlabel('Number of Sub-Frequencies')
ylabel('Noise Increase on maps')
legend()

figure()
clf()
plot(nsubvals, mrms_cov5em5[:,0]/mrms_cov5em5[0,0], 'b', label='I')
plot(nsubvals, mrms_cov5em5[:,1]/mrms_cov5em5[0,1], 'g', label='Q')
plot(nsubvals, mrms_cov5em5[:,2]/mrms_cov5em5[0,2], 'r', label='U')
ylim(0.9,1.4)
xlabel('Number of Sub-Frequencies')
ylabel('Noise Increase on maps')
legend()


figure()
clf()
plot(nsubvals, mrms_cov5em4[:,0], 'b:', label='I - tol=5e-4')
plot(nsubvals, mrms_cov5em4[:,1], 'g:', label='Q - tol=5e-4')
plot(nsubvals, mrms_cov5em4[:,2], 'r:', label='U - tol=5e-4')

plot(nsubvals, mrms_cov1em4[:,0], 'b--', label='I - tol=1e-4')
plot(nsubvals, mrms_cov1em4[:,1], 'g--', label='Q - tol=1e-4')
plot(nsubvals, mrms_cov1em4[:,2], 'r--', label='U - tol=1e-4')

plot(nsubvals, mrms_cov5em5[:,0], 'b', label='I - tol=5e-5')
plot(nsubvals, mrms_cov5em5[:,1], 'g', label='Q - tol=5e-5')
plot(nsubvals, mrms_cov5em5[:,2], 'r', label='U - tol=5e-5')
xlabel('Number of Sub-Frequencies')
ylabel('Noise Increase on maps')
legend(loc='upper left')






#### Comparison with Matt: difference probably due to different number of input sub frequencies : 15 for me and 5 for Matt
cols = ['blue', 'orange', 'green']
index_matt = [0,1,2]
index_jc = [0, 1, 3]
figure()
clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(cols)):
    plot(xx, ally_cov1em4[:,index_jc[isub],iqu], 
        label=stokes[iqu]+' - Nsub={} (JC Sim)'.format(nsubvals[index_jc[isub]]), color=cols[isub])
    plot(x, ally_cov[:,index_matt[isub],iqu], '--', 
        label=stokes[iqu]+' - Nsub={} (Matt Sim)'.format(nsubvals_matt[index_matt[isub]]), color=cols[isub])
xlabel('Deg.')
ylabel('RMS/RMS[sub=1]')
legend()


cols = ['blue', 'orange', 'green']
index_matt = [0,1,2]
index_jc = [0, 1, 3]
figure()
clf()
ylim(0.98,1.25)
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(cols)):
    plot(xx, ally_cov1em4[:,index_jc[isub],iqu]/ally_cov1em4[:,0,iqu], 
        label=stokes[iqu]+' - Nsub={} (JC Sim)'.format(nsubvals[index_jc[isub]]), color=cols[isub])
    plot(x, ally_cov[:,index_matt[isub],iqu]/ally_cov[:,0,iqu], '--', 
        label=stokes[iqu]+' - Nsub={} (Matt Sim)'.format(nsubvals_matt[index_matt[isub]]), color=cols[isub])
xlabel('Deg.')
ylabel('RMS/RMS[sub=1]')
legend()




figure()
clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals)):
    subplot(2,3,isub+1)
    title('Stokes '+stokes[iqu]+r' - PixAv $\nu\nu$ CovMat - Nsub={}'.format(nsubvals[isub]))
    plot(xx, ally_cov5em4[:,isub,iqu], label=r'Tol=$5\times 10^{-4}$')
    plot(xx, ally_cov1em4[:,isub,iqu], label=r'Tol=$1\times 10^{-4}$')
    plot(xx, ally_cov5em5[:,isub,iqu], label=r'Tol=$5\times 10^{-5}$')
    xlabel('Deg.')
    ylabel(r'RMS')
    legend()

clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals)):
    subplot(2,3,isub+1)
    title('Stokes '+stokes[iqu]+r' - PixAv $\nu\nu$ CovMat - Nsub={}'.format(nsubvals[isub]))
    plot(xx, ally_cov5em4[:,isub,iqu]/ally_cov5em5[:,isub,iqu], label=r'Tol=$5\times 10^{-4}$')
    plot(xx, ally_cov1em4[:,isub,iqu]/ally_cov5em5[:,isub,iqu], label=r'Tol=$1\times 10^{-4}$')
    plot(xx, ally_cov5em5[:,isub,iqu]/ally_cov5em5[:,isub,iqu], label=r'Tol=$5\times 10^{-5}$')
    xlabel('Deg.')
    ylabel(r'RMS/RMS($5\times 10^{-5}$)')
    legend()





#### MC on CORI VARYNODES: Nptg=4000 nsub=1,2 tol=5e-4, Nodes=2, 4, 8 16, 32
#### This is to check that having a number of nodes does not harm the results...
## RA_center= 0.
## DEC_center=-57.
center = equ2gal(0., -57.)


rep_nodes = '/Users/hamilton/Qubic/SpectroImager/McCori/VaryNodes'
nnodes = [2, 4, 8, 16, 32]
nsubvals_nodes = np.array([1,2])
allarch_nodes = []
for k in xrange(len(nnodes)):
    arch = []
    for i in xrange(len(nsubvals_nodes)):
        arch.append('mpiQ_Nodes_{}_Ptg_4000_Noutmax_2_Tol_5e-4_*_nf{}_maps_recon.fits'.format(nnodes[k],nsubvals_nodes[i]))
    allarch_nodes.append(arch)



allmeanmat_nodes = []
ally_nodes = []
ally_cov_nodes = []
ally_th_nodes = []
for k in xrange(len(nnodes)):
    theallmeanmat, xnodes, theally, theally_cov, theally_th = do_all_profiles(rep_nodes, allarch_nodes[k], 
    nsubvals_nodes, rot=center, nbins=100)
    allmeanmat_nodes.append(theallmeanmat)
    ally_nodes.append(theally)
    ally_cov_nodes.append(theally_cov)
    ally_th_nodes.append(theally_th)


figure()
clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals_nodes)):
    subplot(1,2,isub+1)
    ylim(0,2)
    title('Stokes '+stokes[iqu]+r' - PixAv $\nu\nu$ CovMat - Nsub={}'.format(nsubvals_nodes[isub]))
    for k in xrange(len(nnodes)):
        plot(xnodes, ally_cov_nodes[k][:,isub,iqu], label='Nnodes = {}'.format(nnodes[k]))
    xlabel('Deg.')
    ylabel('RMS')
    legend()


figure()
clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals_nodes)):
    subplot(1,2,isub+1)
    ylim(0.6,2)
    title('Stokes '+stokes[iqu]+r' - PixAv $\nu\nu$ CovMat - Nsub={}'.format(nsubvals_nodes[isub]))
    for k in xrange(len(nnodes)):
        plot(xnodes, ally_cov_nodes[k][:,isub,iqu]/ally_cov_nodes[0][:,isub,iqu], label='NNodes = {}'.format(nnodes[k]))
    xlabel('Deg.')
    ylabel('RMS / RMS[2]')
    legend()









#### MC on CORI VARYNPTG: Nptg=4000, 8000, 16000, 32000, 64000, 128000, 256000 nsub=1,2 tol=5e-4, Nodes=according to Nptg
## RA_center= 0.
## DEC_center=-57.
center = equ2gal(0., -57.)


rep_ptg = '/Users/hamilton/Qubic/SpectroImager/McCori/VaryNptg'
nptg = [4000, 16000, 64000, 128000, 256000]
nsubvals_ptg = np.array([1,2])
allarch = []
for k in xrange(len(nptg)):
    arch = []
    for i in xrange(len(nsubvals_ptg)):
        arch.append('mpiQ_Nodes_1_Ptg_{}_Noutmax_2_Tol_5e-4_*_nf{}_maps_recon.fits'.format(nptg[k],nsubvals_ptg[i]))
    allarch.append(arch)


allmeanmat_ptg = []
ally_ptg = []
ally_cov_ptg = []
ally_th_ptg = []
for k in xrange(len(nptg)):
    theallmeanmat, xptg, theally, theally_cov, theally_th = do_all_profiles(rep_ptg, allarch[k], 
    nsubvals_ptg, rot=center, nbins=100)
    allmeanmat_ptg.append(theallmeanmat)
    ally_ptg.append(theally)
    ally_cov_ptg.append(theally_cov)
    ally_th_ptg.append(theally_th)


figure()
clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals_ptg)):
    subplot(1,2,isub+1)
    title('Stokes '+stokes[iqu]+r' - PixAv $\nu\nu$ CovMat - Nsub={}'.format(nsubvals_ptg[isub]))
    for k in xrange(len(nptg)):
        plot(xptg, ally_ptg[k][:,isub,iqu], label='Nptg = {}'.format(nptg[k]))
    xlabel('Deg.')
    ylabel('RMS')
    legend()


figure()
clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals_ptg)):
    subplot(1,2,isub+1)
    title('Stokes '+stokes[iqu]+r' - PixAv $\nu\nu$ CovMat - Nsub={}'.format(nsubvals_ptg[isub]))
    for k in xrange(len(nptg)):
        plot(xptg, ally_ptg[k][:,isub,iqu]/ally_ptg[0][:,isub,iqu], label='Nptg = {}'.format(nptg[k]))
    xlabel('Deg.')
    ylabel('RMS / RMS[4000]')
    legend()




#### MC on CORI VARYNsubin: Nsubin=5,10,15,20 nsub=1,2 tol=5e-4, Nodes=according to Nsubin
## RA_center= 0.
## DEC_center=-57.
center = equ2gal(0., -57.)


rep_subin = '/Users/hamilton/Qubic/SpectroImager/McCori/VaryNsub_in'
nsubin = [5, 10, 15, 20]
nsubvals_subin = np.array([1,2,3,4])
allarch = []
for k in xrange(len(nsubin)):
    arch = []
    for i in xrange(len(nsubvals_subin)):
        arch.append('mpiQ_Nodes_1_Ptg_4000_Noutmax_4_Tol_5e-4_nfin_{}*_nf{}_maps_recon.fits'.format(nsubin[k],nsubvals_subin[i]))
    allarch.append(arch)





allmeanmat_subin = []
ally_subin = []
ally_cov_subin = []
ally_th_subin = []
allmrms_subin = []
allmrms_cov_subin = []
allmrms_th_subin = []
for k in xrange(len(nsubin)):
    theallmeanmat, xsubin, theally, theally_cov, theally_th, mean_rms, mean_rms_cov, mean_rms_covpix = do_all_profiles(rep_subin, allarch[k], 
    nsubvals_subin, rot=center, nbins=10)
    allmeanmat_subin.append(theallmeanmat)
    ally_subin.append(theally)
    ally_cov_subin.append(theally_cov)
    ally_th_subin.append(theally_th)
    allmrms_subin.append(mean_rms)
    allmrms_cov_subin.append(mean_rms_cov)
    allmrms_th_subin.append(mean_rms_covpix)


figure()
clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals_subin)):
    subplot(2,2,isub+1)
    title('Stokes '+stokes[iqu]+r' - PixAv $\nu\nu$ CovMat - Nsub={}'.format(nsubvals_subin[isub]))
    for k in xrange(len(nsubin)):
        plot(xsubin, ally_subin[k][:,isub,iqu], label='Nsubin = {}'.format(nsubin[k]))
    xlabel('Deg.')
    ylabel('RMS')
    legend()


figure()
clf()
stokes = ['I', 'Q', 'U']
iqu=1
for isub in xrange(len(nsubvals_subin)):
    subplot(2,2,isub+1)
    title('Stokes '+stokes[iqu]+r' - PixAv $\nu\nu$ CovMat - Nsub={}'.format(nsubvals_subin[isub]))
    for k in xrange(len(nsubin)):
        plot(xsubin, ally_subin[k][:,isub,iqu]/ally_subin[0][:,isub,iqu], label='Nsubin = {}'.format(nsubin[k]))
    xlabel('Deg.')
    ylabel('RMS / RMS[nsubin=5]')
    legend()

stokes = ['I', 'Q', 'U']
iqu=1
figure()
clf()
for k in xrange(len(nsubin)):
    plot(nsubvals_subin, allmrms_cov_subin[k][:,iqu], label=stokes[iqu]+' - Nsubin = {}'.format(nsubin[k]))
ylim(0.9,1.4)
xlabel('Number of Sub-Frequencies')
ylabel('Noise Increase on maps')
legend()



###################################################################################################
#### MC on CORI VARYNsubin: Nsubin=4000, 40000 nsub=1,2,3,4,5,6 tol=5e-4, 1e-4, 5e-5, 1e-5
## RA_center= 0.
## DEC_center=-57.
center = equ2gal(0., -57.)


rep_sim = '/Users/hamilton/Qubic/SpectroImager/McCori/VaryTolNsubNptg'
nsubvals_sim = np.array([1,2,3,4,5,6])
tolvals = np.array(['5e-4', '1e-4', '5e-5', '1e-5'])
allarch4000 = []
allarch40000 = []
for k in xrange(len(tolvals)):
    arch4000 = []
    arch40000 = []
    for i in xrange(len(nsubvals_sim)):
        arch4000.append('mpiQ_Nodes_*_Ptg_4000_Noutmax_6_Tol_{}*_nf{}_maps_recon.fits'.format(tolvals[k],nsubvals_sim[i]))
        arch40000.append('mpiQ_Nodes_*_Ptg_40000_Noutmax_6_Tol_{}*_nf{}_maps_recon.fits'.format(tolvals[k],nsubvals_sim[i]))
    allarch4000.append(arch4000)
    allarch40000.append(arch40000)

allmeanmat_sim4000 = []
ally_sim4000 = []
ally_cov_sim4000 = []
ally_th_sim4000 = []
allmrms_sim4000 = []
allmrms_cov_sim4000 = []
allmrms_th_sim4000 = []
allmeanmat_sim40000 = []
ally_sim40000 = []
ally_cov_sim40000 = []
ally_th_sim40000 = []
allmrms_sim40000 = []
allmrms_cov_sim40000 = []
allmrms_th_sim40000 = []
for k in xrange(len(tolvals)):
    theallmeanmat, xsim, theally, theally_cov, theally_th, mean_rms, mean_rms_cov, mean_rms_covpix = do_all_profiles(rep_sim, 
        allarch4000[k], nsubvals_sim, rot=center, nbins=10)
    allmeanmat_sim4000.append(theallmeanmat)
    ally_sim4000.append(theally)
    ally_cov_sim4000.append(theally_cov)
    ally_th_sim4000.append(theally_th)
    allmrms_sim4000.append(mean_rms)
    allmrms_cov_sim4000.append(mean_rms_cov)
    allmrms_th_sim4000.append(mean_rms_covpix)

    theallmeanmat, xsim, theally, theally_cov, theally_th, mean_rms, mean_rms_cov, mean_rms_covpix = do_all_profiles(rep_sim, 
        allarch40000[k], nsubvals_sim, rot=center, nbins=10)
    allmeanmat_sim40000.append(theallmeanmat)
    ally_sim40000.append(theally)
    ally_cov_sim40000.append(theally_cov)
    ally_th_sim40000.append(theally_th)
    allmrms_sim40000.append(mean_rms)
    allmrms_cov_sim40000.append(mean_rms_cov)
    allmrms_th_sim40000.append(mean_rms_covpix)






