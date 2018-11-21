import os
import qubic
import healpy as hp
import numpy as np
import pylab as plt
from pylab import *
import matplotlib as mpl
import sys
import glob

from joblib import Parallel, delayed
import multiprocessing

import healpy as hp
from pysimulators import FitsArray
from qubic import gal2equ, equ2gal
from Tools import QubicToolsJCH as qt


def get_seenmap(files):
    print('\nGetting Observed pixels map')
    m = FitsArray(files[0])
    dims = np.shape(m)
    npix = dims[1]
    seenmap = np.zeros(npix) == 0
    for i in xrange(len(files)):
            sys.stdout.flush()
            sys.stdout.write('\r Reading: '+files[i]+' ({:3.1f} %)'.format(100*(i+1)*1./len(files)))
            m = FitsArray(files[i])
            bla = np.mean(m, axis=(0,2)) != hp.UNSEEN
            seenmap *= bla
    sys.stdout.flush()
    return seenmap

def read_all_maps(rep, archetype, nsub, seenmap, nmax=False):
    print('\nReading all maps')
    npixok = np.sum(seenmap)
    fout = glob.glob(rep+'/'+archetype)
    if not nmax: nmax=len(fout)
    fout = fout[0:nmax]
    print('Doing: '+rep+archetype,nsub, len(fout))
    mapsout = np.zeros((len(fout), nsub, npixok, 3))
    for ifile in xrange(len(fout)):
        sys.stdout.flush()
        sys.stdout.write('\r Reading: '+fout[ifile]+' ({:3.1f} %)'.format(100*(ifile+1)*1./len(fout)))
        mm = FitsArray(fout[ifile])
        mapsout[ifile,:,:,:] = mm[:,seenmap,:]
    sys.stdout.flush()    
    return mapsout




def get_simple_mean_rms(nsubvals, seenmap, allmapsout):
    print('\n\nCalculating Simple Mean and RMS')
    npixok = np.sum(seenmap)
    rmsmap = np.zeros((len(nsubvals),3,npixok))+hp.UNSEEN
    meanmap = np.zeros((len(nsubvals),3,npixok))+hp.UNSEEN
    for i in xrange(len(nsubvals)):
        print('for nsub = {}'.format(nsubvals[i]))
        mapsout = allmapsout[i]
        for p in xrange(npixok):
            for iqu in [0,1,2]:
                meanoverfreqs = np.mean(mapsout[:,:,p,iqu], axis=1)
                meanmap[i,iqu,p] = np.mean(meanoverfreqs, axis=0)
                rmsmap[i,iqu,p] = np.std(meanoverfreqs, axis=0)
    return meanmap, rmsmap


#Now the test done by Matthieu Tristram: 
# calculate the variance map in each case accounting for the band-band covariance matrix 
# for each pixel from the MC. This is pretty noisy so it may be interesting to get the 
# average matrix
# We calculate all the matrices for each pixel and normalize them to 
# avergae 1 and then calculate the average matrix
def get_rms_covar(nsubvals, seenmap, allmapsout):
    print('\n\nCalculating variance map with freq-freq cov matrix for each pixel from MC')
    seen =  np.where(seenmap == 1)[0]
    npixok = np.sum(seenmap)
    variance_map = np.zeros((len(nsubvals), 3, npixok))+hp.UNSEEN
    allmeanmat = []
    allstdmat = []
    for irec in xrange(len(nsubvals)):
        print('for nsub = {}'.format(nsubvals[irec]))
        allmaps = allmapsout[irec]
        allmat = np.zeros((nsubvals[irec],nsubvals[irec],len(seen), 3))
        for p in xrange(len(seen)):
            for t in [0,1,2]:
                mat = np.cov(allmaps[:,:,p,t].T)
                if np.size(mat) == 1: variance_map[irec,t,p] = mat
                else: variance_map[irec,t,p] = 1./np.sum(np.linalg.inv(mat))
                allmat[:,:,p,t] = mat/np.mean(mat) ### its normalization is irrelevant for the later average
        meanmat = np.zeros((nsubvals[irec],nsubvals[irec],3))
        stdmat = np.zeros((nsubvals[irec],nsubvals[irec],3))
        for t in [0,1,2]:
            meanmat[:,:,t] = np.mean(allmat[:,:,:,t], axis=2)
            stdmat[:,:,t] = np.std(allmat[:,:,:,t], axis=2)
        allmeanmat.append(meanmat)
        allstdmat.append(stdmat)
    return np.sqrt(variance_map), allmeanmat, allstdmat



def mean_cov(vals, invcov):
    AtNid = np.sum(np.dot(invcov, vals))
    AtNiA_inv = 1./np.sum(invcov)
    return AtNid*AtNiA_inv
    

### RMS map using the pixel averaged freq-freq covariance matrix
def get_rms_covarmean(nsubvals, seenmap, allmapsout, allmeanmat):
    print('\n\nCalculating variance map with pixel averaged freq-freq cov matrix from MC')
    seen =  np.where(seenmap == 1)[0]
    npixok = np.sum(seenmap)

    rmsmap_cov = np.zeros((len(nsubvals),3,npixok))+hp.UNSEEN
    meanmap_cov = np.zeros((len(nsubvals),3,npixok))+hp.UNSEEN

    for i in xrange(len(nsubvals)):
        print('for nsub = {}'.format(nsubvals[i]))
        mapsout = allmapsout[i]
        sh = mapsout.shape
        nreals = sh[0]
        for iqu in [0,1,2]:
            covmat = allmeanmat[i][:,:,iqu]
            invcovmat = np.linalg.inv(covmat)
            for p in xrange(npixok):
                vals = np.zeros(nreals)
                for real in xrange(nreals):
                    vals[real] = mean_cov(mapsout[real,:,p,iqu], invcovmat)
                meanmap_cov[i,iqu,p] = np.mean(vals)
                rmsmap_cov[i,iqu,p] = np.std(vals)
    return meanmap_cov, rmsmap_cov

def get_all_maps(rep, filearchetypes, nsubvals, nmax=False):
    ### Get the map of observed pixels
    seenmap = True
    for i in xrange(len(filearchetypes)):
        fa = filearchetypes[i]
        files = glob.glob(rep+'/'+fa)
        seenmap *= get_seenmap(files[0:10])
    npixok = np.sum(seenmap)

    ### Now read all the output maps: only seen pixels are stored in order to save memory
    allmapsout = []
    for i in xrange(len(nsubvals)):
        mapsout = read_all_maps(rep, filearchetypes[i], nsubvals[i], seenmap, nmax=nmax)
        allmapsout.append(mapsout)
    return allmapsout, seenmap

def get_all_rmsmaps(rep, filearchetypes, nsubvals, nmax=False):
    allmapsout, seenmap = get_all_maps(rep, filearchetypes, nsubvals, nmax)
    npixok = np.sum(seenmap)

    ### Simple Mean and RMS over realization (no frequency covariance matrix)
    meanmap, rmsmap = get_simple_mean_rms(nsubvals, npixok, allmapsout)

    ### Mean with freq-freq covariance matrix for each pixel (averaged over realizations): 
    ### this is the code from Matt
    ### it also returns the pixel averaged covariance matrix and its RMS 
    ### (each pixel freq-freq cov matrix has been renormalized before averaging)
    rmsmap_covpix, allmeanmat, allstmat = get_rms_covar(nsubvals, seenmap, allmapsout)

    ### Mean with pixel averaged freq freq covariance matrix
    meanmap_cov, rmsmap_cov = get_rms_covarmean(nsubvals, npixok, allmapsout, allmeanmat)

    return seenmap, meanmap, rmsmap, rmsmap_covpix, allmeanmat, allstmat, meanmap_cov, rmsmap_cov 

def get_profile(figname, nsubvals, rmsmap, seenmap, reso=12, nbins=100, rot=None):
    from pysimulators import profile
    ally = np.zeros((nbins,len(nsubvals),3))
    ioff()
    figure(figname, figsize=(8,8))
    clf()
    nx=len(nsubvals)
    ny=3
    stokes = ['I', 'Q', 'U']
    for i in xrange(len(nsubvals)):
        for iqu in [0,1,2]:
            print(i,iqu)
            print(nx,ny,iqu*len(nsubvals)+i+1)
            print('')
            img = hp.gnomview(qt.smallhpmap(rmsmap[i,iqu,:],seenmap), rot=rot, 
                              reso=12, sub=(ny, nx, iqu*len(nsubvals)+i+1), fig=figure, min=0, max=5, 
                              title=stokes[iqu]+' Nsub={}'.format(nsubvals[i]), return_projected_map=True)
            x, y = profile(img**2,bin=100./nbins)
            x *= reso *1. / 60
            ally[:,i,iqu] = np.sqrt(y)
    ion()
    return x, ally 
    
def do_all_profiles(rep, filearchetypes, nsubvals, reso=12, nbins=100, nmax=False, rot=None):
    seenmap, meanmap, rmsmap, rmsmap_covpix, allmeanmat, allstmat, meanmap_cov, rmsmap_cov  = get_all_rmsmaps(rep, 
    filearchetypes, nsubvals, nmax=nmax)
    if len(seenmap)==0:
        return [],[],[],[],[], [], [],[]
    mean_rms = np.sqrt(np.mean(rmsmap**2,axis=2))
    mean_rms_covpix = np.sqrt(np.mean(rmsmap_covpix**2,axis=2))
    mean_rms_cov = np.sqrt(np.mean(rmsmap_cov**2,axis=2))


    x, ally = get_profile('Simple Average', nsubvals, rmsmap, seenmap, reso=reso, nbins=nbins, rot=rot)
    x, ally_th = get_profile('Freq-Freq Cov (each pix) Average', nsubvals, rmsmap_covpix, seenmap, reso=reso, nbins=nbins, rot=rot)
    x, ally_cov = get_profile('Pixel Averaged Freq-Freq Cov Average', nsubvals, rmsmap_cov, seenmap, reso=reso, nbins=nbins, rot=rot)

    figure('Freq-Freq Pixel Averaged Correlation Matrices')
    clf()
    stokes = ['I', 'Q', 'U']
    for irec in xrange(len(nsubvals)):
        for t in [0,1,2]:
            subplot(3,len(nsubvals),len(nsubvals)*t+irec+1)
            imshow(qt.cov2corr(allmeanmat[irec][:,:,t]), interpolation='nearest',vmin=-1,vmax=1)
            colorbar()
            title(stokes[t])

    fig=figure('profiles', figsize=(8,8))
    clf()
    iqunames = ['I','Q','U']
    for i in xrange(len(nsubvals)):
        for iqu in [0,1,2]:
            subplot(3,3,iqu+1)
            plot(x, ally[:,i,iqu], label='Nsub = {}'.format(nsubvals[i]))
            xlabel('Deg.')
            ylabel('Map RMS')
            if i==0: title(iqunames[iqu]+' MC Raw')
    for i in xrange(len(nsubvals)):
        for iqu in [0,1,2]:
            subplot(3,3,iqu+1+3)
            plot(x, ally_cov[:,i,iqu], label='Nsub = {}'.format(nsubvals[i]))
            xlabel('Deg.')
            ylabel('Map RMS')
            if i==0: title(iqunames[iqu]+' MC With Cov')
    legend(fontsize=12, loc='upper left')
    for i in xrange(len(nsubvals)):
        for iqu in [0,1,2]:
            subplot(3,3,iqu+1+6)
            plot(x, ally_th[:,i,iqu], label='Nsub = {}'.format(nsubvals[i]))
            xlabel('Deg.')
            ylabel('Map RMS')
            if i==0: title(iqunames[iqu]+' Th. from MC')
    legend(fontsize=12, loc='upper left')

    return allmeanmat, x, ally, ally_cov, ally_th, mean_rms, mean_rms_cov, mean_rms_covpix




def plotmaps(maps, rng=None, center=None, reso=8):
    if center==None:
        center = qubic.equ2gal(0., -57.)

    sh = np.shape(maps)
    nf_sub_rec = sh[0]
    stokes = ['I', 'Q', 'U']
    if rng==None:
        rng = [[-300, 300], [-5, 5], [-5,5]]
    else:
        rng = [[-rng[0], rng[0]], [-rng[1], rng[1]], [-rng[2], rng[2]]]
    for i in xrange(nf_sub_rec):
        hp.gnomview(maps[i,:,0], rot=center, reso=reso, 
            sub=(nf_sub_rec, 3, 3*i+1), title='I{}'.format(i), min=rng[0][0], max=rng[0][1])
        hp.gnomview(maps[i,:,1], rot=center, reso=reso, 
            sub=(nf_sub_rec, 3, 3*i+2), title='I{}'.format(i), min=rng[1][0], max=rng[1][1])
        hp.gnomview(maps[i,:,2], rot=center, reso=reso, 
            sub=(nf_sub_rec, 3, 3*i+3), title='I{}'.format(i), min=rng[2][0], max=rng[2][1])



def maps_from_files(files, silent=False):
    if not silent: print('Reading Files')
    nn = len(files)
    mm = FitsArray(files[0])
    sh = np.shape(mm)
    maps = np.zeros((nn, sh[0], sh[1], sh[2]))
    for i in xrange(nn):
        maps[i,:,:,:] = FitsArray(files[i])
    totmap = np.sum(np.sum(np.sum(maps, axis=0), axis=0),axis=1)
    seenmap = totmap > -1e20
    bla = maps[:,:,seenmap,:]
    return maps, seenmap


def get_maps_residuals(frec, fconv=None, silent=False):
    mrec, seenmap = maps_from_files(frec)
    if fconv==None:
        if not silent: print('Getting Residuals from average MC')
        resid = np.zeros_like(mrec)
        mean_mrec = np.mean(mrec, axis =0)
        for i in xrange(len(frec)):
            resid[i,:,:,:] = mrec[i,:,:,:]- mean_mrec[:,:,:]
    else:
        if not silent: print('Getting Residuals from convolved input maps')
        mconv, seenmap_c = maps_from_files(fconv)
        resid = mrec-mconv
    resid[:,:,~seenmap,:] = 0
    return mrec, resid, seenmap


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
    if not silent: print('  Doing All Autos ({}):'.format(nmaps))
    results_auto = Parallel(n_jobs=num_cores,verbose=verbose)(delayed(xpol.get_spectra)(allmaps[i]) for i in xrange(nmaps))
    for i in xrange(nmaps): autos[i,:,:] = results_auto[i][1]

    #### Cross Spectra ran in // - need to prepare indices in a global variable
    if not silent: print('  Doing All Cross ({}):'.format(ncross))
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



def get_maps_cl(frec, fconv=None, lmin=20, delta_ell=40, apodization_degrees=5.):
    mrec, resid, seenmap = get_maps_residuals(frec,fconv=fconv)
    sh = np.shape(mrec)
    nbsub = sh[1]
    ns = hp.npix2nside(sh[2])

    from qubic import apodize_mask
    mymask = apodize_mask(seenmap, apodization_degrees)


    #### Create XPol object
    from qubic import Xpol
    lmax = 2*ns
    xpol = Xpol(mymask, lmin, lmax, delta_ell)
    ell_binned = xpol.ell_binned
    nbins = len(ell_binned)
    # Pixel window function
    pw = hp.pixwin(ns)
    pwb = xpol.bin_spectra(pw[:lmax+1])

    #### Calculate all crosses and auto
    m_autos = np.zeros((nbsub, 6, nbins))
    s_autos = np.zeros((nbsub, 6, nbins))
    m_cross = np.zeros((nbsub, 6, nbins))
    s_cross = np.zeros((nbsub, 6, nbins))
    fact = ell_binned * (ell_binned+1) /2. /np.pi
    for isub in xrange(nbsub):
        m_autos[isub, :, :], s_autos[isub, :, :], m_cross[isub, :, :], s_cross[isub, :, :] = allcross_par(xpol, mrec[:,isub,:,:], silent=False, verbose=0)

    return mrec, resid, seenmap, ell_binned, m_autos*fact/pwb**2, s_autos*fact/pwb**2, m_cross*fact/pwb**2, s_cross*fact/pwb**2

def scaling_dust(freq1, freq2, sp_index=1.59): 
    '''
    Calculate scaling factor for dust contamination
    Frequencies are in GHz
    '''
    freq1 = float(freq1)
    freq2 = float(freq2)
    x1 = freq1 / 56.78
    x2 = freq2 / 56.78
    S1 = x1**2. * np.exp(x1) / (np.exp(x1) - 1)**2.
    S2 = x2**2. * np.exp(x2) / (np.exp(x2) - 1)**2.
    vd = 375.06 / 18. * 19.6
    scaling_factor_dust = (np.exp(freq1 / vd) - 1) / \
                          (np.exp(freq2 / vd) - 1) * \
                          (freq2 / freq1)**(sp_index + 1)
    scaling_factor_termo = S1 / S2 * scaling_factor_dust
    return scaling_factor_termo


def dust_spectra(ll, nu):
    fact = (ll * (ll + 1)) / (2 * np.pi)
    coef = 1.39e-2
    spectra_dust = [np.zeros(len(ll)), 
                  coef * (ll / 80.)**(-0.42) / (fact * 0.52), 
                  coef * (ll / 80.)**(-0.42) / fact, 
                  np.zeros(len(ll))]
    sc_dust = scaling_dust(150, nu)
    return fact * sc_dust * spectra_dust





