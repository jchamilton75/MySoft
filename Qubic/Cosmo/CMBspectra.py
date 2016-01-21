from __future__ import division

import healpy as healpy
from matplotlib.pyplot import *
import numpy as np
import pycamb
#from qubic.utils import progress_bar
from Homogeneity import fitting
from Cosmo import FisherNew

########################## DEFAULT PARAMETERS #################################################
MAXELL = 3000
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
H0 = 67.04
omegab = 0.022032
omegac = 0.12038
h2 = (H0/100.)**2
scalar_amp = np.exp(3.098)/1.E10
omegav = h2 - omegab - omegac
Omegab = omegab/h2
Omegac = omegac/h2
rvalue =0.2
CambDefaultParams = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':rvalue,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False,
         'tensor_index':-rvalue/8}
################################################################################################

def muKarcmin(NET, nbols, fsky, fwhmdeg, duration=1.):
	nbsec = duration * 365 * 24 *3600
	reselt_deg2 = np.pi * (fwhmdeg / 2.35)**2
	nfwhm = fsky * (180 / np.pi)**2 * 4 * np.pi / reselt_deg2
	mukarcmin = np.sqrt(NET**2 * nfwhm / nbsec * (fwhmdeg * 60)**2 / nbols)
	return mukarcmin 

def get_camb_spectra(cambparams, maxell):
	#### Call CAMB: first without lensing and second with lensing
	print("")
	print("Calling Camb (Unlensed) with lmax = {0:.0f}".format(maxell+200))
	thepars = cambparams.copy()
	thepars['DoLensing'] = False
	Tnl,Enl,Bnl,Xnl = pycamb.camb(np.max(maxell)+200,**thepars)
	Bprim = Bnl[0:np.max(maxell)]
	#### Camb for lensing B
	thepars = cambparams.copy()
	thepars['DoLensing'] = True
	print("Calling Camb (Lensed) with lmax = {0:.0f}".format(maxell+200))
	Tl,El,Bl,Xl = pycamb.camb(np.max(maxell)+200,**thepars)
	Tl = Tl[0:np.max(maxell)]	
	El = El[0:np.max(maxell)]	
	Xl = Xl[0:np.max(maxell)]	
	Bl = Bl[0:np.max(maxell)]	
	#### Get Pure lensing B modes
	Bl = Bl - Bprim
	#### build spectra list
	x = np.arange(len(Bl)) + 1
	basic_spectra = [x, Tl, El, Xl, Bprim, Bl]
	return basic_spectra

def update_dict(new, init):
	out = init.copy()
	initkeys = init.keys()
	newkeys = new.keys()
	for k in newkeys:
		if k in initkeys:
			out[k] = new[k]
	return out

def get_spectra(pars, cambparams=CambDefaultParams, basic_spectra=False, maxellcamb=MAXELL):
	#### First get CAMB spectra if needed
	if basic_spectra is False:
		newcambparams = update_dict(pars, cambparams)
		basic_spectra = get_camb_spectra(newcambparams, maxellcamb)
	#### Get the spectra for T, E, X, B
	ell = basic_spectra[0]
	T = basic_spectra[1]
	E = basic_spectra[2]
	X = basic_spectra[3]
	B = basic_spectra[4] + pars['lensing_residual'] * basic_spectra[5]

	#### Then get foregrounds spectra (zero for now)
	Tfg = np.zeros(len(basic_spectra[0]))
	Efg = np.zeros(len(basic_spectra[0]))
	Xfg = np.zeros(len(basic_spectra[0]))
	Bfg = np.zeros(len(basic_spectra[0]))
	#### Now combine add the foregrounds
	if 'ampTfg' in pars.keys():
		T += pars['ampTfg'] * Tfg
	if 'ampEfg' in pars.keys():
		E += pars['ampEfg'] * Efg
	if 'ampXfg' in pars.keys():
		X += pars['ampXfg'] * Xfg
	if 'ampBfg' in pars.keys():
		B += pars['ampBfg'] * Bfg
	#### return the spectra
	spectra = [ell, T, E, X, B]
	return spectra, basic_spectra

def get_binned_spectra(pars, lc, lmin, lmax, cambparams=CambDefaultParams, basic_spectra=False, maxellcamb=MAXELL):
	spectra, basic_spectra = get_spectra(pars, cambparams=CambDefaultParams, basic_spectra=basic_spectra, maxellcamb=MAXELL)
	binned_spectra = bin_spectra(lc, lmin, lmax, spectra)
	return binned_spectra, spectra, basic_spectra

def binspec(ll,cl,lc,lmin,lmax):
	clbinned = np.zeros_like(lc)
	for i in range(len(lc)):
		mask = (ll >= lmin[i]) & (ll <= lmax[i])
		fact = (ll[mask]*(ll[mask]+1))/(lc[i]*(lc[i]+1))
		clbinned[i] = np.mean(cl[mask]/fact)
	return clbinned

def bin_spectra(lc, lmin, lmax, spectra):
	return [lc, 
			binspec(spectra[0], spectra[1], lc, lmin, lmax), 
			binspec(spectra[0], spectra[2], lc, lmin, lmax), 
			binspec(spectra[0], spectra[3], lc, lmin, lmax), 
			binspec(spectra[0], spectra[4], lc, lmin, lmax)]

def err_binned_spectra(lc, spectra, fsky, mukarcminT, bl_or_fwhmdeg, deltal=1):
	ell = spectra[0]
	Tl = spectra[1]/(ell*(ell+1)/(2*np.pi))
	El = spectra[2]/(ell*(ell+1)/(2*np.pi))
	Xl = spectra[3]/(ell*(ell+1)/(2*np.pi))
	Bl = spectra[4]/(ell*(ell+1)/(2*np.pi))
	return [lc,
			th_errors_bins(lc, ell, Tl, fsky, mukarcminT, bl_or_fwhmdeg, deltal=deltal), 
			th_errors_bins(lc, ell, El, fsky, mukarcminT*np.sqrt(2), bl_or_fwhmdeg, deltal=deltal), 
			th_errors_bins(lc, ell, Xl, fsky, mukarcminT*np.sqrt(2), bl_or_fwhmdeg, deltal=deltal),
			th_errors_bins(lc, ell, Bl, fsky, mukarcminT*np.sqrt(2), bl_or_fwhmdeg, deltal=deltal)]

def th_errors(ell, cell, fsky, mukarcminT, bl_or_fwhmdeg, deltal=1):
	if np.array(bl_or_fwhmdeg).size == 1:
		fwhmdeg = bl_or_fwhmdeg
		bl = np.exp(-0.5*ell*(ell+1)*np.radians(fwhmdeg/2.35)**2)
	else:
		bl = bl_or_fwhmdeg
	factor = np.sqrt(2./((2*ell+1)*fsky*deltal))
	noiseterm = np.abs(factor*(np.radians(mukarcminT/60))**2/bl**2)
	sampleterm = np.abs(factor*cell)
	sumterm = sampleterm + noiseterm
	min_ell = np.int(180/np.degrees((np.sqrt(2*fsky))))
	mask = (ell >= min_ell)
	sampleterm[~mask] = 1e30
	sumterm[~mask] = 1e30
	return [sampleterm, noiseterm, sumterm]

def th_errors_bins(ellcenter, ell, cell, fsky, mukarcmin, bl_or_fwhmdeg, deltal=1):
	st,nt,tt = th_errors(ell, cell, fsky, mukarcmin, bl_or_fwhmdeg, deltal=deltal)
	ist = np.interp(ellcenter,ell,st)*ellcenter*(ellcenter+1)/(2*np.pi)
	int = np.interp(ellcenter,ell,nt)*ellcenter*(ellcenter+1)/(2*np.pi)
	itt = np.interp(ellcenter,ell,tt)*ellcenter*(ellcenter+1)/(2*np.pi)
	return [ist, int, itt]

def return_bins_and_errors(pars, args):
	## Decrypt extra-arguments
	lcenter = args[0]
	lmin = args[1]
	lmax = args[2]
	fsky = args[3]
	mukarcminT = args[4]
	fwhmdeg = args[5]
	deltal = args[6]
	if len(args) >= 8:
		cambparams = args[7]
	else:
		cambparams = CambDefaultParams
	if len(args) >= 9:
		basic_spectra = args[8]
	else:
		basic_spectra = False
	if len(args) >= 10:
		maxell = args[9]
	else:
		maxell = MAXELL
	## get binned spectra
	binned, spec, basic_spec = get_binned_spectra(pars, lcenter, lmin, lmax, cambparams=cambparams, basic_spectra=basic_spectra, maxellcamb=maxell)
	## get binned errors
	errbinned = err_binned_spectra(lcenter, spec, fsky, mukarcminT, fwhmdeg, deltal=deltal)
	## prepare outputs
	nbins = len(lcenter)
	data = np.zeros(4*nbins)
	error = np.zeros(4*nbins)
	for i in np.arange(4):
		data[i*nbins:(i+1)*nbins] = binned[i+1]
		error[i*nbins:(i+1)*nbins] = errbinned[i+1][2]
	return data, error

def fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, deltal=1, cambparams=CambDefaultParams, maxellcamb=MAXELL, der=False, basic_spectra=False, errors=False, noT=False, noE=False, noTE=False, noB=False, minell=False, maxell=False, varnames=False, plotmat=False):
	nbins = maxellcamb/deltal
	lmin = 1+(deltal)*np.arange(nbins)	
	lmax = lmin+deltal-1
	lcenter = (lmax+lmin)/2

	##### Generate basic spectra
	if basic_spectra is False:
		print("Calculate Basic Spectra")
		binned, spec, basic_spectra = get_binned_spectra(pars, lcenter, lmin, lmax)

	##### Generate derivatives
	if der is False:
		print("Calculate Derivatives")
		args = [lcenter, lmin, lmax, fsky, mukarcminT, fwhmdeg, deltal, CambDefaultParams, False, MAXELL]
		der = FisherNew.give_derivatives(pars, args, return_bins_and_errors)

	##### Calculate Error bars if needed
	args = [lcenter, lmin, lmax, fsky, mukarcminT, fwhmdeg, deltal, CambDefaultParams, basic_spectra, MAXELL]
	data, theerrors = return_bins_and_errors(pars, args)
	nbins = len(data)/4
	allell = np.ravel(np.array([lcenter, lcenter, lcenter, lcenter]))
	allellmin = np.ravel(np.array([lmin, lmin, lmin, lmin]))
	allellmax = np.ravel(np.array([lmax, lmax, lmax, lmax]))
	##### Update errors
	if errors is not False:
		theerrors = errors
	if noT is True:
		theerrors[0*nbins:1*nbins] = 1e30
	if noE is True:
		theerrors[1*nbins:2*nbins] = 1e30
	if noTE is True:
		theerrors[2*nbins:3*nbins] = 1e30
	if noB is True:
		theerrors[3*nbins:4*nbins] = 1e30
	if minell is not False:
		mask = allell < minell
		theerrors[mask] = 1e30
	if maxell is not False:
		mask = allell > maxell
		theerrors[mask] = 1e30
	### Remove low ell not allowed by fsky
	fskyminell = np.int(180/np.degrees((np.sqrt(2*fsky))))
	mask = allellmin < fskyminell
	theerrors[mask] = 1e30

	##### Calculate Fisher Matrix
	fm = FisherNew.fishermatrix(pars, args, der, theerrors, return_bins_and_errors)

	if plotmat:
		clf()
		a0,leg0 = FisherNew.plot_fisher(fm, pars,'b', varnames=varnames, onesigma=True)
		subplot(2,2,2)
		axis('off')
		legend([a0],[leg0],loc='upper right',fontsize=10)

	return fm, basic_spectra, der, lcenter, data, theerrors


