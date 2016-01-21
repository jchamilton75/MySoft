from __future__ import division
from pylab import *
import healpy as hp
from matplotlib.pyplot import *
import numpy as np
import os
import sys
import qubic
import pycamb
import string
import random
import qubic
from scipy.constants import c
import scipy.integrate
from Homogeneity import SplineFitting



def give_baselines(inst, freqHz=150e9, doplot=False):
	dh = inst.horn.spacing
	kappa = 1.344 #inst.horn.kappa
	wavelength = c / freqHz
	nh = len(inst.horn)
	xyhorns = inst.horn.center

	### Get all baselines
	nbs = nh * (nh-1) // 2
	bs = np.zeros((nbs, 2))
	k=0
	for i in xrange(nh):
		if inst.horn.open[i]:
			for j in xrange(i+1,nh):
				if inst.horn.open[j]:
					bs[k] = (xyhorns[j,:2]-xyhorns[i,:2]) / wavelength
					k += 1
	### remove zero baseline
	mask = np.sum(bs**2,axis=1) != 0
	bs = bs[mask,:]
	nbs = mask.sum()

	### Sort baselines according to uv location
	#isort = np.lexsort((bs[:, 0], bs[:, 1]))
	isort = np.lexsort((np.round(10000*bs[:, 0]), np.round(10000*bs[:, 1])))
	bs_sorted = bs[isort]
	diff = np.sum((bs_sorted[1:] - bs_sorted[:-1])**2, axis=-1)

	eps = inst.horn.spacing / wavelength /10000
	ichange = np.r_[np.where(diff > eps)[0], nbs - 1]
	bs_unique = bs_sorted[ichange]
	nbs_unique = np.diff(np.r_[0, ichange + 1])
	index_baselines = []
	s = np.r_[0, ichange]
	for i in xrange(len(bs_unique)):
		index_baselines.append(isort[s[i]:s[i+1]])

	ellbs = 2*np.pi*np.sqrt(bs[:,0]**2+bs[:,1]**2)
	ellbs_unique = 2*np.pi*np.sqrt(bs_unique[:,0]**2+bs_unique[:,1]**2)


	if doplot:
		clf()
		du = np.min(np.sqrt(bs_unique[:,0]**2+bs_unique[:,1]**2))
		plot(ellbs_unique,nbs_unique,'ro')
		ell = np.arange(1+np.ceil(np.max(ellbs_unique)))
		plot(ell, (np.sqrt(nh)-np.sqrt(2)/2-ell/2/np.pi/du)*sqrt(nh),'k')

	isort = np.argsort(ellbs_unique)
	return bs, ellbs, bs_unique[isort], ellbs_unique[isort], nbs_unique[isort]


def get_kappa1(ell,omega,nu,deltanu,nn=10000,gaussian=False):
	if gaussian:
		signu=deltanu/sqrt(2*np.pi)
		nuvals=linspace(nu-deltanu,nu+deltanu,nn)
		filt=1./deltanu*np.exp(-(nuvals-nu)**2/(2*signu**2))
	else:
	    nuvals=linspace(nu-deltanu/2,nu+deltanu/2,nn)
    	filt=np.ones(nn)/deltanu


	values = np.zeros(len(ell))
	for i in xrange(len(ell)):
		u0 = ell[i] / (2 * np.pi)
		lobe = omega * np.exp(-omega * np.pi * (u0 * (nuvals-nu) / nu)**2)
		values[i] = scipy.integrate.trapz(lobe * filt, nuvals)

	kappa1 = omega / values
	return kappa1



def errbi(ll, deltal, omega, fsky, nh, wpixel, nu, dnu_nu, epsilon, eta, net_polar, time, cl=1, nmodules=1, lmin=0, neq_nh=False):
	if dnu_nu is 0:
		kappa1 = 1
	else:
		deltanu = nu * dnu_nu
		kappa1 = get_kappa1(ll,omega,nu,deltanu,nn=100)

	firstfac = np.sqrt(2 * kappa1 / ((2 * ll + 1) * fsky * deltal))
	samplevar=firstfac*cl

	fskyinit = omega / (4 * np.pi)
	if neq_nh is False:
		l0 = 2 * np.sqrt(nh) / np.sqrt(fskyinit)
		neq_nh = (1. - np.sqrt(2) / (2 * np.sqrt(nh)) - ll / l0)
		mm = neq_nh < 0
		neq_nh[mm] = 0


	omegascan = 4 * np.pi * fsky
	noisevar = firstfac * eta / neq_nh**2 / nh * net_polar**2 * omegascan * kappa1 / wpixel / epsilon / time
	noisevar = noisevar / nmodules

	deltacl = samplevar + noisevar

	ellmin = np.min(ll)
	truc = np.nan_to_num(cl**2 / (deltacl * sqrt(deltal))**2)
	mask = ll > lmin
	snr = sqrt(sum(truc[mask]))
	deltacl[~mask]=0
	noisevar[~mask]=0
	samplevar[~mask]=0

	return deltacl, noisevar, samplevar, snr


def errbi_bins(ellbins, ll, 
	omega=2*pi*(np.radians(12)/2.35)**2, 
	fsky=0.01, nh=400, wpixel=1, 
	nu=150e9, dnu_nu=0.25, epsilon=0.3, eta=1.6, net_polar=500, time=365.*24.*3600., 
	cl=1, nmodules=1, lmin=25, neq_nh=False):
	ellmin = ellbins[:len(ellbins)-1]
	ellmax = ellbins[1:len(ellbins)]
	ellval = 0.5 * (ellmin + ellmax)

	deltacl, noisevar, samplevar, truc = errbi(ll, 1, omega, fsky, nh, wpixel, nu, dnu_nu, epsilon, eta, net_polar, time, cl=cl, nmodules=nmodules, lmin=lmin, neq_nh=neq_nh)

	thedeltacl = np.zeros(len(ellval))
	thesamplevar = np.zeros(len(ellval))
	thenoisevar = np.zeros(len(ellval))
	for i in xrange(len(ellval)):
		mask = (ll >= ellmin[i]) & (ll < ellmax[i])
		thedeltacl[i] = np.sqrt(np.sum(deltacl[mask]**2))/np.sum(mask)
		thesamplevar[i] = np.sqrt(np.sum(samplevar[mask]**2))/np.sum(mask)
		thenoisevar[i] = np.sqrt(np.sum(noisevar[mask]**2))/np.sum(mask)

	return ellval, thedeltacl, thenoisevar, thesamplevar




def errim(ll, omega, fsky, nbols, wpixel, epsilon, eta, net_polar, time, fwhmdeg, cl=1, lmin=0):

	firstfac = np.sqrt(2 / ((2 * ll + 1) * fsky))
	samplevar=firstfac*cl

	omegascan = 4 * np.pi * fsky
	bl = np.exp(-0.5 * ll * (ll + 1) * np.radians(fwhmdeg/2.35)**2)
	noisevar = firstfac * eta * net_polar**2 * omegascan / wpixel / epsilon / time / nbols / bl**2

	deltacl = samplevar + noisevar

	ellmin = np.min(ll)
	truc = np.nan_to_num(cl**2 / deltacl**2)
	mask = ll > lmin
	snr = sqrt(sum(truc[mask]))
	deltacl[~mask]=0
	noisevar[~mask]=0
	samplevar[~mask]=0
	return deltacl, noisevar, samplevar, snr




def give_imager_errors(ellbins, ll, cl, fwhmdeg=0.1, fsky=0.01, nbols=1, wpixel=1, epsilon=0.3, eta=1.6, net_polar=20, time=365.*24.*3600., lmin=25):
	### bins
	ellmin = ellbins[:len(ellbins)-1]
	ellmax = ellbins[1:len(ellbins)]
	ellval = 0.5 * (ellmin + ellmax)
	### errors
	omega = fsky * 4 * np.pi
	deltacl, noisevar, samplevar, snr= errim(ll, omega, fsky, nbols, wpixel, epsilon, eta, net_polar, time, fwhmdeg, cl=cl, lmin=lmin)
	### bin them
	thedeltacl = np.zeros(len(ellval))
	thesamplevar = np.zeros(len(ellval))
	thenoisevar = np.zeros(len(ellval))
	for i in xrange(len(ellval)):
		mask = (ll >= ellmin[i]) & (ll < ellmax[i])
		thedeltacl[i] = np.sqrt(np.sum(deltacl[mask]**2))/np.sum(mask)
		thesamplevar[i] = np.sqrt(np.sum(samplevar[mask]**2))/np.sum(mask)
		thenoisevar[i] = np.sqrt(np.sum(noisevar[mask]**2))/np.sum(mask)
	return ellval, thedeltacl, thenoisevar, thesamplevar, 0, 0, 0





def give_qubic_errors(inst, ellbins, ll, cl, 
	fsky=0.01, nh=400, wpixel=1, 
	nu=150e9, dnu_nu=0.25, epsilon=0.3, eta=1.6, net_polar=500, time=365.*24.*3600., 
	nmodules=1, lmin=25, neq_nh=False, plot_baselines=False, symplot='ro'):

	nh = len(inst.horn)
	omega = 2*pi*(np.radians(inst.primary_beam.fwhm*180/np.pi)/2.35)**2
	baselines = 0
	if neq_nh is False:
		print('calculating baselines')
		bs, ellbs, bs_unique, ellbs_unique, nbs_unique = give_baselines(inst, freqHz=nu)
		baselines = [bs, ellbs, bs_unique, ellbs_unique, nbs_unique]
		neq_nh=SplineFitting.MySplineFitting(ellbs_unique,nbs_unique/nh,nbs_unique*0+1,5)
		ell = np.arange(np.max(ellbs_unique)-1)+1
		if plot_baselines: 
			subplot(2,1,1)
			plot(bs_unique[:,0],bs_unique[:,1],symplot,alpha=0.1, ms=3)
			subplot(2,1,2)
			plot(ellbs_unique,nbs_unique/nh,symplot, alpha=0.2)
			plot(ell, neq_nh(ell),'g',lw=3)

	ellav, deltacl, noisevar, samplevar= errbi_bins(ellbins, ll,
		omega=omega, fsky=fsky, nh=nh, wpixel=wpixel, 
		nu=nu, dnu_nu=dnu_nu, epsilon=epsilon, eta=eta, net_polar=net_polar, 
		time=time, cl=cl, nmodules=nmodules, lmin=lmin, neq_nh=neq_nh(ll))

	spec = np.interp(ellav, ll, cl*(ll*(ll+1))/(2*np.pi))
	nbsig = np.sqrt(np.sum(spec**2/(deltacl*(ellav*(ellav+1))/(2*np.pi))**2))

	
	return ellav, deltacl, noisevar, samplevar, neq_nh, nbsig, baselines



def give_errors(inst, ellbins, ll, cl, type, 
	fsky=0.01, nh=400, wpixel=1, 
	nu=150e9, dnu_nu=0.25, epsilon=0.3, eta=1.6, net_polar=500, time=365.*24.*3600., 
	nmodules=1, lmin=25, neq_nh=False, plot_baselines=False, symplot='ro',fwhmdeg=0.075, nbols=1):
	if type is 'bi':
		return give_qubic_errors(inst, ellbins, ll, cl, fsky=fsky, nh=nh, wpixel=wpixel, nu=nu, dnu_nu=dnu_nu, epsilon=epsilon, eta=eta, net_polar=net_polar, time=time, nmodules=nmodules, lmin=lmin, neq_nh=neq_nh, plot_baselines=plot_baselines, symplot=symplot)
	if type is 'im':
		return give_imager_errors(ellbins, ll, cl, fwhmdeg=fwhmdeg, fsky=fsky, nbols=nbols, wpixel=wpixel, epsilon=epsilon, eta=eta, net_polar=net_polar, time=time, lmin=lmin)





