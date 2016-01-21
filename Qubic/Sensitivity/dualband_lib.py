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
import scipy.constants
from Sensitivity import qubic_sensitivity
from Cosmo import interpol_camb as ic
from scipy import interpolate
from scipy import integrate


def Bnu(nuGHz, temp):
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k
	nu = nuGHz*1e9
	return 2 * h * nu**3 / c**2 / (np.exp(h * nu / k / temp) - 1)

def dBnu_dT(nuGHz, temp):
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k
	nu = nuGHz*1e9
	theBnu = Bnu(nuGHz, temp)
	return (theBnu * c / nu / temp)**2 / 2 * np.exp(h * nu / k / temp) / k

def mbb(nuGHz, beta, temp):
	return (nuGHz/353)**beta * Bnu(nuGHz, temp)

def KCMB2MJy_sr(nuGHz):
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k
	T = 2.725
	nu = nuGHz*1e9
	x = h * nu / k / T
	ex = np.exp(x)
	fac_in = dBnu_dT(nuGHz, T)
	fac_out = 1e20
	return fac_in * fac_out

def freq_conversion(nuGHz_in, nuGHz_out, betadust, Tdust):
	val_in = KCMB2MJy_sr(nuGHz_in) / mbb(nuGHz_in, betadust, Tdust)
	val_out = KCMB2MJy_sr(nuGHz_out) / mbb(nuGHz_out, betadust, Tdust)
	return val_in / val_out


def Dl_BB_dust(ell, freqGHz1, freqGHz2=None, params = None):
	if params is None: params = [13.4 * 0.45, -2.42, 1.59, 19.6]
	if freqGHz2 is None: freqGHz2=freqGHz1
	Dl_353_ell80 = params[0]
	alpha_bb = params[1]
	betadust = params[2]
	Tdust = params[3]
	return Dl_353_ell80 * (freq_conversion(353, freqGHz1, betadust, Tdust) * freq_conversion(353, freqGHz2, betadust, Tdust)) * (ell/80)**(alpha_bb+2)

def Dl_BB_dust_bins(ellbins, freqGHz1, freqGHz2=None, params = None):
	if freqGHz2 is None: freqGHz2=freqGHz1
	res = np.zeros(len(ellbins)-1)
	for i in xrange(len(ellbins)-1):
		lvals = ellbins[i]+np.arange(ellbins[i+1]-ellbins[i])
		dlvals = Dl_BB_dust(lvals, freqGHz1, freqGHz2, params=params)
		res[i] = np.mean(dlvals)
	return res

def get_ClBB_cross_th(lll, freqGHz1, freqGHz2=None, dustParams = None, rvalue=0.05, ClBBcosmo=None, camblib=None):
	fact = (lll*(lll+1))/(2*np.pi)
	if ClBBcosmo is None:
		if camblib is None:
			print('Calling CAMB')
			### Call Camb for primordial spectrum
			H0 = 67.04
			omegab = 0.022032
			omegac = 0.12038
			h2 = (H0/100.)**2
			scalar_amp = np.exp(3.098)/1.E10
			omegav = h2 - omegab - omegac
			Omegab = omegab/h2
			Omegac = omegac/h2
			params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,'reion__use_optical_depth':True,'reion__optical_depth':0.0925,'tensor_ratio':rvalue,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}
			lmaxcamb = np.max(lll)
			T,E,B,X = pycamb.camb(lmaxcamb+1+150,**params)
			B=B[:lmaxcamb+1]
		else:
			B=ic.get_Dlbb_fromlib(lll, rvalue, camblib)
		B=B[:np.max(lll)+1]
		ClBBcosmo = B/fact
	### Get dust component
	dl_dust = Dl_BB_dust(lll, freqGHz1, freqGHz2, params=dustParams)
	ClBBdust = dl_dust/fact
	### sum them
	ClBBtot = ClBBcosmo + ClBBdust
	return ClBBtot, ClBBcosmo, ClBBdust


def get_multiband_covariance(instrument_info, r, ClBBcosmo=None, doplot=False, dustParams=None, verbose=False, all_neq_nh=None, camblib=None):
	### Decrypt instrument info
	inst, ellbins, freqs, type, NETs, fsky, duration, epsilon, name, col = instrument_info

	### Dust Parameter
	if dustParams is None:
		dldust_80_353 = 13.4 * 0.45 #to match Planck XXX on BICEP2
		alphadust = -2.42
		betadust = 1.59
		Tdust = 19.6
		dustParams = [dldust_80_353, alphadust, betadust, Tdust]

	### Binning
	lmax = np.max(ellbins)
	ell = np.arange(lmax+1)
	fact = (ell*(ell+1))/(2*np.pi)
	ellmin = np.array(ellbins[:len(ellbins)-1])
	ellmax = np.array(ellbins[1:len(ellbins)])
	ellav = 0.5 * (ellmax + ellmin)
	### cross-spectra
	allspec = []
	allfreqs = []
	samplevars = []
	noisevars = []
	allvars = []
	neq_nh_ij = []
	num = 0
	for i in np.arange(len(freqs)):
		for j in np.arange(i, len(freqs)):
			thecross = np.str(freqs[i])+'x'+np.str(freqs[j])
			if verbose: print('Doing '+thecross)
			### get cross-spectrum
			spectrum = get_ClBB_cross_th(ell, freqs[i], freqGHz2=freqs[j], rvalue=r, dustParams=dustParams, ClBBcosmo=ClBBcosmo, camblib=camblib)
			ClBBcosmo = spectrum[1]
			spec = np.interp(ellav, ell, spectrum[0]*(ell*(ell+1))/(2*np.pi))
			### get auto errors for i and j assuming that the spectrum is that of the cross
			if all_neq_nh:
				neq_nh_i = all_neq_nh[num][0]
				neq_nh_j = all_neq_nh[num][1]
			else:
				neq_nh_i = False
				neq_nh_j = False
			resi = qubic_sensitivity.give_errors(inst, ellbins, ell, spectrum[0], type[i], net_polar=NETs[i], neq_nh=neq_nh_i,fsky=fsky, time=duration[i], epsilon=epsilon[i])
			resj = qubic_sensitivity.give_errors(inst, ellbins, ell, spectrum[0], type[j], net_polar=NETs[j], neq_nh=neq_nh_j,fsky=fsky, time=duration[j], epsilon=epsilon[j])
			neq_nh_ij.append([resi[4], resj[4]])
			### now combine them
			samplevars.append(np.sqrt(resi[3]*resj[3])*(ellav*(ellav+1))/(2*np.pi))
			noisevars.append(np.sqrt(resi[2]*resj[2])*(ellav*(ellav+1))/(2*np.pi))
			allvars.append(np.sqrt(resi[3]*resj[3])*(ellav*(ellav+1))/(2*np.pi) + np.sqrt(resi[2]*resj[2])*(ellav*(ellav+1))/(2*np.pi))
			allspec.append(spec)
			allfreqs.append(thecross)
			num = num + 1

	if doplot:
		col = get_cmap('jet')(np.linspace(0, 1.0, len(allspec))[::-1])
		clf()
		subplot(1,2,1)
		yscale('log')
		#xscale('log')
		xlim(10,500)
		ylim(1e-4,100)
		title('r = {0:3.2f}'.format(r))
		xlabel('$\ell$')
		ylabel(r'$\frac{\ell(\ell+1)}{2\pi}\,C_\ell$'+'    '+'$[\mu K^2]$ ')
		for i in np.arange(len(allspec)):
			errorbar(ellav, allspec[i], yerr=allvars[i], label = allfreqs[i], color=col[i],fmt='o')
		legend(loc='upper left',framealpha=0.6)
		subplot(1,2,2)
		for i in np.arange(len(allspec)):
			plot(ellav, allspec[i], lw=3,color=col[i])
			plot(ellav, allvars[i],color=col[i], label = allfreqs[i])
			plot(ellav, samplevars[i], "--", color=col[i])
			plot(ellav, noisevars[i], ":", color=col[i])
		xlabel('$\ell$')
		ylabel(r'$\frac{\ell(\ell+1)}{2\pi}\,\Delta C_\ell$'+'    '+'$[\mu K^2]$ ')
		legend(loc='upper left',framealpha=0.6)
		yscale('log')

	#### Now the covariance matrix...
	sz = len(allspec) * len(allspec[0])
	ncross = len(allspec)
	nbins = len(allspec[0])
	covmatnoise = np.zeros((sz,sz))
	covmatsample = np.zeros((sz,sz))

	for k in xrange(nbins):
		for i in xrange(len(allspec)):
			for j in xrange(len(allspec)):
				if i==j: covmatnoise[i*nbins+k,j*nbins+k] = noisevars[i][k] * noisevars[j][k]			
				covmatsample[i*nbins+k,j*nbins+k] = samplevars[i][k] * samplevars[j][k]

	covmat = covmatsample + covmatnoise

	return covmat, covmatnoise, covmatsample, np.ravel(np.array(allspec)), np.ravel(np.array(allvars)), ClBBcosmo, neq_nh_ij


def like_1d(specin, parindex, values, instrument_info, camblib=None, paramsdefault=None, normalize=False, DoPlot=True, CL=0.6827):
	### Decrypt instrument info
	inst, ellbins, freqs, type, NETs, fsky, duration, epsilon, name, col = instrument_info

	### Default parameters (for thos that are not fixed)
	if paramsdefault is None:
		r=0.05
		dldust_80_353 = 13.4 * 0.45 #to match Planck XXX on BICEP2
		alphadust = -2.42
		betadust = 1.59
		Tdust = 19.6
		paramsdefault = np.array([r, dldust_80_353, alphadust, betadust, Tdust])
	nvals = len(values)
	like = np.zeros(nvals)
	for i in xrange(nvals):
		thepars = paramsdefault.copy()
		thepars[parindex] = values[i]
		rval = thepars[0]
		dpars = thepars[1:]
		if i==0: all_neq_nh = None
		covmat, covmatnoise, covmatsample, allspec, allvars, ClBBcosmo, all_neq_nh = get_multiband_covariance(instrument_info, rval, 
            dustParams=dpars, all_neq_nh=all_neq_nh, camblib=camblib)
		covmat = covmatsample+covmatnoise
		invcov = np.linalg.inv(covmat)
		resid = specin - allspec
		chi2 = np.sum(np.dot(resid,invcov)*resid)
		fact=1
		if normalize: fact = 1./(np.sqrt(2*np.pi)**(len(resid)/2))/np.sqrt(np.linalg.det(covmat))
		like[i] = fact*np.exp(-0.5*chi2)
		print(i, nvals, like[i], chi2)
		#clf()
		#err =np.sqrt(np.diag(covmat))
		#ellvals = 0.5 * (ellbins[:-1]+ellbins[1:])
		#errorbar(ellvals,specin,yerr=err,fmt='ro')
		#plot(ellvals,allspec,'k')
		#title(str(chi2))
		#draw()

	like=like/integrate.trapz(like,values)
	m, lo, hi = statlike(values,like, CL=CL)
	if DoPlot:
		if col is None: col = 'r'
		if name is None: name = str(freqs)
		plot(values,like, label = name[0]+' : {0:4.3f} + {1:4.3f} - {2:4.3f}'.format(m,hi-m,m-lo), color=col,lw=3)
		fill_between([lo,hi],[1000,1000],y2=[0,0], color=col,alpha=0.1)
		plot([m,m],[0,1000],color=col)
		plot([hi,hi],[0,1000],'--',color=col)
		plot([lo,lo],[0,1000],'--',color=col)
		ylim(0,np.max(like)*1.1)
		legend()
		draw()
	return like, m, m-lo, hi-m


def like_1d_marginalize(specin, parindex, values, indexmarg, valmarg, instrument_info, camblib=None, 
    paramsdefault=None, normalize=False, DoPlot=True, CL=0.6827):
	### Decrypt instrument info
	inst, ellbins, freqs, type, NETs, fsky, duration, epsilon, name, col = instrument_info

	### Default parameters (for thos that are not fixed)
	if paramsdefault is None:
		r=0.05
		dldust_80_353 = 13.4 * 0.45 #to match Planck XXX on BICEP2
		alphadust = -2.42
		betadust = 1.59
		Tdust = 19.6
		paramsdefault = np.array([r, dldust_80_353, alphadust, betadust, Tdust])
	nvals = len(values)
	nmarg = len(valmarg)
	like = np.zeros(nvals)
	for i in xrange(nvals):
		if i==0: all_neq_nh = None
		sublike = np.zeros(nmarg)
		for j in xrange(nmarg):
			print(i, nvals, j, nmarg)
			thepars = paramsdefault.copy()
			thepars[parindex] = values[i]
			thepars[indexmarg] = valmarg[j]
			rval = thepars[0]
			dpars = thepars[1:]
			covmat, covmatnoise, covmatsample, allspec, allvars, ClBBcosmo, all_neq_nh = get_multiband_covariance(instrument_info, rval, dustParams=dpars, all_neq_nh=all_neq_nh, camblib=camblib)
			invcov = np.linalg.inv(covmat)
			resid = specin - allspec
			chi2 = np.sum(np.dot(resid,invcov)*resid)
			fact=1
			if normalize: fact = 1./(np.sqrt(2*np.pi)**(len(resid)/2))/np.sqrt(np.linalg.det(covmat))
			sublike[j] = fact*np.exp(-0.5*chi2)
		like[i] = integrate.trapz(sublike,valmarg)
	like=like/integrate.trapz(like,values)
	m, lo, hi = statlike(values,like, CL=CL)
	if DoPlot:
		if col is None: col = 'r'
		if name is None: name = str(freqs)
		plot(values,like, label = name[0]+' : {0:4.3f} + {1:4.3f} - {2:4.3f}'.format(m,hi-m,m-lo), color=col,lw=3)
		fill_between([lo,hi],[1000,1000],y2=[0,0], color=col,alpha=0.1)
		plot([m,m],[0,1000],color=col)
		plot([hi,hi],[0,1000],'--',color=col)
		plot([lo,lo],[0,1000],'--',color=col)
		ylim(0,np.max(like)*1.1)
		legend()
		draw()
	return like, m, m-lo, hi-m

#def likelihood_rdust(specin, instrument_info, rval, dustParams=None, all_neq_nh=None, camblib=None):



def statlike(x,like,nn=10,CL=0.6827):
	# resample nn times with splines to get the maximum
	xnew = linspace(np.min(x), np.max(x), nn*len(x))
	ynew = interpolate.spline(x,like,xnew)
	ynew = ynew / integrate.trapz(ynew,x=xnew)
	indexmax = where(ynew == np.max(ynew))[0]
	xmax = xnew[indexmax][0]

	### Integrate from xmax to upper values
	xhi = xnew[indexmax:]
	yhi = ynew[indexmax:]
	ihi = np.r_[0,integrate.cumtrapz(yhi,x=xhi)]
	maxhi = np.max(ihi)

	### Integrate from xmax to lower values
	xlo = xnew[:indexmax][::-1]
	ylo = ynew[:indexmax][::-1]
	if len(xlo)>0:
		ilo = np.r_[0,-integrate.cumtrapz(ylo,x=xlo)]
	else:
		ilo = 0
	maxlo = np.max(ilo)

	if (maxlo>(CL/2)) & (maxhi>(CL/2)):
		print('ok')
		lower = np.interp(CL/2,ilo,xlo)
		higher = np.interp(CL/2,ihi,xhi)
	else:
		if (maxlo<=(CL/2)):
			lower = np.min(x)
			higher = np.interp(CL-maxlo,ihi,xhi)
		elif (maxhi<=(CL/2)):
			higher = np.max(x)
			lower = np.interp(CL-maxhi,ilo,xlo)
		else:
			lower = np.nan
			higher = np.nan
	return xmax, lower, higher








