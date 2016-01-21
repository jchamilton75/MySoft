from __future__ import division

import healpy as healpy
from matplotlib.pyplot import *
import numpy as np
import pycamb
#from qubic.utils import progress_bar
from Homogeneity import fitting




##### Take one model and calculate bins and error bars from both noise and sample variance
def th_errors(ell, cell, fsky, mukarcmin, bl_or_fwhmdeg, deltal=1):
	if np.array(bl_or_fwhmdeg).size == 1:
		fwhmdeg = bl_or_fwhmdeg
		bl = np.exp(-0.5*ell*(ell+1)*np.radians(fwhmdeg/2.35)**2)
	else:
		bl = bl_or_fwhmdeg
	factor = np.sqrt(2./((2*ell+1)*fsky*deltal))
	noiseterm = factor*2*(np.radians(mukarcmin/60))**2/bl**2
	sampleterm = factor*cell
	return [sampleterm, noiseterm, sampleterm+noiseterm]

def th_errors_bins(ellcenter, ell, cell, fsky, mukarcmin, bl_or_fwhmdeg, deltal=1):
	st,nt,tt = th_errors(ell, cell, fsky, mukarcmin, bl_or_fwhmdeg, deltal=deltal)
	ist = np.interp(ellcenter,ell,st)*ellcenter*(ellcenter+1)/(2*np.pi)
	int = np.interp(ellcenter,ell,nt)*ellcenter*(ellcenter+1)/(2*np.pi)
	itt = np.interp(ellcenter,ell,tt)*ellcenter*(ellcenter+1)/(2*np.pi)
	return [ist, int, itt]


####### Functions to get spectra from camb (binned and unbinned)
def binspec(ll,cl,lc,lmin,lmax):
	clbinned = np.zeros_like(lc)
	for i in range(len(lc)):
		mask = (ll >= lmin[i]) & (ll <= lmax[i])
		fact = (ll[mask]*(ll[mask]+1))/(lc[i]*(lc[i]+1))
		clbinned[i] = np.mean(cl[mask]/fact)
	return clbinned


def get_spectrum_bins(pars, x, stuff, spectra=None):
	#### Parameters
	params = stuff[0]
	lmax = stuff[1]
	fsky = stuff[2]
	mukarcmin = stuff[3]
	fwhmdeg = stuff[4]
	deltal = stuff[5]
	xmin = stuff[6]
	xmax = stuff[7]
	consistency = stuff[8]
	parkeys = stuff[9]

	thepars = params.copy()
	for i in np.arange(len(parkeys)):
		thepars[parkeys[i]] = pars[i]
	#if consistency:
	#	thepars['tensor_index'] = -thepars['tensor_ratio']/thepars['consistency']


	if spectra is None:
		#### Camb for primordial B
		print("")
		print("Calling Camb for Primordial B-modes with lmax = {0:.0f}".format(lmax+200))
		for i in np.arange(len(parkeys)):
			print(parkeys[i], thepars[parkeys[i]])
		amplens = thepars['lensing_amplitude']
		Tprim,Eprim,Bprim,Xprim = pycamb.camb(np.max(lmax)+200,**thepars)
		Bprim = Bprim[0:np.max(lmax)]
		#### Camb for lensing B
		thepars = params.copy()
		thepars['tensor_ratio'] = 0
		thepars['tensor_index'] = 0
		thepars['DoLensing'] = True
		print("Calling Camb for Lensing B-modeswith lmax = {0:.0f}".format(lmax+200))
		Tl,El,Bl,Xl = pycamb.camb(np.max(lmax)+200,**thepars)
		Bl = Bl[0:np.max(lmax)]	
		#### Lensing B
		Blensed = amplens * Bl
		##### Make linear combination
		totBB = Bprim + Blensed
	else:
		Bprim = spectra[0]
		Blensed = thepars['lensing_amplitude'] * spectra[1]
		totBB = Bprim + Blensed
	##### Bin the spectra
	ell = np.arange(len(totBB))+1
	clprim_binned = binspec(ell, Bprim, x, xmin, xmax)
	cllens_binned = binspec(ell, Blensed, x, xmin, xmax)
	btot_binned = clprim_binned + cllens_binned
	##### Corresponding error bars
	dclsb, dclnb, dclb = th_errors_bins(x, ell, totBB/(ell*(ell+1)/(2*np.pi)), fsky, mukarcmin, fwhmdeg, deltal=deltal)
	return btot_binned, clprim_binned, cllens_binned, dclsb, dclnb, dclb, Bprim, Blensed, totBB 

##################### Fisher matrix analysis

def give_derivatives(pars, bins, thefunction, otherargs, delt=None):
	der = np.zeros((len(pars), len(bins)))
	if delt is None: delt = np.zeros(len(pars))+0.01
	basebins = thefunction(pars, bins, otherargs)[0]
	for i in range(len(pars)):
		vals = np.array([pars[i], pars[i]*(1+delt[i])])
		dvals = vals[1]-vals[0]
		thevalbins = np.zeros((2, len(bins)))
		thevalbins[0,:] = basebins
		thepars = pars.copy()
		thepars[i] = vals[1]
		thevalbins[1,:] = thefunction(thepars, bins, otherargs)[0]
		for k in range(len(bins)):
			bla = np.gradient(thevalbins[:,k])/dvals
			der[i,k]=bla[0]
	return der

def fishermatrix(pars, bins, errors, thefunction, otherargs,delt=None, der=None):
	if der is None: der = give_derivatives(pars, bins, thefunction, otherargs, delt=delt)
	fm = np.zeros((len(pars), len(pars)))
	for k in range(len(bins)):
		for i in range(len(pars)):
			for j in range(len(pars)):
				fm[i,j] += der[i,k]*der[j,k]/errors[k]**2
	return fm, der


def cont_from_fisher2d(fisher2d, center, color='k',size=100, onesigma=False):
	x0 = center[0]
	y0 = center[1]
	covmat = np.linalg.inv(fisher2d)
	sigs = np.sqrt(np.diag(covmat))
	xx = np.linspace(-5*sigs[0], 5*sigs[0], size)+x0
	yy = np.linspace(-5*sigs[1], 5*sigs[1], size)+y0
	ch2 = np.zeros((size, size))
	for i in range(size):
		for j in range(size):
			vec = np.array([xx[i]-x0,yy[j]-y0])
			bla = np.dot(vec.T,np.dot(fisher2d,vec))
			ch2[j,i]=bla
	contour(xx,yy,ch2,levels=np.array([2.275]),colors=color,linestyles='solid',linewidths=2)
	if not onesigma:
		contour(xx,yy,ch2,levels=np.array([5.99]),colors=color,linestyles='dashed',linewidths=2)
	a=Rectangle((np.max(xx),np.max(yy)),1e-6,1e-6,fc=color)
	return(a)


def submatrix(mat,cols):
	return mat[:,cols][cols,:]


def plot_fisher(fisherin, centervals, labelsin, col, limits=None,nbins=None,fixed=None,size=256, onesigma=False):
	fisher = fisherin.copy()
	mm = np.array(centervals)
	labels = np.array(labelsin)
	ninit = fisher.shape[0]
	#### Cut the fisher matrix if some parameters are fixed
	if fixed is None:
		nn = ninit
		covar = np.linalg.inv(fisher + np.random.rand(ninit,ninit)*1e-10)
		ss  = np.sqrt(np.diag(covar))
	else:
		all = np.arange(ninit).tolist()
		if type(fixed) is list:
			for ii in fixed: all.remove(ii)
		else:
			all.remove(fixed)
		nn = len(all)
		fisher = submatrix(fisher,all)
		mm = mm[all]
		labels = labels[all]
		covar = np.linalg.inv(fisher + np.random.rand(len(labels),len(labels))*1e-10)
		ss  = np.sqrt(np.diag(covar))

	#### New size of the fisher matrix
	nn = fisher.shape[0]

	#### Get limits
	if limits is None:
		limits=[]
		for i in np.arange(nn):
			limits.append([mm[i]-3*ss[i],mm[i]+3*ss[i]])

	### legend
	leg = ''
	for i in np.arange(nn):
		leg = leg + '$\sigma$('+labels[i]+')={0:.2g} '.format(ss[i])
	#### Now plot the matrix
	## First plot the individual likelihoods
	for i in np.arange(nn):
		a=subplot(nn,nn,i*nn+i+1)
		a.tick_params(labelsize=8)
		xlim(limits[i])
		ylim(0,1.2)
		xx=np.linspace(limits[i][0],limits[i][1],1000)
		yy=np.exp(-0.5*(xx-mm[i])**2/ss[i]**2)
		aa=plot(xx,yy,color=col,lw=2)
		title(labels[i])
		plot([mm[i], mm[i]],[0,2],'--',color=col)
		plot([mm[i]-ss[i], mm[i]-ss[i]],[0,2],':',color=col)
		plot([mm[i]+ss[i], mm[i]+ss[i]],[0,2],':',color=col)
		plot([mm[i]-2*ss[i], mm[i]-2*ss[i]],[0,2],':',color=col)
		plot([mm[i]+2*ss[i], mm[i]+2*ss[i]],[0,2],':',color=col)

	## Then plot the ellipses
	for i in np.arange(nn):
		for j in np.arange(nn):
			if (i > j):
				a=subplot(nn, nn, i*nn+j+1)
				a.tick_params(labelsize=8)
				xlim(limits[j])
				ylim(limits[i])
				subcov = submatrix(covar,[j,i])
				plot(mm[j],mm[i],'+',color=col)
				a0=cont_from_fisher2d(np.linalg.inv(subcov), [mm[j],mm[i]], size=size,color=col, onesigma=onesigma)
			if i == (nn-1):
				subplot(nn, nn, i*nn+j+1)
				xlabel(labels[j])
			if (j == 0) & (i != 0):	
				subplot(nn, nn, i*nn+j+1)		
				ylabel(labels[i])

	return a0,leg



def get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lensing_residuals, der=None, spectra = None, parkeys=None, nbins=None, min_ell=None, max_ell=None, deltal=None, plotcl = False, plotmat = False, consistency = True, fixed=None, noiseonly=False, prior=None, title=None):

	if parkeys is None:
		parkeys = ['tensor_ratio', 'scalar_index', 'tensor_index', 'lensing_amplitude']
		parnames = ['$r$', '$n_s$', '$n_t$','$a_l$']

	pars = np.zeros(len(parkeys))
	for i in np.arange(len(parkeys)):
		pars[i] =  params[parkeys[i]]

	pars[3] = lensing_residuals
	rvalue = params['tensor_ratio']

	#### Prepare arguments
	if deltal is None: deltal = 5
	if nbins is None: nbins = 5000/deltal
	if min_ell is None: min_ell = np.int(180/np.degrees((np.sqrt(2*fsky))))
	lmin = 1+(deltal)*np.arange(nbins)
	lmax = lmin+deltal-1
	if max_ell is None:
		maxellbin = 5000#1./np.radians(fwhmdeg / 2.35)
	else:
		maxellbin = max_ell
	mask = (lmax < np.min(np.array([maxellbin,2500]))) & (lmin >= min_ell)
	if len(lmax[mask]) is 0:
		return np.zeros(len(pars)),np.zeros(len(pars)),0,0,0,0
	lcenter = (lmax+lmin)/2
	allargs = [params, np.max(lmax), fsky, mukarcmin, fwhmdeg, deltal, lmin, lmax, consistency, parkeys]
	#### Get errors
	btot_binned, clprim_binned, cllens_binned, dclsb, dclnb, dclb, Bprim, Blensed, totBB = get_spectrum_bins(pars, lcenter, 
		allargs, spectra = spectra)
	spectra = [Bprim, Blensed, totBB]
	dclb[~mask] = 1e30
	dclsb[~mask] = 1e30
	dclnb[~mask] = 1e30
	if noiseonly: dclb=dclnb
	bins = lcenter
	if plotcl:
		clf()
		theell = np.arange(len(totBB)) + 1
		plot(theell, totBB,'k')
		plot(theell,Blensed,'k:')
		plot(theell,Bprim,'k--')
		errorbar(lcenter[mask], btot_binned[mask], yerr = dclb[mask], fmt='ro')
		xlim(0,np.max(lmax[mask]))
		#yscale('log')
	#### Get Fisher matrices
	fmsvl, thederr = fishermatrix(pars, bins, dclsb, get_spectrum_bins, allargs, der=der)
	if der is None:
		der = thederr.copy()
	fm, thederr = fishermatrix(pars, bins, dclb, get_spectrum_bins, allargs, der=der)
	##### Priors
	if prior is not None:
		fmsvl += prior
		fm += prior
	#### Get sigmas
	if fixed is None:
		nn = len(pars)
		sigmas  = np.sqrt(np.diag(np.linalg.inv(fm + np.random.rand(nn,nn)*1e-10)))
		sigmas_svl = np.sqrt(np.diag(np.linalg.inv(fmsvl + np.random.rand(nn,nn)*1e-10)))
	else:
		all = np.arange(len(pars)).tolist()
		if type(fixed) is list:
			for ii in fixed: all.remove(ii)
		else:
			all.remove(fixed)
		nn = len(all)
		sigmas  = np.sqrt(np.diag(np.linalg.inv(submatrix(fm,all) + np.random.rand(nn,nn)*1e-10)))
		sigmas_svl = np.sqrt(np.diag(np.linalg.inv(submatrix(fmsvl,all) + np.random.rand(nn,nn)*1e-10)))
	if plotmat:
		clf()
		a1,l1=plot_fisher(fmsvl, pars, parnames,'b', fixed=fixed)
		a0,l0=plot_fisher(fm, pars, parnames,'r', fixed=fixed)
		subplot(nn,nn,nn)
		axis('off')
		legend([a0,a1],['Noisy: '+l0,'SVL: '+l1],title=title,fontsize=10)
	#### return sigmas
	return sigmas, sigmas_svl, der, spectra, fm, fmsvl

