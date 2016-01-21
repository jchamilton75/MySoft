from __future__ import division

import healpy as healpy
from matplotlib.pyplot import *
import numpy as np
import pycamb
#from qubic.utils import progress_bar
from Homogeneity import fitting

#### Just need a function that returns the data and errors in bins from a dictionary with the parameters (pars) and possible extra arguments args

def give_derivatives(pars, args, thefunction, delt=0.01):
	### First call to the function with default arguments
	data, dum = thefunction(pars, args)
	nbins = len(data)
	nder = len(pars)
	der = np.zeros((nder, nbins))
	newvals = np.array(pars.values()) * (1 + delt)
	pk = pars.keys()
	for i in range(nder):
		newpars = pars.copy()
		newpars[pk[i]] = newvals[i]
		newdata, newerror = thefunction(newpars, args)
		der[i,:] = (newdata - data) / (newpars[pk[i]] - pars[pk[i]])
	return der

def fishermatrix(pars, args, der, errors, thefunction):
	fm = np.zeros((len(pars), len(pars)))
	for k in range(len(errors)):
		for i in range(len(pars)):
			for j in range(len(pars)):
				fm[i,j] += der[i,k]*der[j,k]/errors[k]**2
	return fm


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

def give_sigmas(fisherin, pars, fixed=None):
	fisher = fisherin.copy()
	ninit = fisher.shape[0]
	labels = pars.keys()
	labelsinit = np.array(labels)
	if fixed is None:
		nn = ninit
		covar = np.linalg.inv(fisher + np.random.rand(ninit,ninit)*1e-10)
		ss  = np.sqrt(np.diag(covar))
	else:
		allvars = pars.keys()
		fixedindex = []
		allindex = np.arange(len(pars)).tolist()
		for i in np.arange(len(pars)):
			if pars.keys()[i] in fixed: 
				allindex.remove(i)
				fixedindex.append(i)
				allvars.remove(pars.keys()[i])
				labels.remove(labelsinit[i])
		nn = len(allvars)
		fisher = submatrix(fisher,allindex)
		covar = np.linalg.inv(fisher + np.random.rand(len(labels),len(labels))*1e-10)
		ss  = np.sqrt(np.diag(covar))
	return ss, covar


def plot_fisher(fisherin, pars, col, limits=None,nbins=None,fixed=None,size=256, onesigma=False, varnames=False):
	fisher = fisherin.copy()
	mm = np.array(pars.values())
	ninit = fisher.shape[0]
	if varnames is not False:
		labels = varnames.values()
	else:
		labels = pars.keys()
	labelsinit = np.array(labels)
	#### Cut the fisher matrix if some parameters are fixed
	if fixed is None:
		nn = ninit
		#covar = np.linalg.inv(fisher + np.random.rand(ninit,ninit)*1e-10)
		covar = np.linalg.inv(fisher)
		ss  = np.sqrt(np.diag(covar))
	else:
		allvars = pars.keys()
		fixedindex = []
		allindex = np.arange(len(pars)).tolist()
		for i in np.arange(len(pars)):
			if pars.keys()[i] in fixed: 
				allindex.remove(i)
				fixedindex.append(i)
				allvars.remove(pars.keys()[i])
				labels.remove(labelsinit[i])
		nn = len(allvars)
		fisher = submatrix(fisher,allindex)
		mm = mm[allindex]
		#covar = np.linalg.inv(fisher + np.random.rand(len(labels),len(labels))*1e-10)
		covar = np.linalg.inv(fisher)
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






