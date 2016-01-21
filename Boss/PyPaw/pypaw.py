from __future__ import division
from pylab import *
from matplotlib.pyplot import *
import numpy as np

def readfile(file):
	data = np.loadtxt(file)
	return(data.T)


def readfile2dict(file, keys=None):
	data = np.loadtxt(file)
	nkeys = data.shape[0]
	if keys is None:
		keys = []
		for i in np.arange(nkeys): keys.append('x'+str(i))
	dictdata = dict(zip(keys,data.T)) 
	return dictdata

def scatter(x,y,cut=None,color=None,xlab=None,ylab=None,clearscreen=True,marker=',', alpha=1, text=None):
	if cut is None:
		newx = x
		newy = y
	else:
		newx = x[cut]
		newy = y[cut]
	if clearscreen: clf()
	if xlab: xlabel(xlab)
	if ylab: ylabel(ylab)
	lab = 'Entries : '+np.str(len(newx))
	if text is not None: lab = text+'\n'+lab
	if color is None:
		plot(newx,newy,marker, label = lab,alpha=alpha)
	else:
		plot(newx,newy,marker, label = lab,alpha=alpha, color=color)
	legend(loc='upper right')


def histo(x, cut=None, range=None, bins=10, alpha=1, clearscreen=True, color='blue', xlab=None, text=None):
	if cut is None:
		newx = x
	else:
		newx = x[cut]
	if clearscreen: clf()
	if xlab: xlabel(xlab)
	lab = 'Entries : '+str(len(newx))+'\n Mean = {0:.3g} +/- {1:.3g}'.format(np.mean(newx), np.std(newx))
	if text is not None: lab = text+'\n'+lab
	truc=hist(newx, range=range, bins=bins, alpha=alpha, color=color, label=lab)
	legend(loc='upper right')
	return np.array([len(newx), np.mean(newx), np.std(newx)])



	



