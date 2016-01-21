from __future__ import division
from pylab import *
import healpy as hp
from matplotlib.pyplot import *
import numpy as np
import os
import sys
import qubic
import pycamb
from pyoperators import DenseBlockDiagonalOperator, Rotation3dOperator
from pysimulators import FitsArray
from pyoperators import MPI

from qubic import (
    QubicAcquisition, create_sweeping_pointings, equ2gal, map2tod, tod2map_all,
    tod2map_each, create_random_pointings, QubicInstrument)



def profile(x,y,range=None,nbins=10,fmt=None,plot=True, dispersion=True):
  if range == None:
    mini = np.min(x)
    maxi = np.max(x)
  else:
    mini = range[0]
    maxi = range[1]
  dx = (maxi - mini) / nbins
  xmin = np.linspace(mini,maxi-dx,nbins)
  xmax = xmin + dx
  xc = xmin + dx / 2
  yval = np.zeros(nbins)
  dy = np.zeros(nbins)
  dx = np.zeros(nbins) + dx / 2
  for i in np.arange(nbins):
    ok = (x > xmin[i]) & (x < xmax[i])
    yval[i] = np.mean(y[ok])
    if dispersion: 
      fact = 1
    else:
      fact = np.sqrt(len(y[ok]))
    dy[i] = np.std(y[ok])/fact
  if plot: errorbar(xc, yval, xerr=dx, yerr=dy, fmt=fmt)
  return xc, yval, dx, dy




rep = '/Users/hamilton/Qubic/MapMaking/SubFP/ns256/'
models = ['ref', 'SFP', 'even', 'evenodd','rnd']
#models = ['ref', 'SFP']
noises = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 10., 100., 1000.]

racenter = 0.0      # deg
deccenter = -57.0   # deg
center = equ2gal(racenter, deccenter)

mratiosigs = np.zeros((len(noises), len(models), 3))
sratiosigs = np.zeros((len(noises), len(models), 3))
mratiosigs_noiseless = np.zeros((len(noises), len(models), 3))
sratiosigs_noiseless = np.zeros((len(noises), len(models), 3))
for inoise in np.arange(len(noises)):
	print(inoise, len(noises))
	strnoise = np.str(noises[inoise])
	res = []
	res_noiseless = []
	allcov = []
	for i in np.arange(len(models)):
		maps = FitsArray(rep+'maps'+models[i]+'_'+strnoise+'.fits')
		maps_noiseless = FitsArray(rep+'maps'+models[i]+'_'+strnoise+'_noiseless.fits')
		cov = maps[3]
		if i > 0: cov *= 2
		res_I = maps[0] - maps[4]
		res_Q = maps[1] - maps[5]
		res_U = maps[2] - maps[6]
		res_I_noiseless = maps[0] - maps_noiseless[0]
		res_Q_noiseless = maps[1] - maps_noiseless[1]
		res_U_noiseless = maps[2] - maps_noiseless[2]
		#hp.gnomview(cov, rot=center, reso=30, title='Cov',norm='hist')
		#hp.gnomview(maps[1], rot=center, reso=30, title='Q Rec')
		#hp.gnomview(maps[5], rot=center, reso=30, title='Q In')
		#hp.gnomview(maps_noiseless[1], rot=center, reso=30, title='Q Rec Noiseless')
		res.append([res_I, res_Q, res_U])
		res_noiseless.append([res_I_noiseless, res_Q_noiseless, res_U_noiseless])
		allcov.append(cov)

	sigvals = np.zeros((10, len(models), 3))
	sigvals_noiseless = np.zeros((10, len(models), 3))
	for iqu in np.arange(3):
		for i in np.arange(len(models)):
			xx,yy,dx,sigvals[:,i,iqu] = profile(allcov[i],res[i][iqu],range=[0, np.max(allcov[0])], nbins=10, plot=False, dispersion=False)
			xx,yy,dx,sigvals_noiseless[:,i,iqu] = profile(allcov[i],res_noiseless[i][iqu],range=[0, np.max(allcov[0])], nbins=10, plot=False, dispersion=False)

	ratiosigs = np.zeros((10, len(models), 3))
	ratiosigs_noiseless = np.zeros((10, len(models), 3))
	for iqu in np.arange(3):
		for i in np.arange(len(models)):
			ratiosigs[:,i,iqu] = sigvals[:,i,iqu]/sigvals[:,0,iqu]
			ratiosigs_noiseless[:,i,iqu] = sigvals_noiseless[:,i,iqu]/sigvals_noiseless[:,0,iqu]

	mratiosigs[inoise, :, :] = np.mean(ratiosigs, axis = 0)
	sratiosigs[inoise, :, :] = np.std(ratiosigs, axis = 0)
	mratiosigs_noiseless[inoise, :, :] = np.mean(ratiosigs_noiseless, axis = 0)
	sratiosigs_noiseless[inoise, :, :] = np.std(ratiosigs_noiseless, axis = 0)

	# clf()
	# ylim(0.9,1.5)
	# xlabel('Coverage Full Instrument')
	# ylabel('RMS ratio w.r.t. Full Inst.')
	# title('U - noise = '+str(noises[inoise])+' - no Intensity')
	# plot(xx, xx*0+sqrt(2), 'k:')
	# plot(xx, xx*0+1, 'k--')
	# for i in np.arange(len(models)):
	# 	plot(xx,ratiosigs[:,i,1], label=models[i])
	# legend(loc='lower right')
	# savefig('ratio_noise_'+str(noises[inoise])+'_noI.png')






colmod = ['k', 'r', 'g', 'b', 'm']
striqu = ['I','Q','U']
clf()
for iqu in np.arange(2)+1:
	subplot(2,1,iqu)
	xscale('log')
	yscale('log')
	ylim(0.9,1.5)
	xlabel('Noise level')
	ylabel('RMS ratio w.r.t. Full Inst.')
	title(striqu[iqu])
	plot(noises, np.array(noises)*0+sqrt(2), 'k:')
	plot(noises, np.array(noises)*0+1, 'k--')
	for i in np.arange(len(models)-1)+1:
		errorbar(noises, mratiosigs[:,i,iqu], yerr=sratiosigs[:,i, iqu], fmt='-o', label=models[i], color=colmod[i])
		#errorbar(noises, mratiosigs_noiseless[:,i,iqu], yerr=sratiosigs_noiseless[:,i, iqu], fmt='-x', color=colmod[i])

legend(loc='lower right', frameon=False)
savefig('rms_ratio_models_noI.png')





####################################################################################
models = ['ref', 'SFP', 'even', 'evenodd','rnd']
i=1
inoise=7
strnoise = np.str(noises[inoise])
rep = '/Users/hamilton/Qubic/MapMaking/SubFP/ns256/'
maps = FitsArray(rep+'maps'+models[i]+'_'+strnoise+'.fits')
maps_noiseless = FitsArray(rep+'maps'+models[i]+'_'+strnoise+'_noiseless.fits')
cov = maps[3]
res_I = maps[0] - maps[4]
res_Q = maps[1] - maps[5]
res_U = maps[2] - maps[6]
res_I_noiseless = maps[0] - maps_noiseless[0]
res_Q_noiseless = maps[1] - maps_noiseless[1]
res_U_noiseless = maps[2] - maps_noiseless[2]

clf()
hp.gnomview(cov, rot=center, reso=30, title='Cov',norm='hist',sub=(2,3,1))
hp.gnomview(maps[1], rot=center, reso=30, title='Q Rec',sub=(2,3,2),min=-2, max=2)
hp.gnomview(maps[5], rot=center, reso=30, title='Q In',sub=(2,3,4),min=-2, max=2)
hp.gnomview(maps_noiseless[1], rot=center, reso=30, title='Q Rec Noiseless',sub=(2,3,5),min=-2, max=2)
hp.gnomview(res_Q, rot=center, reso=30, title='Out-In',sub=(2,3,3),min=-2, max=2)
hp.gnomview(res_Q_noiseless, rot=center, reso=30, title='Out-NoiselessOut',sub=(2,3,6),min=-2, max=2)

clf()
hp.gnomview(cov, rot=center, reso=30, title='Cov',sub=(2,2,1))
hp.gnomview(maps[1], rot=center, reso=30, title='Q Rec',sub=(2,2,2), min=-3, max=3)
hp.gnomview(res_Q, rot=center, reso=30, title='Out-In',sub=(2,2,3),min=-0.1, max=0.1)
hp.gnomview(res_Q_noiseless, rot=center, reso=30, title='Out-NoiselessOut',sub=(2,2,4))

rep = '/Users/hamilton/Qubic/MapMaking/SubFP/ns256_noI/'
maps = FitsArray(rep+'maps'+models[i]+'_'+strnoise+'.fits')
maps_noiseless = FitsArray(rep+'maps'+models[i]+'_'+strnoise+'_noiseless.fits')
cov = maps[3]
res_I_noI = maps[0] - maps[4]
res_Q_noI = maps[1] - maps[5]
res_U_noI = maps[2] - maps[6]
res_I_noiseless_noI = maps[0] - maps_noiseless[0]
res_Q_noiseless_noI = maps[1] - maps_noiseless[1]
res_U_noiseless_noI = maps[2] - maps_noiseless[2]


clf()
hp.gnomview(res_Q, rot=center, reso=30, title='Out-In',sub=(2,2,1))
hp.gnomview(res_Q_noiseless, rot=center, reso=30, title='Out-NoiselessOut',sub=(2,2,2),min=-0.01,max=0.01)
hp.gnomview(res_Q_noI, rot=center, reso=30, title='Out-In No I',sub=(2,2,3))
hp.gnomview(res_Q_noiseless_noI, rot=center, reso=30, title='Out-NoiselessOut NoI',sub=(2,2,4),min=-.01,max=0.01)


#############################
models = ['ref', 'SFP', 'even', 'evenodd','rnd']
inoise=0
strnoise = np.str(noises[inoise])

allres_Q = []
allres_Q_new = []
allres_Q_noI = []
allres_Q_noise0 = []
allres_Q_noise0_new = []
allres_Q_noise0_noI = []
inmapI = []

for i in np.arange(len(models)):
	rep = '/Users/hamilton/Qubic/MapMaking/SubFP/ns256/'
	maps = FitsArray(rep+'maps'+models[i]+'_'+strnoise+'.fits')
	maps_noiseless = FitsArray(rep+'maps'+models[i]+'_'+strnoise+'_noiseless.fits')
	cov = maps[3]
	res_I = maps[0] - maps[4]
	res_Q = maps[1] - maps[5]
	res_U = maps[2] - maps[6]
	res_I_noise0 = maps_noiseless[0] - maps[4]
	res_Q_noise0 = maps_noiseless[1] - maps[5]
	res_U_noise0 = maps_noiseless[2] - maps[6]
	#hp.gnomview(maps[1], rot=center, reso=30)
	#hp.gnomview(maps[5], rot=center, reso=30)
	allres_Q.append(res_Q)
	allres_Q_noise0.append(res_Q_noise0)
	rep = '/Users/hamilton/Qubic/MapMaking/SubFP/ns256_newprecond/'
	maps = FitsArray(rep+'maps'+models[i]+'_'+strnoise+'.fits')
	maps_noiseless = FitsArray(rep+'maps'+models[i]+'_'+strnoise+'_noiseless.fits')
	cov = maps[3]
	res_I = maps[0] - maps[4]
	res_Q = maps[1]/400 - maps[5]
	res_U = maps[2]/400 - maps[6]
	res_I_noise0 = maps_noiseless[0]/400 - maps[4]
	res_Q_noise0 = maps_noiseless[1]/400 - maps[5]
	res_U_noise0 = maps_noiseless[2] - maps[6]
	hp.gnomview(maps[1]/400, rot=center, reso=30, title='Reconstructed')
	hp.gnomview(maps[5], rot=center, reso=30, title='Input')
	hist(maps[1]/maps[5],range=[0,1000],bins=100)
	allres_Q_new.append(res_Q)
	allres_Q_noise0_new.append(res_Q_noise0)
	rep = '/Users/hamilton/Qubic/MapMaking/SubFP/ns256_noI/'
	maps = FitsArray(rep+'maps'+models[i]+'_'+strnoise+'.fits')
	maps_noiseless = FitsArray(rep+'maps'+models[i]+'_'+strnoise+'_noiseless.fits')
	cov = maps[3]
	res_I = maps[0] - maps[4]
	res_Q = maps[1] - maps[5]
	res_U = maps[2] - maps[6]
	res_I_noise0 = maps_noiseless[0] - maps[4]
	res_Q_noise0 = maps_noiseless[1] - maps[5]
	res_U_noise0 = maps_noiseless[2] - maps[6]
	allres_Q_noI.append(res_Q)
	allres_Q_noise0_noI.append(res_Q_noise0)


clf()
mm=0.1
for i in np.arange(len(models)):
	hp.gnomview(allres_Q_noise0[i], rot=center, reso=30, title='Q Res with I: '+models[i],sub=(3,len(models),1+i), min=-mm,max=mm)
	hp.gnomview(allres_Q_noise0_new[i], rot=center, reso=30, title='Q Res with I (new precond): '+models[i],sub=(3,len(models),len(models)+1+i), min=-mm,max=mm)
	hp.gnomview(allres_Q_noise0_noI[i], rot=center, reso=30, title='Q Res no I: '+models[i],sub=(3,len(models),2*len(models)+1+i), min=-mm,max=mm)

#savefig('residuals.png')







