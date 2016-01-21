from __future__ import division
from pyoperators import pcg
from pysimulators import profile
from qubic import (
    create_random_pointings, equ2gal, QubicAcquisition, PlanckAcquisition,
    QubicPlanckAcquisition, QubicInstrument)
from qubic.data import PATH
from qubic.io import read_map
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np



def statstr(vec):
    m=np.mean(vec)
    s=np.std(vec)
    return '{0:.4f} +/- {1:.4f}'.format(m,s)

def plotinst(inst,shift=0.12):
  for xyc, quad in zip(inst.detector.center, inst.detector.quadrant): 
    if quad < 4:
      plot(xyc[0],xyc[1],'ro')
    else:
      plot(xyc[0]+shift,xyc[1],'bo')
    xlim(-0.06, 0.18)


def display(input, msg, iplot=1, reso=5, Trange=[100, 5, 5]):
    out = []
    for i, (kind, lim) in enumerate(zip('IQU', Trange)):
        map = input[..., i]
        out += [hp.gnomview(map, rot=center, reso=reso, xsize=800, min=-lim,
                            max=lim, title=msg + ' ' + kind,
                            sub=(3, 3, iplot + i), return_projected_map=True)]
    return out


def profile(x,y,range=None,nbins=10,fmt=None,plot=True, dispersion=True, color=None):
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
  if plot: errorbar(xc, yval, xerr=dx, yerr=dy, fmt=fmt, color=color)
  return xc, yval, dx, dy


nside = 256
racenter = 0.0      # deg
deccenter = -57.0   # deg
center = equ2gal(racenter, deccenter)

sky = read_map(PATH + 'syn256_pol.fits')
sampling = create_random_pointings([racenter, deccenter], 1000, 10)
detector_nep = 4.7e-17/np.sqrt(365*86400 / (len(sampling)*sampling.period))

### QUBIC Full
instFull = QubicInstrument(detector_nep=detector_nep, filter_nu=150*1e9)
clf()
subplot(1,1,1, aspect='equal')
plotinst(instFull)
title('Full Instrument')
savefig('FullInst.png')

### QUBIC Single Focal Plane
instSingleFP = instFull[0:992]
clf()
subplot(1,1,1, aspect='equal')
plotinst(instSingleFP)
title('Single Focal Plane')
savefig('SFP.png')

#Same with One bolo out of two
numbols = np.arange(len(instFull.detector.index))
mask2 = (numbols % 2) == 0
instEven = instFull[mask2]
clf()
subplot(1,1,1, aspect='equal')
plotinst(instEven)
title('Even Bolometers')
savefig('Even.png')

#Same with half of the bolos randomly chosen
rndnum = np.random.rand(len(instFull.detector.index))
ind = np.argsort(rndnum)
instRnd = instFull[ind[0:len(instFull.detector.index)/2]]
clf()
subplot(1,1,1, aspect='equal')
plotinst(instRnd)
title('Random Bolometers')
savefig('Random.png')

#Now even bolos on X fp and Odd bolos on Y fp
numbols = np.arange(len(instFull.detector.index))
mask2 = (numbols % 2) == 0
mask2[len(mask2)/2:] = ~mask2[len(mask2)/2:]
instEvenOdd = instFull[mask2]
clf()
subplot(1,1,1, aspect='equal')
plotinst(instEvenOdd)
title('Even/Odd Bolometers')
savefig('EvenOdd.png')

all_instruments = [instFull, instSingleFP, instEven, instRnd, instEvenOdd]
names = ['Full Instrument', 'Single Focal Plane', 'Even Bolometers', 'Random Bolometers', 'Even and Odd Bolometers']
all_solutions_fusion = []
all_coverages = []


for i in xrange(len(all_instruments)):
	acq_qubic = QubicAcquisition(all_instruments[i], sampling, nside=nside,
                             detector_nep=detector_nep)
	all_coverages.append(acq_qubic.get_coverage())
	convolved_sky = acq_qubic.instrument.get_convolution_peak_operator()(sky)
	acq_planck = PlanckAcquisition(150, acq_qubic.scene, true_sky=convolved_sky)
	acq_fusion = QubicPlanckAcquisition(acq_qubic, acq_planck)

	H = acq_fusion.get_operator()
	invntt = acq_fusion.get_invntt_operator()
	obs = acq_fusion.get_observation()

	A = H.T * invntt * H
	b = H.T * invntt * obs

	solution_fusion = pcg(A, b, disp=True, maxiter=10000, tol=5e-6)
	all_solutions_fusion.append(solution_fusion)

mask = all_coverages[0] > np.max(all_coverages[0]/10)

reso=3
Trange=[1, 1, 1]
for i in xrange(len(all_instruments)):
	figure(i, figsize=(10,9))
	clf()
	subplot(2,1,1, aspect='equal')
	plotinst(all_instruments[i])
	title(names[i])
	resid = all_solutions_fusion[i]['x'] - convolved_sky
	resid[~mask,:] = 0
	display(resid, 'Difference map', iplot=7, reso=reso, Trange=Trange)
	#savefig(names[i]+'.png')


cols=['black', 'red','blue','green', 'orange']
aa=0.2
rng = [-2,4]
fs=8
nb=20
clf()
for i in xrange(len(all_instruments)):
	resid = all_solutions_fusion[i]['x'] - convolved_sky
	idata = profile(all_coverages[i][mask]/np.max(all_coverages[i]), np.nan_to_num(resid[mask,0]), nbins=nb, range=[0,1],color=cols[i], plot=False)
	qdata = profile(all_coverages[i][mask]/np.max(all_coverages[i]), np.nan_to_num(resid[mask,1]), nbins=nb, range=[0,1],color=cols[i], plot=False)
	udata = profile(all_coverages[i][mask]/np.max(all_coverages[i]), np.nan_to_num(resid[mask,2]), nbins=nb, range=[0,1],color=cols[i], plot=False)

	subplot(3,1,1)
	yscale('log')
	xlabel('Normalized coverage')
	ylabel('I RMS residuals')
	ylim(0.1,2)
	plot(idata[0], idata[3], color=cols[i], label=names[i], lw=2)
	if i==0: plot(idata[0], idata[3]*sqrt(2), '--', color=cols[i], label=names[i]+' x sqrt(2)', lw=2)
	legend(fontsize=fs, loc='upper right')

	subplot(3,1,2)
	yscale('log')
	xlabel('Normalized coverage')
	ylabel('Q RMS residuals')
	ylim(0.1,2)
	plot(qdata[0], qdata[3], color=cols[i], label=names[i], lw=2)
	if i==0: plot(qdata[0], qdata[3]*sqrt(2), '--', color=cols[i], label=names[i]+' x sqrt(2)', lw=2)
	legend(fontsize=fs, loc='upper right')

	subplot(3,1,3)
	yscale('log')
	xlabel('Normalized coverage')
	ylabel('U RMS residuals')
	ylim(0.1,2)
	plot(udata[0], udata[3], color=cols[i], label=names[i], lw=2)
	if i==0: plot(udata[0], udata[3]*sqrt(2), '--', color=cols[i], label=names[i]+' x sqrt(2)', lw=2)
	legend(fontsize=fs, loc='upper right')

#savefig('rms.png')




cols=['black', 'red','blue','green', 'orange']
aa=0.2
rng = [-2,4]
fs=8
nb=20
clf()
for i in xrange(len(all_instruments)):
	resid = all_solutions_fusion[i]['x'] - convolved_sky
	idata = profile(all_coverages[i][mask]/np.max(all_coverages[i]), np.nan_to_num(resid[mask,0]), nbins=nb, range=[0,1],color=cols[i], plot=False)
	qdata = profile(all_coverages[i][mask]/np.max(all_coverages[i]), np.nan_to_num(resid[mask,1]), nbins=nb, range=[0,1],color=cols[i], plot=False)
	udata = profile(all_coverages[i][mask]/np.max(all_coverages[i]), np.nan_to_num(resid[mask,2]), nbins=nb, range=[0,1],color=cols[i], plot=False)
	if i == 0 :
		theidata = idata
		theqdata = qdata
		theudata = udata

	subplot(3,1,1)
	xlabel('Normalized coverage')
	ylabel('I RMS residuals ratio \n w.r.t. Full Instrument')
	ylim(0.,3)
	plot(linspace(0,1,10),np.zeros(10)+sqrt(2), 'k--')
	plot(idata[0], idata[3]/theidata[3], color=cols[i], label=names[i], lw=2)
	legend(fontsize=fs, loc='upper right')

	subplot(3,1,2)
	xlabel('Normalized coverage')
	ylabel('Q RMS residuals ratio \n w.r.t. Full Instrument')
	ylim(0.,3)
	plot(qdata[0], qdata[3]/theqdata[3], color=cols[i], label=names[i], lw=2)
	plot(linspace(0,1,10),np.zeros(10)+sqrt(2), 'k--')
	legend(fontsize=fs, loc='upper right')

	subplot(3,1,3)
	xlabel('Normalized coverage')
	ylabel('U RMS residuals ratio \n w.r.t. Full Instrument')
	ylim(0.,3)
	plot(udata[0], udata[3]/theudata[3], color=cols[i], label=names[i], lw=2)
	plot(linspace(0,1,10),np.zeros(10)+sqrt(2), 'k--')
	legend(fontsize=fs, loc='upper right')

#savefig('rms_ratio.png')








