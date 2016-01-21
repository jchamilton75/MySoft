from __future__ import division
from pyoperators import pcg
from pysimulators import profile
from qubic import (
    create_random_pointings, equ2gal, QubicAcquisition, PlanckAcquisition,
    QubicPlanckAcquisition)
from qubic.data import PATH
from qubic.io import read_map
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from pyoperators import MPI


############ Functions
def plotinst(inst,shift=0.12):
  clf()
  for xyc, quad in zip(inst.detector.center, inst.detector.quadrant): 
    if quad < 4:
      plot(xyc[0],xyc[1],'ro')
    else:
      plot(xyc[0]+shift,xyc[1],'bo')


### map2tod copied from former QUBIC library
def map2tod(acq, map, convolution=False, max_nbytes=None):
  if convolution:
    convolution = acq.get_convolution_peak_operator()
    map = convolution(map)
  H = acq.get_operator()
  tod = H(map)
  if convolution:
    return tod, map
  return tod

### tod2map








##################### General stuff ###############################################
rank = MPI.COMM_WORLD.rank

nside = 256
racenter = 0.0      # deg
deccenter = -57.0   # deg

sky = read_map(PATH + 'syn256_pol.fits')
sampling = create_random_pointings([racenter, deccenter], 1000, 10)
detector_nep = 4.7e-17/np.sqrt(365*86400 / len(sampling)*sampling.period)
####################################################################################


##################### Various Instruments ##########################################





def get_maps(spectra, inst, sampling, nside, x0, coverage_threshold=0.01,savefile=None, savefile_noiseless=None, noI=False):
  #if x0 is None:
  #  print("Running Synfast")
  #  x0 = np.array(hp.synfast(spectra[1:],nside,fwhm=0,pixwin=True,new=True)).T
  #  if noI: x0[:,0]*=0
  acquisition = QubicAcquisition(inst, sampling,
                                nside=nside,
                                synthbeam_fraction=0.99)
                                #max_nbytes=6e9)
  # simulate the timeline
  print('Now doing MAP2TOD (simulate data)')
  tod, x0_convolved = map2tod(acquisition, x0, convolution=True)
  print('TOD Done, now adding noise')
  bla = acquisition.get_noise()
  tod_noisy = tod + bla 
  # reconstruct using all available bolometers
  print('Now doing TOD2MAP (make map)')
  map_all, cov_all = tod2map_all(acquisition, tod_noisy, tol=1e-4, coverage_threshold=coverage_threshold)
  print('Map done')
  print('Now doing TOD2MAP (make map) on noiseless data')
  map_all_noiseless, cov_all_noiseless = tod2map_all(acquisition, tod, tol=1e-4, coverage_threshold=coverage_threshold)
  print('Noiseless Map done')
  mask = map_all_noiseless != 0
  x0_convolved[~mask,:] = 0
  rank = MPI.COMM_WORLD.rank
  if rank == 0:
    if savefile is not None:
      print('I am rank='+str(rank)+' and I try to save the file '+savefile)
      FitsArray(np.array([map_all[:,0], map_all[:,1], map_all[:,2], cov_all, x0_convolved[:,0], x0_convolved[:,1], x0_convolved[:,2]]), copy=False).save(savefile)
      print('I am rank='+str(rank)+' and I just saved the file '+savefile)
      print('I am rank='+str(rank)+' and I try to save the file '+savefile_noiseless)
      FitsArray(np.array([map_all_noiseless[:,0], map_all_noiseless[:,1], map_all_noiseless[:,2], cov_all_noiseless, x0_convolved[:,0], x0_convolved[:,1], x0_convolved[:,2]]), copy=False).save(savefile_noiseless)
      print('I am rank='+str(rank)+' and I just saved the file '+savefile_noiseless)
  return map_all, x0_convolved, cov_all, x0

# some display
def display(map, cov, msg, sub):
    for i, (kind, lim) in enumerate(zip('IQU', [200, 10, 10])):
        map_ = map[..., i].copy()
        mask = cov == 0
        map_[mask] = np.nan
        hp.gnomview(map_, rot=center, reso=5, xsize=400, min=-lim, max=lim,
                    title=msg + ' ' + kind, sub=(3, 3, 3 * (sub-1) + i+1))




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


def get_ratio_rms_cov(instFull, inst2, spectra, sampling, nside, pfFull=0, x0=None, plot=True, sub1=None, sub2=None, center=[316.45, -58.76], lab='Alt. Inst.', coverage_threshold=0.01, refsavefile=None, altsavefile=None, refsavefile_noiseless=None, altsavefile_noiseless=None, noI=False):
  nb=10
  iqu=1
  lim=[100,1,1][iqu]
  name = ['I','Q','U'][iqu]
  doplot=False
  ## reference instrument
  if pfFull == 0:
    print("Doing Reference Instrument")
    mapsFull, initconvFull, covFull, x0 = get_maps(spectra, instFull, sampling, nside, x0, coverage_threshold=coverage_threshold, savefile=refsavefile, savefile_noiseless=refsavefile_noiseless, noI=noI)
    maskFull = covFull > coverage_threshold
    residFull = initconvFull - mapsFull
    pfFull_I = profile(covFull[maskFull], residFull[maskFull,0],fmt='ro',nbins=nb, range=[0,np.max(covFull[maskFull])], plot=doplot)
    pfFull_Q = profile(covFull[maskFull], residFull[maskFull,1],fmt='bo',nbins=nb, range=[0,np.max(covFull[maskFull])], plot=doplot)
    pfFull_U = profile(covFull[maskFull], residFull[maskFull,2],fmt='go',nbins=nb, range=[0,np.max(covFull[maskFull])], plot=doplot)
    pfFull = [pfFull_I, pfFull_Q, pfFull_U, maskFull, residFull]
    if plot:
      figure(0)
      clf()
      hp.gnomview(initconvFull[:,iqu],min=-lim,max=lim,title=name+' In Ref. Inst.', sub=sub1, rot=center, reso=30)
      figure(1)
      clf()
      hp.gnomview(residFull[:,iqu],min=-lim/10,max=lim/10,title=name+' Res Ref. Inst.', sub=sub1, rot=center, reso=30)
      figure(2)
      clf()
      hp.gnomview(mapsFull[:,iqu],min=-lim,max=lim,title=name+' Out Ref. Inst.', sub=sub1, rot=center, reso=30)
      figure(3)
      clf()
      hp.gnomview(residFull[:,iqu]-residFull[:,iqu],title=name+' Res Fin Ref. Inst.', sub=sub1, rot=center, reso=30)
  else:
    pfFull_I = pfFull[0]
    pfFull_Q = pfFull[1]
    pfFull_U = pfFull[2]
    maskFull = pfFull[3]
    residFull = pfFull[4]

  ## alternative one
  print("Doing Alternative instrument")
  maps2, initconv2, cov2, x0 = get_maps(spectra, inst2, sampling, nside, x0, coverage_threshold=coverage_threshold, savefile=altsavefile, savefile_noiseless=altsavefile_noiseless, noI=noI)
  mask2 = cov2 > coverage_threshold
  resid2 = initconv2 - maps2
  if plot:
    figure(0)
    hp.gnomview(initconv2[:,iqu],min=-lim,max=lim,fig=0, title=name+' In '+lab,sub=sub2, rot=center, reso=30)
    figure(1)
    hp.gnomview(resid2[:,iqu],min=-lim/10,max=lim/10,fig=0, title=name+' Res '+lab,sub=sub2, rot=center, reso=30)
    figure(2)
    hp.gnomview(maps2[:,iqu],min=-lim,max=lim,fig=1, title=name+' Out '+lab,sub=sub2, rot=center, reso=30)
    figure(3)
    hp.gnomview(resid2[:,iqu]-residFull[:,iqu],fig=1, title=name+' Res Fin '+lab,sub=sub2, rot=center, reso=30)
  pf2_I = profile(cov2[mask2]*2, resid2[mask2,0],fmt='ro',   nbins=nb, range=[0,np.max(cov2[mask2]*2)], plot=doplot)
  pf2_Q = profile(cov2[mask2]*2, resid2[mask2,1],fmt='bo',nbins=nb, range=[0,np.max(cov2[mask2]*2)], plot=doplot)
  pf2_U = profile(cov2[mask2]*2, resid2[mask2,2],fmt='go',nbins=nb, range=[0,np.max(cov2[mask2]*2)], plot=doplot)
  return pfFull_I[0], pf2_I[3]/pfFull_I[3], pf2_Q[3]/pfFull_Q[3], pf2_U[3]/pfFull_U[3], pfFull, x0


def meanweight(y,dy):
  weights = 1./dy**2
  mask =  isfinite(weights)
  mm = np.sum(y[mask]*weights[mask])/np.sum(weights[mask])
  ss = np.sqrt(1./np.sum(weights[mask]))
  return mm,ss




#### Get input Power spectra
################# Input Power spectrum ###################################
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
print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',omegav/h2

params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':0.1,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}

lmaxcamb = 3*512
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
lll = np.arange(1,lmaxcamb+1)
fact = (lll*(lll+1))/(2*np.pi)
spectra = [lll, T/fact, E/fact, B/fact, X/fact]
#clf()
#plot(lll,np.sqrt(spectra[1]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{TT}$')
#plot(lll,np.sqrt(abs(spectra[4])*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{TE}$')
#plot(lll,np.sqrt(spectra[2]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{EE}$')
#plot(lll,np.sqrt(spectra[3]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{BB}$')
#yscale('log')
#xlim(0,lmaxcamb+1-200)
#ylim(0.001,100)
#xlabel('$\ell$')
#ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
#legend(loc='lower right',frameon=False)
#savefig('toto.png')
print('Got input power spectra from CAMB')




print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)
if len(sys.argv) > 1:
  noise = np.float(sys.argv[1])
  noI = sys.argv[2]
else:
  noise=0.1
  noI=False

strnoise = np.str(noise)
print('Noise level is set to '+strnoise)
print('noI is set to '+np.str(noI))


# parameters
nside = 256
racenter = 0.0      # deg
deccenter = -57.0   # deg
center = equ2gal(racenter, deccenter)
angspeed = 1        # deg/sec
delta_az = 30.      # deg
angspeed_psi = 0.1  # deg/sec
maxpsi = 45.        # deg
nsweeps_el = 300
duration = 24       # hours
ts = 30             # seconds
#coverage_threshold = 0.01
ang = 20.

### Full Instrument
instFull = QubicInstrument(detector_tau=0.0001,
                            detector_sigma=noise,
                            detector_fknee=0.,
                            detector_fslope=1)
plotinst(instFull)
title('Full Insttrument')
savefig('FullInst.png')

#Same with a Single Focal Plane
instSingleFP = instFull[0:992]
plotinst(instSingleFP)
xlim(-0.1,0.2)
title('Single Focal Plane')
savefig('SFP.png')

#Same with One bolo out of two
numbols = np.arange(len(instFull.detector.index))
mask2 = (numbols % 2) == 0
instEven = instFull[mask2]
plotinst(instEven)
title('Even Bolometers')
savefig('Even.png')




#Same with half of the bolos randomly chosen
rndnum = np.random.rand(len(instFull.detector.index))
ind = np.argsort(rndnum)
instRnd = instFull[ind[0:len(instFull.detector.index)/2]]
plotinst(instRnd)
title('Random Bolometers')
savefig('Random.png')

#Now even bolos on X fp and Odd bolos on Y fp
numbols = np.arange(len(instFull.detector.index))
mask2 = (numbols % 2) == 0
mask2[len(mask2)/2:] = ~mask2[len(mask2)/2:]
instEvenOdd = instFull[mask2]
plotinst(instEvenOdd)
title('Even/Odd Bolometers')
savefig('EvenOdd.png')


### First do it once
mrsfp = np.zeros((3,10))
mreven = np.zeros((3,10))
mrevenodd = np.zeros((3,10))
mrrnd = np.zeros((3,10))
x0 = None
plot = False
sampling = create_random_pointings([racenter, deccenter], duration*3600/ts, ang, period=ts)
sampling.angle_hwp = np.random.random_integers(0, 7, len(sampling)) * 11.25 

### make input maps
if rank ==0:
  print('Rank '+str(rank)+' is Running Synfast')
  x0 = np.array(hp.synfast(spectra[1:],nside,fwhm=0,pixwin=True,new=True)).T
  if noI: x0[:,0]*=0

x0 = MPI.COMM_WORLD.bcast(x0)

cbins, mrsfp[0, :], mrsfp[1, :], mrsfp[2, :], pfFull, x0 = get_ratio_rms_cov(instFull, instSingleFP, spectra, sampling, nside, x0=x0, sub1=231, sub2=232, center=center, lab = 'Single FP', plot=plot, 
  refsavefile='mapsref_'+strnoise+'.fits', 
  altsavefile='mapsSFP_'+strnoise+'.fits', 
  refsavefile_noiseless='mapsref_'+strnoise+'_noiseless.fits', 
  altsavefile_noiseless='mapsSFP_'+strnoise+'_noiseless.fits', 
  noI=noI)

cbins, mreven[0, :], mreven[1, :], mreven[2, :], pfFull, x0 = get_ratio_rms_cov(instFull, instEven, spectra, sampling, nside, pfFull=pfFull, x0=x0, sub1=231, sub2=233, center=center, lab = 'Even Bols', plot=plot, altsavefile='mapseven_'+strnoise+'.fits', 
  altsavefile_noiseless='mapseven_'+strnoise+'_noiseless.fits', 
  noI=noI)

cbins, mrevenodd[0, :], mrevenodd[1, :], mrevenodd[2, :], pfFull, x0 = get_ratio_rms_cov(instFull, instEvenOdd, spectra, sampling, nside, pfFull=pfFull, x0=x0, sub1=231, sub2=234, center=center, lab = 'Even/Odd Bols', plot=plot, 
  altsavefile='mapsevenodd_'+strnoise+'.fits', 
  altsavefile_noiseless='mapsevenodd_'+strnoise+'_noiseless.fits',
  noI=noI)

cbins, mrrnd[0, :], mrrnd[1, :], mrrnd[2, :], pfFull, x0 = get_ratio_rms_cov(instFull, instRnd, spectra, sampling, nside, pfFull=pfFull, x0=x0, sub1=231, sub2=235,center=center, lab = 'Rnd Bols', plot=plot, altsavefile='mapsrnd_'+strnoise+'.fits', 
  altsavefile_noiseless='mapsrnd_'+strnoise+'_noiseless.fits',
  noI=noI)



mmrsfp0,ssrsfp0 = meanweight(mrsfp[0][1:],mrsfp[0][1:]*0+1)
mmreven0,ssreven0 = meanweight(mreven[0][1:],mreven[0][1:])
mmrevenodd0,ssrevenodd0 = meanweight(mrevenodd[0][1:],mrevenodd[0][1:]*0+1)
mmrrnd0,ssrrnd0 = meanweight(mrrnd[0][1:],mrrnd[0][1:]*0+1)
mmrsfp1,ssrsfp1 = meanweight(mrsfp[1][1:],mrsfp[1][1:]*0+1)
mmreven1,ssreven1 = meanweight(mreven[1][1:],mreven[1][1:]*0+1)
mmrevenodd1,ssrevenodd1 = meanweight(mrevenodd[1][1:],mrevenodd[1][1:]*0+1)
mmrrnd1,ssrrnd1 = meanweight(mrrnd[1][1:],mrrnd[1][1:]*0+1)
mmrsfp2,ssrsfp2 = meanweight(mrsfp[2][1:],mrsfp[2][1:]*0+1)
mmreven2,ssreven2 = meanweight(mreven[2][1:],mreven[2][1:]*0+1)
mmrevenodd2,ssrevenodd2 = meanweight(mrevenodd[2][1:],mrevenodd[2][1:]*0+1)
mmrrnd2,ssrrnd2 = meanweight(mrrnd[2][1:],mrrnd[2][1:]*0+1)

print('\n\n')
print('############## Results ##################')
print('Stokes I - nside='+str(nside)+' - ts={0:.2f} - ang={1:.2f}'.format(ts,ang))
print('Ratio to Single FP : {0:.2f} +/- {1:.2f}'.format(mmrsfp0,ssrsfp0))
print('Ratio to Even bolos : {0:.2f} +/- {1:.2f}'.format(mmreven0,ssreven0))
print('Ratio to Even/Odd bolos : {0:.2f} +/- {1:.2f}'.format(mmrevenodd0,ssrevenodd0))
print('Ratio to Rnd : {0:.2f} +/- {1:.2f}'.format(mmrrnd0,ssrrnd0))
print('\n')
print('Stokes Q - nside='+str(nside)+' - ts={0:.2f} - ang={1:.2f}'.format(ts,ang))
print('Ratio to Single FP : {0:.2f} +/- {1:.2f}'.format(mmrsfp1,ssrsfp1))
print('Ratio to Even bolos : {0:.2f} +/- {1:.2f}'.format(mmreven1,ssreven1))
print('Ratio to Even/Odd bolos : {0:.2f} +/- {1:.2f}'.format(mmrevenodd1,ssrevenodd1))
print('Ratio to Rnd : {0:.2f} +/- {1:.2f}'.format(mmrrnd1,ssrrnd1))
print('\n')
print('Stokes U - nside='+str(nside)+' - ts={0:.2f} - ang={1:.2f}'.format(ts,ang))
print('Ratio to Single FP : {0:.2f} +/- {1:.2f}'.format(mmrsfp2,ssrsfp2))
print('Ratio to Even bolos : {0:.2f} +/- {1:.2f}'.format(mmreven2,ssreven2))
print('Ratio to Even/Odd bolos : {0:.2f} +/- {1:.2f}'.format(mmrevenodd2,ssrevenodd2))
print('Ratio to Rnd : {0:.2f} +/- {1:.2f}'.format(mmrrnd2,ssrrnd2))
print('\n')


# figure(4)
# clf()
# subplot(3,1,1)
# yscale('log')
# ylim(0.1,1000)
# title('Stokes I - nside='+str(nside)+' - ts={0:.2f} - ang={1:.2f}'.format(ts,ang))
# errorbar(cbins, mrsfp[0], yerr=srsfp[0], fmt='bo-', label='Ratio to Single FP : {0:.2f} +/- {1:.2f}'.format(mmrsfp0,ssrsfp0))
# errorbar(cbins, mreven[0], yerr=sreven[0], fmt='ro-', label='Ratio to Even bolos : {0:.2f} +/- {1:.2f}'.format(mmreven0,ssreven0))
# errorbar(cbins, mrevenodd[0], yerr=srevenodd[0], fmt='mo-', label='Ratio to Even/Odd bolos : {0:.2f} +/- {1:.2f}'.format(mmrevenodd0,ssrevenodd0))
# errorbar(cbins, mrrnd[0], yerr=srrnd[0], fmt='go-', label='Ratio to Rnd : {0:.2f} +/- {1:.2f}'.format(mmrrnd0,ssrrnd0))
# plot(cbins, cbins*0+sqrt(2), 'k:')
# xlabel('Coverage')
# ylabel('RMS Ratio')
# legend(loc='upper right', fontsize=10, frameon=False)
# subplot(3,1,2)
# yscale('log')
# ylim(0.1,1000)
# title('Stokes Q - nside='+str(nside)+' - ts={0:.2f} - ang={1:.2f}'.format(ts,ang))
# errorbar(cbins, mrsfp[1], yerr=srsfp[1], fmt='bo-', label='Ratio to Single FP : {0:.2f} +/- {1:.2f}'.format(mmrsfp1,ssrsfp1))
# errorbar(cbins, mreven[1], yerr=sreven[1], fmt='ro-', label='Ratio to Even bolos : {0:.2f} +/- {1:.2f}'.format(mmreven1,ssreven1))
# errorbar(cbins, mrevenodd[1], yerr=srevenodd[1], fmt='mo-', label='Ratio to Even/Odd bolos : {0:.2f} +/- {1:.2f}'.format(mmrevenodd1,ssrevenodd1))
# errorbar(cbins, mrrnd[1], yerr=srrnd[1], fmt='go-', label='Ratio to Rnd : {0:.2f} +/- {1:.2f}'.format(mmrrnd1,ssrrnd1))
# plot(cbins, cbins*0+sqrt(2), 'k:')
# xlabel('Coverage')
# ylabel('RMS Ratio')
# legend(loc='upper right', fontsize=10, frameon=False)
# subplot(3,1,3)
# yscale('log')
# ylim(0.1,1000)
# title('Stokes U - nside='+str(nside)+' - ts={0:.2f} - ang={1:.2f}'.format(ts,ang))
# errorbar(cbins, mrsfp[2], yerr=srsfp[2], fmt='bo-', label='Ratio to Single FP : {0:.2f} +/- {1:.2f}'.format(mmrsfp2,ssrsfp2))
# errorbar(cbins, mreven[2], yerr=sreven[2], fmt='ro-', label='Ratio to Even bolos : {0:.2f} +/- {1:.2f}'.format(mmreven2,ssreven2))
# errorbar(cbins, mrevenodd[2], yerr=srevenodd[2], fmt='mo-', label='Ratio to Even/Odd bolos : {0:.2f} +/- {1:.2f}'.format(mmrevenodd2,ssrevenodd2))
# errorbar(cbins, mrrnd[2], yerr=srrnd[2], fmt='go-', label='Ratio to Rnd : {0:.2f} +/- {1:.2f}'.format(mmrrnd2,ssrrnd2))
# plot(cbins, cbins*0+sqrt(2), 'k:')
# xlabel('Coverage')
# ylabel('RMS Ratio')
# legend(loc='upper right', fontsize=10, frameon=False)
# savefig('onesim_noiseless_ns'+str(nside)+'_ts{0:.2f}_ang{1:.2f}.png'.format(ts,ang))



