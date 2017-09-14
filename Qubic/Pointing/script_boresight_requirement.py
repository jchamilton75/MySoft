from __future__ import division

import healpy as hp
import numpy as np

from pyoperators import pcg, DiagonalOperator
from qubic import (equ2gal, map2tod, tod2map_all,
                  QubicScene,
                  QubicInstrument, 
                  QubicAcquisition, 
                  QubicSampling,
                  read_spectra,
                  create_random_pointings)
from optparse import OptionParser
from scipy.optimize import curve_fit
from copy import copy
from cPickle import dump, load
import matplotlib.pyplot as mp


nside_in = 1024
nside_out = 128

racenter = 0.0
deccenter = -46.0
center = equ2gal(racenter, deccenter-30)

def corrupt_pointing(pointing,
                    sigma=0, # arcsec
                    seed=None, boresight=False):
  if seed is not None: np.random.seed(seed)
  nsamples = len(pointing)
  if boresight:
    print('Add noise on BORESIGHT')
    newaz = pointing.azimuth.copy()
    newel = pointing.elevation.copy()
    newpitch = pointing.pitch.copy() + np.random.normal(0, sigma / 60 / 60, nsamples)
  else:
    print('Add noise on POINTING')
    newaz = pointing.azimuth.copy() + np.random.normal(0, sigma / 60 / 60, nsamples)
    newel = pointing.elevation.copy() + np.random.normal(0, sigma / 60 / 60, nsamples)
    newpitch = pointing.pitch.copy()
   
  p = QubicSampling(azimuth=newaz, 
                   elevation=newel,
                   pitch=newpitch,
                   period=pointing.period,
                   latitude=pointing.latitude,
                   longitude=pointing.longitude)
  p.angle_hwp = pointing.angle_hwp.copy()
  p.date_obs = pointing.date_obs.copy()
  return p

def run_one_sky_realization(input_map, pointing, corrupted_pointing):
   band = 150
   print 'pointing.comm.size = ', pointing.comm.size
   print 'corrupted_pointing.comm.size = ', corrupted_pointing.comm.size 
   scene_in = QubicScene(nside_in, kind='IQU')
   scene_out = QubicScene(nside_out, kind='IQU')
   acquisition = QubicAcquisition(band, pointing, scene_out, detector_nep=1.e-30, effective_duration=1000)
   acquisition_corrupted = QubicAcquisition(band, corrupted_pointing, scene_in, detector_nep=1.e-30, effective_duration=1000)

   # convolve input maps
   convolution = acquisition_corrupted.get_convolution_peak_operator()
   input_map_convolved = convolution(input_map)

   tod = map2tod(acquisition_corrupted, input_map_convolved)
   input_map_convolved = np.array(hp.ud_grade(input_map_convolved.T, nside_out=nside_out)).T

   tolerance = 1e-4
   reconstructed_map, coverage = tod2map_all(acquisition, tod, tol=tolerance)

   return input_map_convolved, reconstructed_map, coverage

cov_thr = 0.2
nrealizations = 1
nbsim = 3
sigma = np.append(0, np.logspace(-2, np.log10(0.5), nbsim)*3600)
Tval = np.array([0,1])
boresight = np.array([True, False])

noise_std = np.zeros((len(Tval), len(boresight), len(sigma), nrealizations))

for isigma, sigma_pointing in enumerate(sigma):
  print 'sigma =', sigma_pointing
  for realization in xrange(nrealizations):
    for it in xrange(len(Tval)):
      for jb in xrange(len(boresight)):
        print 'realization number', realization
        np.random.seed(realization)
        pointing = create_random_pointings(equ2gal(racenter, deccenter), 1000, 10)
        if sigma_pointing != 0:
          pointing_corrupted = corrupt_pointing(pointing, 
                                                 sigma=sigma_pointing,
                                                 seed=realization, boresight=boresight[jb])
        else:
          pointing_corrupted = pointing

        spectra = read_spectra(0)
        input_map = np.array(hp.synfast(spectra, nside_in,
                                       fwhm=0, pixwin=True, new=True,                              
                                       verbose=False)).T
        input_map[:,0] *= Tval[it]
        print('MEAN, STD TEMPERATURE: {0:} , {1:}'.format(np.mean(input_map[:,0]), np.std(input_map[:,0])))

        input_map_convolved, reconstructed_map, coverage = run_one_sky_realization(input_map, pointing, pointing_corrupted)
        mask = coverage > coverage.max() * cov_thr

        #hp.gnomview(input_map_convolved[:,1], rot=center, reso=20,min=-5,max=5)
        #hp.gnomview(reconstructed_map[:,1], rot=center, reso=20,min=-5,max=5)
        #hp.gnomview(reconstructed_map[:,1]-input_map_convolved[:,1], rot=center, reso=20,min=-5,max=5)

        noise_std[it, jb, isigma, realization] = (reconstructed_map - input_map_convolved)[mask, 1:].std()

with open('noise_std_boresight.pkl', 'w') as f:
   dump(noise_std, f)


with open('noise_std_boresight_1024.pkl', 'r') as f:
       noise_std = load(f)

# for file called _old
#nbsim=20
#sigma = np.append(0, np.logspace(-2, np.log10(0.5), nbsim)*3600)

# for new files:
nbsim=40
sigma = np.append(0, np.logspace(-4, np.log10(0.5), nbsim)*3600)


mstd = np.mean(noise_std, axis=3)
sstd = np.std(noise_std, axis=3)/np.sqrt(nrealizations)

clf()
decal=0
plot((sigma+decal), mstd[0,0,:]/mstd[0,0,1], 'b-', label='Boresight, T=0')
plot((sigma+decal), mstd[1,0,:]/mstd[1,0,1], 'b--', label='Boresight, T=1')
plot((sigma+decal), mstd[0,1,:]/mstd[0,1,1], 'r-', label='Pointing, T=0')
plot((sigma+decal), mstd[1,1,:]/mstd[1,1,1], 'r--', label='Pointing, T=1')
mp.xlabel('Pointing error [Arcsec]')
mp.ylabel('Residual $\sigma$ relative')
legend()
ylim(0.99, 1.2)
xlim(-0.01, 1800)

clf()
decal=0
plot(sigma+decal, mstd[0,0,:], 'b-', label='Boresight, T=0')
plot(sigma+decal, mstd[1,0,:], 'b--', label='Boresight, T=1')
plot(sigma+decal, mstd[0,1,:], 'r-', label='Pointing, T=0')
plot(sigma+decal, mstd[1,1,:], 'r--', label='Pointing, T=1')
mp.xlabel('Pointing error [Arcsec]')
mp.ylabel('Residual $\sigma$')
legend()
ylim(0,12)
xlim(0, 1800)


azel_ac = 3.*60
azel_ac_fin = 15.
bore_ac = 5.*60


clf()
decal=0
ratio_bore = mstd[1,0,:]/mstd[1,0,1]
ratio_bore[0]=1
ratio_ptg = mstd[1,1,:]/mstd[1,1,1]
ratio_ptg[0]=1
plot(sigma+decal, ratio_bore, 'bo-', label='Boresight')
plot(sigma+decal, ratio_ptg, 'ro-', label='Pointing')
mp.xlabel('Pointing error [Arcsec]')
mp.ylabel('Residual $\sigma$ relative on Q,U maps')
mp.plot([azel_ac, azel_ac], [0, 100], ls='--', c='m', lw=2, label = 'Mount Az, El Mechanical Accuracy: {0:3.0f} arcmin'.format(azel_ac/60))
mp.plot([azel_ac_fin, azel_ac_fin], [0, 100], ls='--', c='r', lw=2, label = 'Az, El Reconstruction Accuracy: {0:3.0f} arcsec'.format(azel_ac_fin))
mp.plot([bore_ac, bore_ac], [0, 100], ls='--', c='b', lw=2, label = 'Mount Boresight Mechanical Accuracy: {0:3.0f} arcmin'.format(bore_ac/60))
plot(sigma, sigma*0+1,'k:')
plot(sigma, sigma*0+1.005,'k--')
legend(numpoints=1, fontsize=10, loc='upper left')
ylim(0.998,1.1)
#xlim(0, 130)
title('QUBIC Pointing Accuracy Requirements')
xscale('log')
xlim(1, 1800)
savefig('requirements_both.png')


clf()
decal=0
ratio_ptg = mstd[1,1,:]/mstd[1,1,1]
ratio_ptg[0]=1
plot(sigma+decal, ratio_ptg, 'ro-', label='Pointing')
mp.xlabel('Pointing error on Az and El. [Arcsec]')
mp.ylabel('Residual $\sigma$ relative on Q,U maps')
mp.plot([azel_ac, azel_ac], [0, 100], ls='--', c='m', lw=2, label = 'Mount Az, El Mechanical Accuracy: {0:3.0f} arcmin'.format(azel_ac/60))
mp.plot([azel_ac_fin, azel_ac_fin], [0, 100], ls='--', c='r', lw=2, label = 'Az, El Reconstruction Accuracy: {0:3.0f} arcsec'.format(azel_ac_fin))
plot(sigma, sigma*0+1,'k:')
plot(sigma, sigma*0+1.005,'k--')
ylim(0.998,1.1)
xlim(0, 200)
legend(numpoints=1, fontsize=10, loc='upper left')
title('QUBIC Pointing Accuracy Requirements')
savefig('requirements_pointing.png')


clf()
decal=0
ratio_bore = mstd[1,0,:]/mstd[1,0,1]
ratio_bore[0]=1
plot((sigma+decal)/60, ratio_bore, 'bo-', label='Boresight')
mp.xlabel('Boresight angle error [Arcmin]')
mp.ylabel('Residual $\sigma$ relative on Q,U maps')
mp.plot([bore_ac/60, bore_ac/60], [0, 100], ls='--', c='b', lw=2, label = 'Mount Boresight Mechanical Accuracy: {0:3.0f} arcmin'.format(bore_ac/60))
plot(sigma, sigma*0+1,'k:')
plot(sigma, sigma*0+1.005,'k--')
ylim(0.998,1.05)
xlim(0, 1800./60)
legend(numpoints=1, fontsize=10, loc='upper left')
title('QUBIC Boresight Accuracy Requirements')
savefig('requirements_boresight.png')



azel_std = noise_std.mean(axis=1)
azel_err = noise_std.std(axis=1)/np.sqrt(nrealizations)
n = azel_std[0]
azel_std /= n
azel_err /= n
#mp.subplot(211)
clf()
mp.errorbar(sigma/60/60, azel_std / azel_std[0], yerr=azel_err / azel_std[0], fmt='ro')
mp.plot([20./60/60, 20./60/60], [0.98, 1.2], ls='--', c='k')
mp.plot([120./60/60, 120./60/60], [0.98, 1.2], ls='--', c='k')
# def f(x, p0, p1): return p0 * x**p1
# s = azel_std - 1.
# p, pcov = curve_fit(f, sigma, s)
mp.xlim(-10./60/60, 250./60/60)
mp.ylim(0.98, 1.05)
mp.xlabel('Pointing error, [deg]')
mp.ylabel('Residual $\sigma$, relative')
# x = np.arange(0., 30., 0.1)
# mp.plot(x, f(x, p[0], p[1]) + 1.)

#################################################################

### This needs to run stuff on NERSC (to get pure reconstruction)

bb = np.empty((len(sigma), nrealizations))
for isigma, sigma_pointing in enumerate(sigma):
   for realization in xrange(nrealizations):
       s = hp.mrdfits('cls/cellpure_rec_map_sigma{}_r{}_mask1_0_0.fits'.format(sigma_pointing, realization))
       bb[isigma, realization] = s[3][0]

bb_mean = bb.mean(axis=1)
bb_std = bb.std(axis=1)
n = bb_mean[0]
bb_mean /= n
bb_std /= n
mp.errorbar(sigma, bb_mean, yerr=bb_std)
mp.plot([20., 20.], [0.8, 2.], ls='--', c='k')
mp.plot([120., 120.], [0.8, 2.], ls='--', c='k')
mp.plot([-50, 300], [bb_mean[1], bb_mean[1]], ls='--', c='k')
mp.plot([-50, 300], [bb_mean[4], bb_mean[4]], ls='--', c='k')
mp.xlim(-10., 250.)
mp.ylim(0.8, 1.6)
mp.xlabel('Pointing error, [arcsec]')
mp.ylabel('$C_\ell^{BB}$, relative')








###### A test
nside_in=256
nside_out=128
sigma = 5000


nn=1000
realization=0
np.random.seed(realization)
spectra = read_spectra(0)
input_map = np.array(hp.synfast(spectra, nside_in,
                               fwhm=0, pixwin=True, new=True,                              
                               verbose=False)).T
input_map[:,0]=0
pointing = create_random_pointings(equ2gal(racenter, deccenter), nn, 10)
corrupted_pointing = corrupt_pointing(pointing, 
                                     sigma=sigma,
                                     seed=realization)

band = 150
print 'pointing.comm.size = ', pointing.comm.size
print 'corrupted_pointing.comm.size = ', corrupted_pointing.comm.size 
scene_in = QubicScene(nside_in, kind='IQU')
scene_out = QubicScene(nside_out, kind='IQU')
acquisition = QubicAcquisition(band, pointing, scene_out, detector_nep=1.e-30, effective_duration=1)
acquisition_corrupted = QubicAcquisition(band, corrupted_pointing, scene_in, detector_nep=1.e-30, effective_duration=1)

# convolve input maps
convolution = acquisition_corrupted.get_convolution_peak_operator()
input_map_convolved = convolution(input_map)

tod = map2tod(acquisition_corrupted, input_map_convolved)

input_map_convolved, reconstructed_map, coverage = run_one_sky_realization(input_map, pointing, corrupted_pointing)
mask = coverage > coverage.max() * cov_thr

#hp.gnomview(input_map_convolved[:,1], rot=center, reso=20,min=-5,max=5)
#hp.gnomview(reconstructed_map[:,1], rot=center, reso=20,min=-5,max=5)
hp.gnomview(reconstructed_map[:,1]-input_map_convolved[:,1], rot=center, reso=20,min=-5,max=5)




