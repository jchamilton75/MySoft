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

def corrupt_pointing(pointing,
                    sigma=0, # arcsec
                    seed=None):
   if seed is not None: np.random.seed(seed)
   nsamples = len(pointing)
   p = QubicSampling(azimuth=pointing.azimuth.copy() + np.random.normal(0, sigma / 60 / 60, nsamples), 
                     elevation=pointing.elevation.copy() + np.random.normal(0, sigma / 60 / 60, nsamples),
                     pitch=pointing.pitch.copy(),
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
   acquisition = QubicAcquisition(band, pointing, scene_out, detector_nep=0.)
   acquisition_corrupted = QubicAcquisition(band, corrupted_pointing, scene_in, detector_nep=0.)

   # convolve input maps
   convolution = acquisition_corrupted.get_convolution_peak_operator()
   input_map_convolved = convolution(input_map)

   tod = map2tod(acquisition_corrupted, input_map_convolved)
   input_map_convolved = np.array(hp.ud_grade(input_map_convolved.T, nside_out=nside_out)).T

   tolerance = 1e-4
   reconstructed_map, coverage = tod2map_all(acquisition, tod, tol=tolerance)

   return input_map_convolved, reconstructed_map, coverage

cov_thr = 0.2
nrealizations = 2
sigma = np.arange(0, 70, 20.)
noise_std = np.empty((len(sigma), nrealizations))
for isigma, sigma_pointing in enumerate(sigma):
   print 'sigma =', sigma_pointing
   for realization in xrange(nrealizations):
       print 'realization number', realization
       np.random.seed(realization)
       spectra = read_spectra(0)
       input_map = np.array(hp.synfast(spectra, nside_in,
                                       fwhm=0, pixwin=True, new=True,                              
                                       verbose=False)).T
       pointing = create_random_pointings(equ2gal(racenter, deccenter), 1000, 10)
       if sigma_pointing != 0:
           pointing_corrupted = corrupt_pointing(pointing, 
                                                 sigma=sigma_pointing,
                                                 seed=realization)
       else:
           pointing_corrupted = pointing
       input_map_convolved, reconstructed_map, coverage = run_one_sky_realization(input_map, pointing, pointing_corrupted)
       mask = coverage > coverage.max() * cov_thr
       noise_std[isigma, realization] = (reconstructed_map - input_map_convolved)[mask, 1:].std()
with open('noise_std.pkl', 'w') as f:
  dump(noise_std, f)

with open('noise_std.pkl', 'r') as f:
      noise_std = load(f)


azel_std = noise_std.mean(axis=1)
azel_err = noise_std.std(axis=1)
n = azel_std[0]
azel_std /= n
azel_err /= n
#mp.subplot(211)
mp.errorbar(sig, azel_std / azel_std[0], yerr=azel_err / azel_std[0])
mp.plot([20., 20.], [0.8, 1.2], ls='--', c='k')
mp.plot([120., 120.], [0.8, 1.2], ls='--', c='k')
# def f(x, p0, p1): return p0 * x**p1
# s = azel_std - 1.
# p, pcov = curve_fit(f, sigma, s)
mp.xlim(-10., 250.)
mp.ylim(0.95, 1.2)
mp.xlabel('Pointing error, [arcsec]')
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




