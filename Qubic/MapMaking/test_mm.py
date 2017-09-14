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


racenter = 0.0
deccenter = -46.0

def corrupt_pointing(pointing,
                    sigma=0, # arcsec
                    seed=None):
   nsamples = len(pointing)
   p = QubicSampling(azimuth=pointing.azimuth.copy()+np.random.normal(0, sigma / 60 / 60, nsamples), 
                     elevation=pointing.elevation.copy()+np.random.normal(0, sigma / 60 / 60, nsamples),
                     pitch=pointing.pitch.copy(),
                     period=pointing.period,
                     latitude=pointing.latitude,
                     longitude=pointing.longitude)
   p.angle_hwp = pointing.angle_hwp.copy()
   p.date_obs = pointing.date_obs.copy()
   if seed is not None: np.random.seed(seed)
   return p


def run_one_sky_realization(input_map_hi, pointing, corrupted_pointing, nside_out):
   band = 150
   print 'pointing.comm.size = ', pointing.comm.size
   print 'corrupted_pointing.comm.size = ', corrupted_pointing.comm.size 
   scene = QubicScene(nside_out, kind='IQU')
   thenside = hp.npix2nside(len(input_map_hi[:,0]))
   scene_hi = QubicScene(thenside, kind='IQU')

   acquisition = QubicAcquisition(band, pointing, scene, detector_nep=1.e-30, photon_noise=False)
   acquisition_corrupted = QubicAcquisition(band, corrupted_pointing, scene, detector_nep=1.e-30, photon_noise=False)

   acquisition_corrupted_hi = QubicAcquisition(band, corrupted_pointing, scene_hi, detector_nep=1.e-30, photon_noise=False)

   # convolve input maps
   convolution = acquisition_corrupted_hi.get_convolution_peak_operator()
   input_map_convolved_hi = convolution(input_map_hi)

   tod = map2tod(acquisition_corrupted_hi, input_map_hi)

   tolerance = 1e-4
   reconstructed_map, coverage = tod2map_all(acquisition, tod, tol=tolerance)

   return np.array(hp.ud_grade(input_map_convolved_hi.T, nside_out=nside_out)).T, reconstructed_map, coverage





nside = 128
super_resol = [1, 2, 4]

cov_thr = 0.2
#sigma_pointing = np.linspace(0.000001, 20., nb)
sigma_pointing = np.array([0.000001, 1., 10., 20., 50., 100., 200., 500., 1000.])
nb = len(sigma_pointing)
rms = np.zeros((nb, len(super_resol)))

for s in xrange(len(super_resol)):
    for i in xrange(nb):
        realization= i
        np.random.seed(realization)
        spectra = read_spectra(0)
        input_map_hi = np.array(hp.synfast(spectra, nside*super_resol[s],
                                   fwhm=0, pixwin=True, new=True,                              
                                   verbose=False)).T

        pointing = create_random_pointings(equ2gal(racenter, deccenter), 1000, 10)
        pointing_corrupted = corrupt_pointing(pointing, 
                                            sigma=sigma_pointing[i],
                                            seed=realization)

        input_map_convolved, reconstructed_map, coverage = run_one_sky_realization(input_map_hi, pointing, pointing_corrupted, nside_out=nside)
     
        mask = coverage > coverage.max() * cov_thr
        rms[i,s]=(reconstructed_map - input_map_convolved)[mask, 1:].std()



clf()
for s in xrange(len(super_resol)):
    plot(sigma_pointing, rms[:,s]/rms[0,s],'o-', label='super_resol={}'.format(super_resol[s]))
legend(loc='upper left')
xscale('log')
ylim(0.9,2)
#yscale('log')





