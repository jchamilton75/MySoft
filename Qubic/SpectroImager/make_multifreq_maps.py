from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
#from Quad import qml
#from Quad import pyquad
import healpy as hp
from pyoperators import MPI, BlockDiagonalOperator, BlockRowOperator,BlockColumnOperator, DiagonalOperator, PackOperator, pcg , asoperator , MaskOperator
from pysimulators.interfaces.healpy import (HealpixConvolutionGaussianOperator)
from pysimulators import FitsArray
from qubic import (
    QubicAcquisition, QubicInstrument,
    QubicScene, create_sweeping_pointings, equ2gal, create_random_pointings, PlanckAcquisition, QubicPlanckAcquisition)
#from MYacquisition import PlanckAcquisition, QubicPlanckAcquisition
from qubic.io import read_map, write_map
import gc
import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)
import sys
import os
from qubic.data import PATH
from functools import reduce
import pycamb

from SpectroImager import SpectroImagerGohar as sig
from qubic import QubicSampling



reload(sig)
tod_file = sys.argv[1]
sampling_file = sys.argv[2]
dnu_nu_all = np.float(sys.argv[3])
nsubfreq = np.float(sys.argv[4])
nu_center = np.float(sys.argv[5])
subdelta_reconstruction=np.float(sys.argv[6])
nside=np.float(sys.argv[7])

# tod_file = 'TOD_True_0.0_0.25_150.0_5e-28_0.01_1000_20.fits'
# sampling_file = 'SAMPLING_True_0.0_0.25_150.0_5e-28_0.01_1000_20.fits'
# dnu_nu_all=0.25
# nsubfreq = 2
# nu_center=150.
# subdelta_reconstruction=0.125/2
# nside=256


nu_min=nu_center*(1.-dnu_nu_all/2)
nu_max=nu_center*(1.+dnu_nu_all/2)
dnu_nu = dnu_nu_all/nsubfreq

Nbpixels = 12*nside**2
scene = QubicScene(nside, kind='IQU')

print('############ Reading CMB File')
cmb = FitsArray('CMB_MAPS.fits')
print('############ Reading DUST File')
dust = FitsArray('DUST_MAPS.fits')
print('############ Reading TOD File: {}'.format(tod_file))
Y = FitsArray(tod_file)
print('############ Reading Sampling File: {}'.format(sampling_file))
bla = FitsArray(sampling_file)
sampling = QubicSampling(bla[0],bla[1],bla[2],bla[3])


print('############ DELTANU/NU {0:6.3f}'.format(dnu_nu))
print('############     Sub-Delta Reconstruction : {0:6.3f}'.format(subdelta_reconstruction))


print('######## Making Convolution')
x0_convolved=sig.convolved_true_maps(nu_min,nu_max,dnu_nu,subdelta_reconstruction,cmb,dust, 
                                     verbose=False)
FitsArray(x0_convolved, copy=False).save('x0convolved_{0:}_{1:}_{2:}_{3:}_{4:}_{5:}.fits'.format(nu_center, 
	nu_min, nu_max, dnu_nu, subdelta_reconstruction, nside))



### Maps
effective_duration=1
Nbpixels = 12*nside**2
scene = QubicScene(nside, kind='IQU')
maps, bands, deltas=sig.reconstruct(Y,nu_min,nu_max,dnu_nu,subdelta_reconstruction,
                                               sampling,scene,effective_duration, 
                                               return_mono=False, verbose=False)

FitsArray(maps, copy=False).save('OUTMAPS_{0:}_{1:}_{2:}_{3:}_{4:}_{5:}.fits'.format(nu_center, 
	nu_min, nu_max, dnu_nu, subdelta_reconstruction, nside))








