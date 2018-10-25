#!/bin/env python
from __future__ import division
import sys
path = "/obs/jhamilton/.local/lib/python2.7/site-packages/"
sys.path.insert(0,path)
import healpy as hp
import numpy as np
print(np.version.version)

import matplotlib.pyplot as mp
from qubic import (create_random_pointings, gal2equ,
                  read_spectra,
                  compute_freq,
                  QubicScene,
                  QubicMultibandInstrument,
                  QubicMultibandAcquisition,
                  PlanckAcquisition,
                  QubicMultibandPlanckAcquisition)
import qubic
from SpectroImager import SpectroImLib as si
from pysimulators import FitsArray
import time

import os

#python /Users/hamilton/Python/MySoft/Qubic/SpectroImager/MCmpi.py ./test2 nb_ptg 500 nf_sub_rec 1 tol 1e-3 seed 1
#python /Users/hamilton/Python/MySoft/Qubic/SpectroImager/MCmpi.py ./test1 nb_ptg 500 nf_sub_rec 1 tol 1e-3 seed 1


rep = '/Users/hamilton/Qubic/SpectroImager/TestSeed/'
def minmax(x):
	return np.min(x), np.max(x)



### Compare Pointing: they are identical
eq1 = FitsArray(rep+'test1_equatorial.fits')
eq2 = FitsArray(rep+'test2_equatorial.fits')
hwp1 = FitsArray(rep+'test1_angle_hwp.fits')
hwp2 = FitsArray(rep+'test2_angle_hwp.fits')
print(minmax(eq1-eq2))
print(minmax(hwp1-hwp2))


### Compare Input Maps: they are identical
import pickle
mapsin_1 = FitsArray(rep+'test1_maps_input.fits')
mapsin_2 = FitsArray(rep+'test2_maps_input.fits')
mapsin_1_p = pickle.load( open( rep+'test1_maps_input.p', "rb" ) )
mapsin_2_p = pickle.load( open( rep+'test2_maps_input.p', "rb" ) )

print(minmax(mapsin_1-mapsin_2))
print(minmax(mapsin_1_p-mapsin_2_p))

for i in xrange(15):
	for j in [0,1,2]:
		print(i,j,minmax(mapsin_1[i,:,j]-mapsin_2[i,:,j]))




### Compare TOD:
### Result: 
tod_1 = FitsArray(rep+'test1_TOD.fits')
tod_2 = FitsArray(rep+'test2_TOD.fits')
print(minmax(tod_1-tod_2))




### Compare Maps Out They differ strongly
mapsout_1 = FitsArray(rep+'test1_nf1_maps_recon.fits')
mapsout_2 = FitsArray(rep+'test2_nf1_maps_recon.fits')
print(minmax(mapsout_1-mapsout_2))


### Compare convolved input maps: amazingly they are different ! Why, the input maps are identical
mapsinconv_1 = FitsArray(rep+'test1_nf1_maps_convolved.fits')
mapsinconv_2 = FitsArray(rep+'test2_nf1_maps_convolved.fits')
print(minmax(mapsinconv_1-mapsinconv_2))










