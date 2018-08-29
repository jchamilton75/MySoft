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
import glob


rep = '/Users/hamilton/CMB/Qubic/SpectroImager/TestSeed/'
name = 'try1e-3'



filesout = glob.glob(rep+name+'_nf*_maps_recon.fits')
filesinconv = glob.glob(rep+name+'_nf*_maps_convolved.fits')
maxnfreq = len(filesout)
center_gal = qubic.equ2gal(0., -57.)


# Now read all maps
allmapsout = []
allmapsinconv = []
for i in arange(maxnfreq):
	mapsout = FitsArray(filesout[i])
	allmapsout.append(mapsout)
	mapsinconv = FitsArray(filesinconv[i])
	allmapsinconv.append(mapsinconv)

nn=0
maps_convolved = allmapsinconv[nn]
maps_recon = allmapsout[nn]
diffmap = maps_recon - maps_convolved
diffmap[maps_recon == hp.UNSEEN] = hp.UNSEEN
nf_sub_rec = nn+1

stokes = ['I', 'Q', 'U']
for istokes in [0,1,2]:
    plt.figure(istokes) 
    if istokes==0:
        xr=200 
    else:
        xr=10
    for i in xrange(nf_sub_rec):
        # proxy to get nf_sub_rec maps convolved
        rms = np.std(diffmap[i,maps_convolved[i,:,istokes] != hp.UNSEEN,istokes])
        in_old=hp.gnomview(maps_convolved[i,:,istokes], rot=center_gal, reso=10, sub=(nf_sub_rec,3,3*i+1), 
        	min=-xr, max=xr,title='In '+stokes[istokes]+' Sub{}'.format(i), return_projected_map=True)
        out_old=hp.gnomview(maps_recon[i,:,istokes], rot=center_gal, reso=10,sub=(nf_sub_rec,3,3*i+2), 
        	min=-xr, max=xr,title='Out '+stokes[istokes]+' Sub{}'.format(i), return_projected_map=True)
        res_old=hp.gnomview(diffmap[i,:,istokes], rot=center_gal, reso=10,sub=(nf_sub_rec,3,3*i+3), 
        	min=-xr, max=xr,title='Res '+stokes[istokes]+' Sub{0} \n RMS={1:5.3f}'.format(i,rms), return_projected_map=True)






