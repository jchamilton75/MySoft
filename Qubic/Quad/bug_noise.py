from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
import healpy as hp
from pyoperators import MPI, DiagonalOperator, PackOperator, pcg
from qubic import (
    QubicAcquisition, QubicInstrument,
    QubicScene, create_sweeping_pointings, equ2gal, create_random_pointings)
from MYacquisition import PlanckAcquisition, QubicPlanckAcquisition
from qubic.io import read_map, write_map
import gc
import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)
import sys
import os
from qubic.data import PATH

### Input map set to zero
nside_in=64
mapi= np.zeros(12*nside_in**2)
mapq= np.zeros(12*nside_in**2)
mapu= np.zeros(12*nside_in**2)
input_maps=np.array([mapi,mapq,mapu]).T



######## Simulation parameters
maxiter = 1000
tol = 5e-6
racenter = 0.0      # deg
deccenter = -57.0   # deg
angspeed = 1        # deg/sec
delta_az = 20.      # deg
angspeed_psi = 0.1  # deg/sec
maxpsi = 45.        # deg
nsweeps_el = 300
duration = 24       # hours
ts = 5.             # seconds
np.random.seed(0)
center = equ2gal(racenter, deccenter)
covlim=0.1
effective_duration = 1. ## Years

####### Create some sampling
sampling = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)
    
### Prepare input / output data
nside = 64
scene = QubicScene(nside, kind='IQU')

noise_vals = logspace(-5,0,10)
valspix = []
valsstd = []
for n in noise_vals:
    instrument = QubicInstrument(filter_nu=150e9,
                    detector_nep=n*4.7e-17)

    acq = QubicAcquisition(instrument, sampling, scene, photon_noise=False, 
                            effective_duration=effective_duration)
    noise = acq.get_noise()
    valspix.append(std(noise, axis=1))
    valsstd.append(std(noise))

clf()
plot(noise_vals,valsstd)
xscale('log')

########## Important: mettre photon_noise = False dans QubicAcquisition quand on veut baisser le niveau de bruit !
########## Maintenant il y a le keyword effective_duration (en années) qui scale le bruit (valable uniquement sous l'hypothèse de bruit blanc hélas)




sampling = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)


instrument = QubicInstrument(filter_nu=150e9,
                detector_nep=4.7e-17)

acq = QubicAcquisition(instrument, sampling, scene, photon_noise=False, effective_duration=1)
noise = acq.get_noise()

acq_with_photon = QubicAcquisition(instrument, sampling, scene, photon_noise=True, effective_duration=1)
noise_wp = acq_with_photon.get_noise()

signoise_det = std(noise, axis=1)
signoise_det_wp = std(noise_wp, axis=1)

clf()
plot(signoise_det)
plot(signoise_det_wp)




####### Create some sampling with nominal ts and short duration
sampling2 = create_sweeping_pointings(
    [racenter, deccenter], duration*0+0.1, ts*0+0.01, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)
    

instrument = QubicInstrument(filter_nu=150e9,
                detector_nep=4.7e-17)

acq = QubicAcquisition(instrument, sampling, scene, photon_noise=False)
noise = acq.get_noise()

acq_with_photon = QubicAcquisition(instrument, sampling, scene, photon_noise=True)
noise_wp = acq_with_photon.get_noise()

signoise_det = std(noise, axis=1)
signoise_det_wp = std(noise_wp, axis=1)

clf()
plot(signoise_det)
plot(signoise_det_wp)





