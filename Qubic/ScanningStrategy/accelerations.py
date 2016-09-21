from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
import healpy as hp
from pyoperators import MPI, DiagonalOperator, PackOperator, pcg
from qubic import (
    QubicAcquisition, QubicInstrument,
    QubicScene, create_sweeping_pointings, equ2gal, create_random_pointings, PlanckAcquisition, QubicPlanckAcquisition)
from qubic.io import read_map, write_map
import gc
import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)
import sys
import os
from qubic.data import PATH


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
duration = 24.       # hours
ts = duration*3600/2**23            # seconds Chosen in order to have a power of 2 in 
center = equ2gal(racenter, deccenter)

####### Create some sampling
center = equ2gal(racenter, deccenter)

sadlc = np.array([-24.18947, -66.472016])

sampling = create_sweeping_pointings(
    [racenter, deccenter], duration, ts*10, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi, latitude=sadlc[0], longitude=sadlc[1])


ok = np.abs(sampling.elevation-50)<20
samplingok = sampling[ok]

samplingok.time = 3600*(((samplingok.time/3600 +36) % 24)-12)
clf()
subplot(2,1,1)
plot(samplingok.time/3600, samplingok.azimuth,',')
xlabel('Time [Hours]')
ylabel('Azimuth [Deg.]')
subplot(2,1,2)
plot(samplingok.time/3600, samplingok.elevation,',')
xlabel('Time [Hours]')
ylabel('Elevation [Deg.]')
savefig('daily_azel.png')




####### short time
from scipy.ndimage import gaussian_filter1d
duration =0.1
sampling = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi, latitude=sadlc[0], longitude=sadlc[1])


ok = np.abs(sampling.elevation-50)<20
samplingok = sampling[ok]

sm = 1./ts
az = gaussian_filter1d(sampling.azimuth, sm)
azprime = np.diff(az)/np.diff(samplingok.time)
tt = samplingok.time[0:-1]
azsec = np.diff(azprime)/np.diff(tt)
tt2 = samplingok.time[0:-2]
size = 1. #m
acc = np.radians(azsec)*size

samplingok.time = 3600*(((samplingok.time/3600 +36) % 24)-12)
clf()
subplot(3,1,1)
plot(samplingok.time/60, az)
xlabel('Time [Minutes]')
ylabel('Azimuth [Deg.]')
xlim(0,2)
subplot(3,1,2)
plot(tt/60, azprime)
xlabel('Time [Minutes]')
ylabel('Angular speed [deg/s]')
ylim(-1.5,1.5)
xlim(0,2)
subplot(3,1,3)
plot(tt2/60, acc)
xlabel('Time [Minutes]')
ylabel('Acceleration [$m/s^2$]')
xlim(0,2)
savefig('typical_acc.png')

