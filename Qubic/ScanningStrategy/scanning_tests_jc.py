from __future__ import division
from pyoperators import pcg
from pysimulators import profile
from qubic import (
    create_random_pointings, equ2gal, QubicAcquisition, PlanckAcquisition,
    QubicPlanckAcquisition, create_sweeping_pointings)
from qubic.data import PATH
from qubic.io import read_map
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np

nside = 64
maxiter = 1000
tol = 5e-6
racenter = 0.0      # deg
deccenter = -57.0   # deg
angspeed = 1        # deg/sec
delta_az = 15.      # deg
angspeed_psi = 0.1  # deg/sec
maxpsi = 45.        # deg
nsweeps_el = 300
duration = 24       # hours
ts = 60             # seconds
np.random.seed(0)
sky = read_map(PATH + 'syn256_pol.fits')

center = equ2gal(racenter, deccenter)

# some display
def display(input, msg, iplot=1):
    out = []
    for i, (kind, lim) in enumerate(zip('IQU', [50, 5, 5])):
        map = input[..., i]
        out += [hp.gnomview(map, rot=center, reso=5, xsize=800, min=-lim,
                            max=lim, title=msg + ' ' + kind,
                            sub=(3, 3, iplot + i), return_projected_map=True)]
    return out



###### Study the required sampling... (save CPU)
samplings = np.array([2.5, 5., 10., 30., 60.])

allx = []
ally = []
allcov = []
for thets in samplings:
    # get the sampling model
    sampling = create_sweeping_pointings(
        [racenter, deccenter], duration, thets, angspeed, delta_az, nsweeps_el,
        angspeed_psi, maxpsi)

    detector_nep = 4.7e-17*np.sqrt(len(sampling) * sampling.period / (365 * 86400))
    acq_qubic = QubicAcquisition(150, sampling, nside=nside,
                             detector_nep=detector_nep)
                            
    coverage_map = acq_qubic.get_coverage()
    coverage_map = coverage_map / np.max(coverage_map)
    angmax = hp.pix2ang(nside, coverage_map.argmax())
    maxloc = np.array([np.degrees(angmax[1]), 90.-np.degrees(angmax[0])])

    figure(0)
    cov = hp.gnomview(coverage_map, rot=maxloc, reso=5, xsize=800, return_projected_map=True,sub=(2,1,1))
    subplot(2,1,2)
    x, y = profile(cov)
    x *= 5 / 60
    y *= np.degrees(np.sqrt(4 * np.pi / acq_qubic.scene.npixel))
    plot(x, y)

    allx.append(x)
    ally.append(y)
    allcov.append(coverage_map)

clf()
for i in xrange(len(samplings)):
    plot(allx[i], ally[i], label=samplings[i])
legend()

clf()
for i in xrange(1,len(samplings)-1):
    plot(allx[i], ally[i]/ally[0], label=samplings[i])
legend()
ylim(0,2)




#### OK Use 5sec
###### Study the required sampling... (save CPU)
ts = 5. #seconds
alldaz = np.array([10., 15., 20., 25., 30., 40., 50. ])
nside=128
allx = []
ally = []
allcov = []
for thedaz in alldaz:
    # get the sampling model
    sampling = create_sweeping_pointings(
        [racenter, deccenter], duration, ts, angspeed, thedaz, nsweeps_el,
        angspeed_psi, maxpsi)

    detector_nep = 4.7e-17*np.sqrt(len(sampling) * sampling.period / (365 * 86400))
    acq_qubic = QubicAcquisition(150, sampling, nside=nside,
                             detector_nep=detector_nep)
                            
    coverage_map = acq_qubic.get_coverage()
    coverage_map = coverage_map / np.max(coverage_map)
    angmax = hp.pix2ang(nside, coverage_map.argmax())
    maxloc = np.array([np.degrees(angmax[1]), 90.-np.degrees(angmax[0])])

    figure(0)
    cov = hp.gnomview(coverage_map, rot=maxloc, reso=5, xsize=800, return_projected_map=True,sub=(2,1,1))
    subplot(2,1,2)
    x, y = profile(cov)
    x *= 5 / 60
    y *= np.degrees(np.sqrt(4 * np.pi / acq_qubic.scene.npixel))
    plot(x, y)

    allx.append(x)
    ally.append(y)
    allcov.append(coverage_map)

clf()
for i in xrange(len(alldaz)):
    plot(allx[i], ally[i], label=alldaz[i])
legend()



omega_vals = np.zeros(len(alldaz))
eta_vals = np.zeros(len(alldaz))
for i in xrange(len(alldaz)):
    cov = allcov[i]
    cov[cov < 0.1] = 0
    omega_vals[i] = cov.sum() / len(cov) * 4 * np.pi
    eta_vals[i] = cov.sum() / np.sum(cov**2)
fsky_vals = omega_vals / 4 / np.pi    

clf()
subplot(2,1,1)
plot(alldaz, fsky_vals*100,'ro')
xlabel('$\Delta$Az (degrees)')
ylabel('$f_{sky}$ ($\%$)')
subplot(2,1,2)
plot(alldaz, eta_vals,'ro')
xlabel('$\Delta$Az (degrees)')
ylabel('$\eta_{sky}$ ')

clf()
plot(alldaz, omega_vals / np.sqrt(fsky_vals) * eta_vals,'bo-')





