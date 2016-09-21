from __future__ import division
from pyoperators import pcg
from pysimulators import profile
from qubic import (
    create_random_pointings, equ2gal, QubicAcquisition, PlanckAcquisition,
    QubicPlanckAcquisition, QubicScene, QubicInstrument)
from qubic.data import PATH
from qubic.io import read_map
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np

racenter = 0.0      # deg
deccenter = -57.0   # deg
center = equ2gal(racenter, deccenter)
maxiter = 1000

nside = 1024

sampling = create_random_pointings([racenter, deccenter], 5000, 5)
scene = QubicScene(nside, kind='I')

#sky = read_map(PATH + 'syn256_pol.fits')[:,0]
sky = np.zeros(12*nside**2)
ip0=hp.ang2pix(nside, np.radians(90.-center[1]), np.radians(center[0]))
v0 = np.array(hp.pix2vec(nside, ip0))
sky[ip0] = 1000
#hp.gnomview(sky, rot=center, reso=5, xsize=800)

instrument = QubicInstrument(filter_nu=150e9,
                            detector_nep=1e-30)

acq_qubic = QubicAcquisition(instrument, sampling, scene, effective_duration=1, photon_noise=False)

coverage = acq_qubic.get_coverage()
observed = coverage > 0.01 * np.max(coverage)
H = acq_qubic.get_operator()
invntt = acq_qubic.get_invntt_operator()
y, sky_convolved = acq_qubic.get_observation(sky, convolution=True)

A = H.T * invntt * H
b = H.T * invntt * y

tol = 5e-6
solution_qubic = pcg(A, b, disp=True, maxiter=maxiter, tol=tol)



################ With only one detector
sampling = create_random_pointings([racenter, deccenter], 500000, 15)

instrument = QubicInstrument(filter_nu=150e9,
                            detector_nep=1e-30)
instrument = instrument[0]
instrument.detector.center=np.array([[0.,0.,-0.3]])

acq_qubic = QubicAcquisition(instrument, sampling, scene, effective_duration=1, photon_noise=False)

coverage = acq_qubic.get_coverage()
observed = coverage > 0.01 * np.max(coverage)
H = acq_qubic.get_operator()
invntt = acq_qubic.get_invntt_operator()
y, sky_convolved = acq_qubic.get_observation(sky, convolution=True)

A = H.T * invntt * H
b = H.T * invntt * y

tol = 5e-6
solution_qubic = pcg(A, b, disp=True, maxiter=maxiter, tol=tol)
###############################



themap = solution_qubic['x']

hp.gnomview(themap,reso=5, rot=center)



#out = hp.gnomview(solution_qubic['x'], rot=center, reso=5, xsize=800, return_projected_map=True)
#out = hp.gnomview(np.log(solution_qubic['x']), rot=center, reso=5, xsize=800, return_projected_map=True)

#### profile de la source reconstruite
vecs = np.array(hp.pix2vec(nside, np.arange(12*nside**2)[observed]))
cosang = np.dot(v0, vecs)
ang = np.degrees(np.arccos(cosang))
order =argsort(ang)
theang = ang[order]
theprofile = solution_qubic['x'][observed][order]

clf()
plot(theang,10*np.log10(np.abs(theprofile)))


####
sb = instrument.get_synthbeam(scene, idet=231)
vecs = np.array(hp.pix2vec(nside, np.arange(12*nside**2)[sb != 0]))
vecinit = np.array(hp.pix2vec(nside, [0]))
cosang = np.dot(vecinit.T, vecs)
ang = np.degrees(np.arccos(cosang))[0,:]
order = argsort(ang)
newang = ang[order]
newsb = sb[order]
clf()
plot(newang, newsb)

from SynthBeam import MultiQubicInstrument
nu_cent = 150e9  # the central frequency of the bandwidth
theta_max = 30
syb_f = 0.99
NFREQS = 1  # monochromatic
NPOINTS = 1 # no size of detectors
q = MultiQubicInstrument.MultiQubicInstrument(filter_name=nu_cent, synthbeam_fraction=syb_f, NFREQS=NFREQS, NPOINTS=NPOINTS)
scene2 = QubicScene(nside)
sb_ga = q.direct_convolution(scene2, 231, theta_max)  # the approximated one

sz=1000
res = 1.2
ga = hp.gnomview(
    sb_ga, rot=[45, 90.7], reso=res, xsize=sz, min=0, 
    return_projected_map=True, title='Gaussian approximation',
    margins=4 * [0.01])
i, j = np.unravel_index(np.argmax(ga), ga.shape)
x = np.arange(sz) * res / 60
x -= x[j]


mainbeam = theang < 1
np.interp(0.5, theprofile[mainbeam]/theprofile[0], theang[mainbeam])
clf()
plot(theang[mainbeam],theprofile[mainbeam]/theprofile[0],',')


clf()
plot(theang, theprofile/theprofile[0])

clf()
plot(theang, 10*np.log10(np.abs(theprofile/theprofile[0])),label='Point source profile (on reconstructed map)',lw=3)
#plot(x*sqrt(2), 10*np.log10(np.diag(ga)/np.max(ga)),label='Input Synthesized Beam (on TOD)',lw=3)
plot(x, 10*np.log10(ga[i]/np.max(ga[i])),label='Input Synthesized Beam (on TOD)',lw=3)
xlim(0,10)
ylim(-60,10)
xlabel('Angle [deg.]')
ylabel('Profile [dB]')
legend(frameon=False)
savefig('ptsrc_log.png')
