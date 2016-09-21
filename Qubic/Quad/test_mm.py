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


#### Get input Power spectra
################# Input Power spectrum ###################################
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
import pycamb
H0 = 67.04
omegab = 0.022032
omegac = 0.12038
h2 = (H0/100.)**2
scalar_amp = np.exp(3.098)/1.E10
omegav = h2 - omegab - omegac
Omegab = omegab/h2
Omegac = omegac/h2
print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',omegav/h2

params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':0.05,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}

lmaxcamb = 3*512
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
ell = np.arange(1,lmaxcamb+1)
fact = (ell*(ell+1))/(2*np.pi)
spectra = [ell, T/fact, E/fact, B/fact, X/fact]


clf()
plot(ell,np.sqrt(spectra[1]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TT}$')
plot(ell,np.sqrt(abs(spectra[4])*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TE}$')
plot(ell,np.sqrt(spectra[2]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{EE}$')
plot(ell,np.sqrt(spectra[3]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{BB} (r=0.1)$')
yscale('log')
xlim(0,600)
ylim(0.01,100)
xlabel('$\ell$')
ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
legend(loc='lower right',frameon=False)


### Input map
nside_in=256
mapi,mapq,mapu=hp.synfast(spectra[1:],nside_in,new=True, pixwin=True)
x0=np.array([mapi,mapq,mapu]).T


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
ts = 20.             # seconds
center = equ2gal(racenter, deccenter)

def plotinst(inst, color='r'):
  for xyc, quad in zip(inst.detector.center, inst.detector.quadrant): 
      plot(xyc[0],xyc[1],'o', color=color)
  xlim(-0.06, 0.06)

# some display
def display(input, msg, iplot=1, center=None, nlines=1, reso=5, lims=[50, 5, 5]):
    out = []
    for i, (kind, lim) in enumerate(zip('IQU', lims)):
        map = input[..., i]
        out += [hp.gnomview(map, rot=center, reso=reso, xsize=800, min=-lim,
                            max=lim, title=msg + ' ' + kind,
                            sub=(nlines, 3, iplot + i), return_projected_map=True)]
    return out


####### Create some sampling
center = equ2gal(racenter, deccenter)

npointings = 1000
dtheta = 15
#sampling = create_random_pointings([racenter, deccenter],npointings,dtheta)
sampling = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)
    


####### Full instrument
#instrument = QubicInstrument(filter_nu=150e9,
#                    detector_nep=4.7e-17*np.sqrt(len(sampling) * sampling.period / (365 * 86400)))
instrument = QubicInstrument(filter_nu=150e9,
                    detector_nep=2.7e-17)
clf()
plotinst(instrument)

### Prepare input / output data
nside = 256
scene = QubicScene(nside, kind='IQU')

def get_qubic_map(instrument, sampling, scene, input_maps, withplanck=True, covlim=0.1):
    acq = QubicAcquisition(instrument, sampling, scene, photon_noise=True, effective_duration=1)
    C = acq.get_convolution_peak_operator()
    coverage = acq.get_coverage()
    observed = coverage > covlim * np.max(coverage)
    acq_restricted = acq[:, :, observed]
    H = acq_restricted.get_operator()
    x0_convolved = C(input_maps)
    if not withplanck:
        pack = PackOperator(observed, broadcast='rightward')
        y_noiseless = H(pack(x0_convolved))
        noise = acq.get_noise()
        y = y_noiseless + noise
        invntt = acq.get_invntt_operator()
        A = H.T * invntt * H
        b = (H.T * invntt)(y)
        preconditioner = DiagonalOperator(1 / coverage[observed], broadcast='rightward')
        solution_qubic = pcg(A, b, M=preconditioner, disp=True, tol=1e-3, maxiter=1000)
        maps = pack.T(solution_qubic['x'])
        maps[~observed] = 0
    else:
        acq_planck = PlanckAcquisition(150, acq.scene, true_sky=x0_convolved)#, fix_seed=True)
        acq_fusion = QubicPlanckAcquisition(acq, acq_planck)
        map_planck_obs=acq_planck.get_observation()
        H = acq_fusion.get_operator()
        invntt = acq_fusion.get_invntt_operator()
        y = acq_fusion.get_observation()
        A = H.T * invntt * H
        b = H.T * invntt * y
        solution_fusion = pcg(A, b, disp=True, maxiter=1000, tol=1e-3)
        maps = solution_fusion['x']
        maps[~observed] = 0
    x0_convolved[~observed] = 0
    return(maps, x0_convolved, observed)    


omega = instrument.primary_beam.solid_angle
S = np.pi*instrument.horn.radius**2
lamb = 3e8/150e9
S*omega/lamb**2

#### Change transmissions
#instrument.optics.components[7][2] = 0.99 
#instrument.detector.efficiency=1

import time
t0 = time.time()
themap_0_QP, x0_convolved, observed = get_qubic_map(instrument, sampling, scene, x0, withplanck = True)
t1 = time.time()
themap_0_Q, x0_convolved, observed = get_qubic_map(instrument, sampling, scene, x0, withplanck = False)
t2 = time.time()


clf()
display(themap_0_Q,'Q 0',center=center, reso=10, lims=[200, 2,2], nlines =2, iplot=1)
display(themap_0_QP,'QP 0',center=center, reso=10, lims=[200, 2,2], nlines =2, iplot=4)

display(x0_convolved,'Q 0',center=center, reso=10, lims=[200, 2,2], nlines =2, iplot=1)

resQ = themap_0_Q-x0_convolved
resQP = themap_0_QP-x0_convolved


clf()
display(themap_0_Q-x0_convolved,'Res Q 0',center=center, reso=10, lims=[10, 1,1], nlines =2, iplot=1)
display(themap_0_QP-x0_convolved,'Res QP 0',center=center, reso=10, lims=[10, 1,1], nlines =2, iplot=4)

np.std(resQ[observed,:], axis=0)
np.std(resQP[observed,:], axis=0)


clf()
display(x0_convolved,'Input',center=center, reso=2.5, lims=[200, 3,3], nlines =3, iplot=1)
display(themap_0_QP,'Reconstructed',center=center, reso=2.5, lims=[200, 3,3], nlines =3, iplot=4)
display(themap_0_QP-x0_convolved,'Residuals',center=center, reso=2.5, lims=[3, 3,3], nlines =3, iplot=7)


#### Test running twice
instrument = QubicInstrument(filter_nu=150e9,
                    detector_nep=4.7e-17*np.sqrt(len(sampling) * sampling.period / (365 * 86400)))

np.random.seed(2)
themap_1_Q, x0_convolved, observed = get_qubic_map(instrument, sampling, scene, x0, withplanck = False)
hp.write_map('toto1.fits', (themap_0_Q - x0_convolved).T)
#themap_1_Q, x0_convolved, observed = get_qubic_map(instrument, sampling, scene, x0, withplanck = False)

resi0 = np.array(hp.read_map('toto0.fits',field=(0,1,2))).T
resi1 = np.array(hp.read_map('toto1.fits',field=(0,1,2))).T



#resi0 = themap_0_Q - x0_convolved
#resi1 = themap_1_Q - x0_convolved


clf()
display(resi0,'Res Q 0',center=center, reso=5, lims=[10, 1,1], nlines =3, iplot=1)
display(resi1,'Res QP 0',center=center, reso=5, lims=[10, 1,1], nlines =3, iplot=4)
display(resi1-resi0,'Res QP 0',center=center, reso=5, lims=[10, 1,1], nlines =3, iplot=7)

np.std(resi0[observed,:], axis=0)
np.std(resi1[observed,:], axis=0)
np.std(resi0[observed,:]-resi1[observed,:], axis=0)
np.std(resi0[observed,:]-resi1[observed,:], axis=0)/sqrt(2)


clf()
subplot(1,3,1)
plot(resi0[observed,0], resi1[observed,0],',')
subplot(1,3,2)
plot(resi0[observed,1], resi1[observed,1],',')
subplot(1,3,3)
plot(resi0[observed,2], resi1[observed,2],',')







sz = 512
x = linspace(0,1,sz)
xx = 








### test
scene = QubicScene(1024)
inst = instrument[231]
sb = inst.get_synthbeam(scene)

hp.mollview(np.log10(sb[0,:]))

clsb = hp.anafast(sb[0,:])

ell = np.arange(len(clsb))

plot(ell, clsb)
xlim(0,800)





