from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
from Quad import qml
from Quad import pyquad
from Quad import mapmake_jc_lib as mm
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
import scipy.ndimage.filters as filt

from MultiThread import multi_process as mtp

def powspec_inst(ts, tod):
    ps = np.abs(fft(tod, axis=1))**2
    avps = np.mean(ps, axis=0)
    freq = fftfreq(len(avps), ts)
    mask = freq>0
    return(freq[mask], avps[mask])
    



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
nside_in=64
mapi,mapq,mapu=hp.synfast(spectra[1:],nside_in,new=True)
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
npow = 23
ts = 24*3600/2**npow            # seconds Chosen in order to have a power of 2 in 24 hours
np.random.seed(0)
center = equ2gal(racenter, deccenter)
covlim =0.1


real_duration = 365*2
ndetectors = 992

### Prepare input / output data
nside = 64
scene = QubicScene(nside, kind='IQU')

sampling = create_sweeping_pointings([racenter, deccenter], 
            duration, ts, angspeed, delta_az, nsweeps_el, angspeed_psi, maxpsi)
            
            
speedup = np.array([1])
#fknee_vals = np.append(0,np.logspace(-1,1,num=5))
fknee_vals = [10.]
ncorrvals = [1, 10, 100, 100000]
nbmc=12
allsigs = np.zeros((len(speedup), len(fknee_vals), len(ncorrvals), nbmc, 3))

            

def highpass_tod(tod, ts, fcut):
    ft = np.fft.fft(tod, axis=1)
    freq = np.fft.fftfreq(len(tod[0,:]), ts)
    mask = np.abs(freq) < fcut
    ft[:,mask]=0
    newtod = np.real(np.fft.ifft(ft))
    return(newtod)

def toloop(arguments):
    print(arguments)
    print('Getting noise')
    noise = acq.get_noise()/np.sqrt(real_duration*ndetectors)
    print('noise done')
    #noise = highpass_tod(noise, ts, 0.05)
    y = y_noiseless + noise
    b = (H.T * invntt)(y)
    solution_qubic = pcg(A, b, M=preconditioner, disp=True, tol=1e-3, maxiter=1000)
    maps = pack.T(solution_qubic['x'])
    maps[~observed] = 0
    x0_convolved[~observed,:]=0    
    residuals = maps-x0_convolved
    pxsize_arcmin2 = 4*pi*(180/pi)**2 / (12*nside**2) * 60**2
    sigs= np.std(residuals[observed,:], axis=0)*sqrt(pxsize_arcmin2)
    return(sigs)


for i in xrange(len(speedup)):
    s = speedup[i]
    sampling = create_sweeping_pointings([racenter, deccenter], 
                duration, ts, angspeed*s, delta_az, nsweeps_el*s, angspeed_psi, maxpsi)
    for j in xrange(len(fknee_vals)):
        f = fknee_vals[j]
        ####### Full instrument
        for incv in xrange(len(ncorrvals)):
            fslope =1
            signoise = 4.7e-17
            instrument = QubicInstrument(filter_nu=150e9,
                                detector_nep=signoise,
                                detector_fknee = f,
                                detector_fslope = fslope,
                                detector_ncorr = ncorrvals[incv])

            ####### Only one detector
            det = 231
            instrument_one= instrument[231]

            acq = QubicAcquisition(instrument_one, sampling, scene, photon_noise=False)
            C = acq.get_convolution_peak_operator()
            coverage = acq.get_coverage()
            observed = coverage > covlim * np.max(coverage)
            acq_restricted = acq[:, :, observed]
            H = acq_restricted.get_operator()
            x0_convolved = C(input_maps)
            pack = PackOperator(observed, broadcast='rightward')
            y_noiseless = H(pack(x0_convolved))
            invntt = acq.get_invntt_operator()
            A = H.T * invntt * H
            preconditioner = DiagonalOperator(1 / coverage[observed], broadcast='rightward')

            # Parallel loop
            sigsmc = mtp.parallel_for(toloop, np.arange(nbmc),nprocs=6)

            allsigs[i,j,incv,:,:]=np.array(sigsmc)


msigs = np.mean(allsigs, axis=3)
errsigs = np.std(allsigs, axis=3)






clf()
errorbar(fknee_vals,msigs[0,:,1]/msigs[0,0,1], yerr=errsigs[0,:,1]/msigs[0,0,1],fmt='o-', label='Q')
errorbar(fknee_vals,msigs[0,:,2]/msigs[0,0,2], yerr=errsigs[0,:,2]/msigs[0,0,2],fmt='o-', label='U')
xlabel('$f_{knee}$ [Hz]')
ylabel('Maps RMS increase')
xscale('log')
xlim(0.05, 20)
ylim(0,3)
grid()
legend(loc='upper left', numpoints=1)
#title('2 Years - independent detectors - Max correlations 24h - Not Filtered')
#savefig('QUratio_increase_fknee_unfiltered.png')
title('2 Years - independent detectors - Max correlations 24h - Filtered')
#savefig('QUratio_increase_fknee_filtered.png')


clf()
for i in xrange(len(speedup)): 
    errorbar(fknee_vals,msigs[i,:,1], yerr=errsigs[i,:,1],fmt='o-', label=speedup[i])
    #errorbar(fknee_vals,msigs_18[i,:,1], yerr=errsigs[i,:,1],fmt='o:', label=speedup[i])
    #errorbar(fknee_vals,msigs_21[i,:,1], yerr=errsigs[i,:,1],fmt='o--', label=speedup[i])
xlabel('$f_{knee}$ [Hz]')
ylabel('Q Maps RMS [$\mu K$.arcmin]')
xscale('log')
xlim(0.05, 20)
legend()

clf()
errorbar(fknee_vals,msigs[0,:,1], yerr=errsigs[0,:,1],fmt='o-', label='Q')
errorbar(fknee_vals,msigs[0,:,2], yerr=errsigs[0,:,2],fmt='o-', label='U')
xlabel('$f_{knee}$ [Hz]')
ylabel('Q Maps RMS [$\mu K$.arcmin]')
xscale('log')
xlim(0.05, 20)
ylim(0,30)
grid()
legend(loc='upper left', numpoints=1)
title('2 Years - independent detectors - Max correlations 24h - Filtered')
savefig('QUincrease_fknee_filtered.png')




