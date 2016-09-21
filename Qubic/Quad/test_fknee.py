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
ts = 24*3600/2**23            # seconds Chosen in order to have a power of 2 in 24 hours
np.random.seed(0)
center = equ2gal(racenter, deccenter)
covlim =0.1

####### Create some sampling
sampling = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)

    
### Prepare input / output data
nside = 64
scene = QubicScene(nside, kind='IQU')
####### Full instrument
instrument = QubicInstrument(filter_nu=150e9,
                    detector_nep=4.7e-17,
                    detector_fknee = 10.,
                    detector_fslope = 1.)

####### Only one detector
det = 231
instrument_one= instrument[231]


maps, x0_convolved, observed, y_noiseless, noise = mm.get_qubic_map(instrument_one, sampling, scene, input_maps, 
                                                                    withplanck=False, covlim=0.1, return_tod=True,
                                                                    photon_noise=False)


residuals = maps-x0_convolved
pxsize_arcmin2 = 4*pi*(180/pi)**2 / (12*nside**2) * 60**2
sigs= np.std(residuals[observed,:], axis=0)*sqrt(pxsize_arcmin2)
print(sigs)

clf()
mm.display(x0_convolved,msg='Init', center=center, nlines=3)
mm.display(maps,msg='Maps', center=center, nlines=3, iplot=4)
mm.display(residuals,msg='Residuals', center=center,nlines=3,iplot=7)


freq, ps = powspec_inst(ts, y_noiseless+noise)


clf()
subplot(2,1,1)
yscale('log')
xscale('log')
plot(freq, ps)
subplot(2,3,4)
hist(residuals[observed,0]*sqrt(pxsize_arcmin2), label='$\sigma={0:.1f} \mu K.arcmin$'.format(sigs[0]))
legend(fontsize=10, frameon=False, loc='upper left')
subplot(2,3,5)
hist(residuals[observed,1]*sqrt(pxsize_arcmin2), label='$\sigma={0:.1f} \mu K.arcmin$'.format(sigs[1]))
legend(fontsize=10, frameon=False, loc='upper left')
subplot(2,3,6)
hist(residuals[observed,2]*sqrt(pxsize_arcmin2), label='$\sigma={0:.1f} \mu K.arcmin$'.format(sigs[2]))
legend(fontsize=10, frameon=False, loc='upper left')







#### Now let's do a MC
real_duration = 365
ndetectors = 992

speedup = np.array([1])
fknee_vals = np.append(0,np.logspace(-1,1,num=3))
nbmc=1000
allsigs = np.zeros((len(speedup), len(fknee_vals), nbmc, 3))
for i in xrange(len(speedup)):
    s = speedup[i]
    sampling = create_sweeping_pointings([racenter, deccenter], 
                    duration, ts, angspeed*s, delta_az, nsweeps_el*s, angspeed_psi, maxpsi)
    for j in xrange(len(fknee_vals)):
        f = fknee_vals[j]
        instrument = QubicInstrument(filter_nu=150e9,
                            detector_nep=4.7e-17,
                            detector_fknee = f,
                            detector_fslope = 1.)

        #maps, x0_convolved, observed, y_noiseless, noise = mm.get_qubic_map(instrument_one, sampling, scene, input_maps, 
        #                                                            withplanck=False, covlim=0.1, return_tod=True)

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
        
        for nn in xrange(nbmc):
            noise = acq.get_noise()/np.sqrt(real_duration*ndetectors)
            y = y_noiseless + noise
            b = (H.T * invntt)(y)
            solution_qubic = pcg(A, b, M=preconditioner, disp=True, tol=1e-3, maxiter=1000)
            maps = pack.T(solution_qubic['x'])
            maps[~observed] = 0
            x0_convolved[~observed,:]=0    
        

            residuals = maps-x0_convolved
            freq, ps = powspec_inst(ts, y_noiseless+noise)

            pxsize_arcmin2 = 4*pi*(180/pi)**2 / (12*nside**2) * 60**2
            sigs= np.std(residuals[observed,:], axis=0)*sqrt(pxsize_arcmin2)
            print(s, f, nn, sigs)
            allsigs[i,j,nn,:] = sigs

            clf()
            subplot(2,1,1)
            yscale('log')
            xscale('log')
            plot(freq, ps)
            subplot(2,3,4)
            hist(residuals[observed,0]*sqrt(pxsize_arcmin2), label='$\sigma={0:.1f} \mu K.arcmin$'.format(sigs[0]))
            legend(fontsize=10, frameon=False, loc='upper left')
            subplot(2,3,5)
            hist(residuals[observed,1]*sqrt(pxsize_arcmin2), label='$\sigma={0:.1f} \mu K.arcmin$'.format(sigs[1]))
            legend(fontsize=10, frameon=False, loc='upper left')
            subplot(2,3,6)
            hist(residuals[observed,2]*sqrt(pxsize_arcmin2), label='$\sigma={0:.1f} \mu K.arcmin$'.format(sigs[2]))
            legend(fontsize=10, frameon=False, loc='upper left')
    



msigs = np.mean(allsigs, axis=2)
errsigs = np.std(allsigs, axis=2)/sqrt(nbmc)

clf()
for i in xrange(len(speedup)):
    errorbar(fknee_vals,msigs[i,:,0]/msigs[i,0,0], yerr=errsigs[i,:,0]/msigs[i,0,0],fmt='go', label='I')
    errorbar(fknee_vals/1.1,msigs[i,:,1]/msigs[i,0,1], yerr=errsigs[i,:,1]/msigs[i,0,1],fmt='ro', label='Q')
    errorbar(fknee_vals*1.1,msigs[i,:,2]/msigs[i,0,2], yerr=errsigs[i,:,2]/msigs[i,0,2],fmt='bo', label='U')
plot(fknee_vals*2, fknee_vals*0+1,'k:')
xlabel('$f_{knee}$ [Hz]')
ylabel('Maps RMS increase')
xscale('log')
xlim(0.01, 20)
ylim(0.99,1.01)
legend(numpoints=1)
title('Independent detectors - Independent noise beyond 24h')
savefig('increase_fknee_new.png')


clf()
for i in xrange(len(speedup)): 
    errorbar(fknee_vals,msigs[i,:,0], yerr=errsigs[i,:,0]*sqrt(nbmc),fmt='go', label='I')
    errorbar(fknee_vals/1.1,msigs[i,:,1], yerr=errsigs[i,:,1]*sqrt(nbmc),fmt='ro', label='Q')
    errorbar(fknee_vals*1.1,msigs[i,:,2], yerr=errsigs[i,:,2]*sqrt(nbmc),fmt='bo', label='U')
xlabel('$f_{knee}$ [Hz]')
ylabel('Maps RMS [$\mu K$.arcmin]')
xscale('log')
xlim(0.01, 20)
ylim(0,20)
legend(numpoints=1)
title('1 year - Independent detectors - Independent noise beyond 24h')
savefig('noise_fknee_new.png')




#### Fully correlated noise
#### Now let's do a MC
fknee_vals = np.append(0,np.logspace(-1,1,num=5))
allsigs2 = np.zeros((len(fknee_vals), 3))
sampling = create_sweeping_pointings([racenter, deccenter], 
                duration, ts, angspeed, delta_az, nsweeps_el, angspeed_psi, maxpsi)
for j in xrange(len(fknee_vals)):
    f = fknee_vals[j]
    instrument = QubicInstrument(filter_nu=150e9,
                        detector_nep=4.7e-17*np.sqrt(len(sampling) * sampling.period / (365 * 86400)),
                        detector_fknee = f,
                        detector_fslope = 1.)

    acq = QubicAcquisition(instrument, sampling, scene)
    C = acq.get_convolution_peak_operator()
    coverage = acq.get_coverage()
    observed = coverage > covlim * np.max(coverage)
    acq_restricted = acq[:, :, observed]
    H = acq_restricted.get_operator()
    x0_convolved = C(input_maps)

    pack = PackOperator(observed, broadcast='rightward')
    y_noiseless = H(pack(x0_convolved))
    noise = acq.get_noise()
    y = y_noiseless + noise[0,:]
    invntt = acq.get_invntt_operator()
    A = H.T * invntt * H
    b = (H.T * invntt)(y)
    preconditioner = DiagonalOperator(1 / coverage[observed], broadcast='rightward')
    solution_qubic = pcg(A, b, M=preconditioner, disp=True, tol=1e-3, maxiter=1000)
    maps = pack.T(solution_qubic['x'])
    maps[~observed] = 0

    residuals = maps-x0_convolved

    psy = np.abs(fft(noise, axis=1))**2
    avpsy = np.mean(psy, axis=0)
    freq = fftfreq(len(avpsy), ts)
    mask = freq>0

    pxsize_arcmin2 = 4*pi*(180/pi)**2 / (12*nside**2) * 60**2
    sigs= np.std(residuals[observed,:], axis=0)*sqrt(pxsize_arcmin2)
    print(f, sigs)
    allsigs2[j,:] = sigs

    clf()
    subplot(2,1,1)
    yscale('log')
    xscale('log')
    plot(freq[mask], avpsy[mask])
    subplot(2,3,4)
    hist(residuals[observed,0]*sqrt(pxsize_arcmin2), label='$\sigma={0:.1f} \mu K.arcmin$'.format(sigs[0]))
    legend(fontsize=10, frameon=False, loc='upper left')
    subplot(2,3,5)
    hist(residuals[observed,1]*sqrt(pxsize_arcmin2), label='$\sigma={0:.1f} \mu K.arcmin$'.format(sigs[1]))
    legend(fontsize=10, frameon=False, loc='upper left')
    subplot(2,3,6)
    hist(residuals[observed,2]*sqrt(pxsize_arcmin2), label='$\sigma={0:.1f} \mu K.arcmin$'.format(sigs[2]))
    legend(fontsize=10, frameon=False, loc='upper left')
    



clf()
plot(fknee_vals,allsigs2[:,1]/allsigs2[0,1],'o-', label='Corr')
plot(fknee_vals,allsigs[0,:,1]/allsigs[0,0,1],'o-', label='Uncorr')
xlabel('$f_{knee}$ [Hz]')
ylabel('Q Maps RMS increase')
xscale('log')
title('Fully correlated noise across pixels')
legend()
#savefig('Qincrease_fknee_raw.png')
















##### Now with filtering
specfilt = [spectra[0]]
maskell = spectra[0]>50
for i in xrange(1,len(spectra)):
    thespec = spectra[i].copy()
    thespec[~maskell]=0
    specfilt.append(thespec)

clf()
plot(ell,np.sqrt(specfilt[1]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TT}$')
plot(ell,np.sqrt(abs(specfilt[4])*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TE}$')
plot(ell,np.sqrt(specfilt[2]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{EE}$')
plot(ell,np.sqrt(specfilt[3]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{BB} (r=0.1)$')
yscale('log')
xlim(0,300)
ylim(0.01,100)
xlabel('$\ell$')
ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
legend(loc='lower right',frameon=False)


def highpass_tod(tod, ts, fcut):
    ft = np.fft.fft(tod, axis=1)
    freq = np.fft.fftfreq(len(tod[0,:]), ts)
    mask = np.abs(freq) < fcut
    ft[:,mask]=0
    newtod = np.real(np.fft.ifft(ft))
    return(newtod)

nside_in=64
mapi,mapq,mapu=hp.synfast(spectra[1:],nside_in,new=True)
input_maps=np.array([mapi,mapq,mapu]).T

mapi,mapq,mapu=hp.synfast(specfilt[1:],nside_in,new=True)
input_maps_filt=np.array([mapi,mapq,mapu]).T

fcutvals = np.append(0,  np.logspace(-5,-1,num=5))

f=1.
instrument = QubicInstrument(filter_nu=150e9,
                    detector_nep=4.7e-17*np.sqrt(len(sampling) * sampling.period / (365 * 86400)),
                    detector_fknee = f,
                    detector_fslope = 1.)

covlim=0.1
acq = QubicAcquisition(instrument, sampling, scene)
C = acq.get_convolution_peak_operator()
coverage = acq.get_coverage()
observed = coverage > covlim * np.max(coverage)
acq_restricted = acq[:, :, observed]
H = acq_restricted.get_operator()
pack = PackOperator(observed, broadcast='rightward')
invntt = acq.get_invntt_operator()
A = H.T * invntt * H

x0_convolved = C(input_maps)
x0cc = x0_convolved.copy()
x0cc[~observed]=0
y_noiseless = H(pack(x0_convolved))

### Adding noise
noise = acq.get_noise()
y = y_noiseless + noise

nsamples = len(noise[0,:])
freq = fftfreq(nsamples, ts)
mask = freq>0

clf()
yscale('log')
xscale('log')
plot(freq[mask], np.mean(np.abs(fft(y_noiseless, axis=1))**2, axis=0)[mask])
plot(freq[mask], np.mean(np.abs(fft(noise, axis=1))**2, axis=0)[mask])
plot(freq[mask], np.mean(np.abs(fft(y, axis=1))**2, axis=0)[mask])

b = (H.T * invntt)(y_noiseless)
preconditioner = DiagonalOperator(1 / coverage[observed], broadcast='rightward')
solution_qubic = pcg(A, b, M=preconditioner, disp=True, tol=1e-3, maxiter=1000)
maps_noiseless = pack.T(solution_qubic['x'])
maps_noiseless[~observed] = 0

clf()
display(maps_noiseless, '', center=center)

b = (H.T * invntt)(y)
preconditioner = DiagonalOperator(1 / coverage[observed], broadcast='rightward')
solution_qubic = pcg(A, b, M=preconditioner, disp=True, tol=1e-3, maxiter=1000)
maps = pack.T(solution_qubic['x'])
maps[~observed] = 0

clf()
display(maps, '', center=center,nlines=2)
display(maps-maps_noiseless, 'resid ', center=center,nlines=2, iplot=4)

y_noiseless_filt = highpass_tod(y_noiseless, ts, 0.01)
b = (H.T * invntt)(y_noiseless_filt)
preconditioner = DiagonalOperator(1 / coverage[observed], broadcast='rightward')
solution_qubic = pcg(A, b, M=preconditioner, disp=True, tol=1e-3, maxiter=1000)
maps_noiseless_filt = pack.T(solution_qubic['x'])
maps_noiseless_filt[~observed] = 0

y_filt = highpass_tod(y, ts, 0.01)
b = (H.T * invntt)(y_filt)
preconditioner = DiagonalOperator(1 / coverage[observed], broadcast='rightward')
solution_qubic = pcg(A, b, M=preconditioner, disp=True, tol=1e-3, maxiter=1000)
mapsfilt = pack.T(solution_qubic['x'])
mapsfilt[~observed] = 0

clf()
display(maps-maps_noiseless, 'resid ', center=center,nlines=2)
display(mapsfilt-maps_noiseless_filt, 'resid filt ', center=center,nlines=2, iplot=4)

resid = (maps-maps_noiseless)
resid_filt = (mapsfilt-maps_noiseless_filt)
np.std(resid_filt[observed,1], axis=0) / np.std(resid[observed,1], axis=0)

clf()
display(maps_noiseless, 'noiseless ', center=center,nlines=2)
display(maps_noiseless_filt, 'noiseless filt ', center=center,nlines=2, iplot=4)
ratio = maps_noiseless_filt[observed,1]/ maps_noiseless[observed,1]
avratio = np.mean(ratio[np.abs(ratio)<10], axis=0)



### small sim
fcut_vals = np.append(0,np.logspace(-4,-2,num=5))
noise_ratio = []
signal_ratio = []
for fcut in fcut_vals:
    y_noiseless_filt = highpass_tod(y_noiseless, ts, fcut)
    b = (H.T * invntt)(y_noiseless_filt)
    preconditioner = DiagonalOperator(1 / coverage[observed], broadcast='rightward')
    solution_qubic = pcg(A, b, M=preconditioner, disp=True, tol=1e-3, maxiter=1000)
    maps_noiseless_filt = pack.T(solution_qubic['x'])
    maps_noiseless_filt[~observed] = 0
    
    y_filt = highpass_tod(y, ts, fcut)
    b = (H.T * invntt)(y_filt)
    preconditioner = DiagonalOperator(1 / coverage[observed], broadcast='rightward')
    solution_qubic = pcg(A, b, M=preconditioner, disp=True, tol=1e-3, maxiter=1000)
    mapsfilt = pack.T(solution_qubic['x'])
    mapsfilt[~observed] = 0
    
    resid = (maps-maps_noiseless)
    resid_filt = (mapsfilt-maps_noiseless_filt)
    noise_ratio.append(np.std(resid_filt[observed,1], axis=0) / np.std(resid[observed,1], axis=0))
    ratio = maps_noiseless_filt[observed,1]/ maps_noiseless[observed,1]
    signal_ratio.append(np.mean(ratio[np.abs(ratio)<10], axis=0))
    

clf()
plot(fcut_vals, np.array(signal_ratio)) 
plot(fcut_vals, np.array(noise_ratio))
xscale('log')
    
clf()
plot(fcut_vals, np.array(signal_ratio) / np.array(noise_ratio)) 
xscale('log')










