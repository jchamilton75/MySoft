from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
from Quad import qml
from Quad import pyquad
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
ts = 5.             # seconds
np.random.seed(0)
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
sampling = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)
    
####### Full instrument
instrument = QubicInstrument(filter_nu=150e9,
                    detector_nep=1e-4*4.7e-17*np.sqrt(len(sampling) * sampling.period / (365 * 86400)))
clf()
plotinst(instrument)

###### Two sub-instrument with each half of the bolometers
numbols = np.arange(len(instrument.detector.index))
mask2 = (numbols % 2) == 0
instEven = instrument[mask2]
instOdd = instrument[~mask2]
clf()
plotinst(instEven, color='r')
plotinst(instOdd, color='b')

### Prepare input / output data
nside = 64
scene = QubicScene(nside, kind='IQU')

def get_qubic_map(instrument, sampling, scene, input_maps, withplanck=True, covlim=0.1):
    acq = QubicAcquisition(instrument, sampling, scene, photon_noise=False)
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
        print(std(noise))
        invntt = acq.get_invntt_operator()
        A = H.T * invntt * H
        b = (H.T * invntt)(y)
        preconditioner = DiagonalOperator(1 / coverage[observed], broadcast='rightward')
        solution_qubic = pcg(A, b, M=preconditioner, disp=True, tol=1e-3, maxiter=1000)
        maps = pack.T(solution_qubic['x'])
        maps[~observed] = 0
    else:
        acq_planck = PlanckAcquisition(150, acq.scene, true_sky=x0_convolved, fix_seed=True)
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
    x0_convolved[~observed,:]=0    
    return(maps, x0_convolved, observed)    

themap_0_Q, x0_convolved, observed = get_qubic_map(instEven, sampling, scene, x0, withplanck = False)

clf()
display(themap_0_Q,'Q 0',center=center, reso=3, lims=[200, 2,2], nlines =3, iplot=1)
display(x0_convolved,'Init',center=center, reso=3, lims=[200, 2,2], nlines =3, iplot=4)
display(themap_0_Q-x0_convolved,'Residuals',center=center, reso=3, lims=[200, 2,2], nlines =3, iplot=7)


themap_1_Q, x0_convolved, obesrved = get_qubic_map(instOdd, sampling, scene, x0, withplanck = False)

themap_0_QP, x0_convolved, observed = get_qubic_map(instEven, sampling, scene, x0)
themap_1_QP, x0_convolved, observed = get_qubic_map(instOdd, sampling, scene, x0)
x0_convolved[~observed,:]=0    

clf()
display(themap_0_Q,'Q 0',center=center, reso=3, lims=[200, 2,2], nlines =4, iplot=1)
display(themap_1_Q,'Q 1',center=center, reso=3, lims=[200, 2,2], nlines =4, iplot=4)
display(themap_0_QP,'QP 0',center=center, reso=3, lims=[200, 2,2], nlines =4, iplot=7)
display(themap_1_QP,'QP 1',center=center, reso=3, lims=[200, 2,2], nlines =4, iplot=10)

clf()
display(themap_0_Q-x0_convolved,'Res Q 0',center=center, reso=3, lims=[2, 0.1,0.1], nlines =4, iplot=1)
display(themap_1_Q-x0_convolved,'Res Q 1',center=center, reso=3, lims=[2, 0.1,0.1], nlines =4, iplot=4)
display(themap_0_QP-x0_convolved,'Res QP 0',center=center, reso=3, lims=[2, 0.1,0.1], nlines =4, iplot=7)
display(themap_1_QP-x0_convolved,'Res QP 1',center=center, reso=3, lims=[2, 0.1,0.1], nlines =4, iplot=10)

#### EB Only
mask=~observed
maskok=observed

ip=np.arange(12*nside**2)
ipok=ip[~mask]

#ellbins=[0,50,100,150,200,250,300,350,3*nside]
ellbins = [0,50,75,100,125,150,175,200]
nbins=len(ellbins)-1
ellmin=np.array(ellbins[0:nbins])
ellmax=np.array(ellbins[1:nbins+1])-1
ellval=(ellmin+ellmax)/2
binspec=pyquad.binspectrum(spectra,ellmin,ellmax)

reload(qml)
fwhmrad = instrument.synthbeam.peak150.fwhm
ll=np.arange(int(np.max(ellbins))+1)
bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)

#ds_dcb=qml.compute_ds_dcb_parpix_direct(ellbins,nside,ipok,bl,polar=True,temp=False,nprocs=8)


mapq_o = x0_convolved[:,1]
mapq_o[mask]=0
mapu_o = x0_convolved[:,2]
mapu_o[mask]=0
themaps_o = [mapq_o,mapu_o]
themaps2_o = [mapq_o,mapu_o]


### Likelihood
covmap=np.zeros(len(ipok)*2)
guess=[0,binspec[:,1],binspec[:,2],binspec[:,3],0]

specoutQ,errorQ,invfisherQ,ds_dcb=qml.qml_cross_noiter([themap_0_Q[:,1],themap_0_Q[:,2]] , 
    [themap_1_Q[:,1], themap_1_Q[:,2]],mask,covmap,ellbins,fwhmrad,guess,
    ds_dcb,spectra,cholesky=False,temp=False,polar=True,plot=True)

specoutQP,errorQP,invfisherQP,ds_dcb=qml.qml_cross_noiter([themap_0_QP[:,1],themap_0_QP[:,2]] , 
    [themap_1_QP[:,1], themap_1_QP[:,2]],mask,covmap,ellbins,fwhmrad,guess,
    ds_dcb,spectra,cholesky=False,temp=False,polar=True,plot=True)

specout_o,error_o,invfisher_o,ds_dcb=qml.qml_cross_noiter(themaps_o, themaps2_o,mask,covmap,ellbins,fwhmrad,guess,
    ds_dcb,spectra,cholesky=False,temp=False,polar=True,plot=True)


clf()
subplot(1,2,1)
xlim(0,np.max(ellmax)*1.2)
plot(ell,E,lw=3)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')
ylim(0,3)
plot(ellval,specoutQ[2]*ellval*(ellval+1)/(2*np.pi), 'ro',label='QUBIC')
plot(ellval,specoutQP[2]*ellval*(ellval+1)/(2*np.pi), 'go',label='QUBIC+Planck')
plot(ellval,specout_o[2]*ellval*(ellval+1)/(2*np.pi), 'ko',label='Original')
legend(numpoints=1)
subplot(1,2,2)
xlim(0,np.max(ellmax)*1.2)
plot(ell,B,lw=3)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')
plot(ellval,specoutQ[3]*ellval*(ellval+1)/(2*np.pi), 'ro')
plot(ellval,specoutQP[3]*ellval*(ellval+1)/(2*np.pi), 'go')
plot(ellval,specout_o[3]*ellval*(ellval+1)/(2*np.pi), 'ko')
ylim(0,0.03)


clf()
plot(ellval, specoutQ[2]/specout_o[2], 'bo-')
plot(ellval, specoutQ[3]/specout_o[3], 'ro-')
plot(ellval, specoutQP[2]/specout_o[2], 'bo--')
plot(ellval, specoutQP[3]/specout_o[3], 'ro--')
ylim([-1,3])




