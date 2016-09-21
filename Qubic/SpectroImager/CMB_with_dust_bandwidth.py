from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
from Quad import qml
from Quad import pyquad
import healpy as hp
from pyoperators import MPI, BlockDiagonalOperator, BlockRowOperator,BlockColumnOperator, DiagonalOperator, PackOperator, pcg , asoperator , MaskOperator
from pysimulators.interfaces.healpy import (HealpixConvolutionGaussianOperator)
from qubic import (
    QubicAcquisition, QubicInstrument,
    QubicScene, create_sweeping_pointings, equ2gal, create_random_pointings, PlanckAcquisition, QubicPlanckAcquisition)
#from MYacquisition import PlanckAcquisition, QubicPlanckAcquisition
from qubic.io import read_map, write_map
import gc
import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)
import sys
import os
from qubic.data import PATH
from functools import reduce
import pycamb

######################################################
### Cosmological parameters and CMB power spectrum ###
######################################################

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
T[0:50]=0
ell = np.arange(1,lmaxcamb+1)
fact = (ell*(ell+1))/(2*np.pi)
spectra = [ell, T/fact, E/fact, B/fact, X/fact]


#############
### Scene ###
#############

nside = 256
Nbpixels = 12*nside**2
scene = QubicScene(nside, kind='IQU')


################
### Sampling ###
################

maxiter = 1000
tol = 5e-6
racenter = 0.0      # deg
deccenter = -57.0   # deg
angspeed = 1        # deg/sec
delta_az = 20.      # deg
angspeed_psi = 0.1  # deg/sec
maxpsi = 45.        # deg
nsweeps_el = 300
duration = 24      # hours
ts = 100.             # seconds
center = equ2gal(racenter, deccenter)
sampling = create_sweeping_pointings(
[racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
angspeed_psi, maxpsi)

###################
### Frequencies ###
###################

nus=[150,220] ## central frequencies of bands to reconstruct
sub_nus=[134,142,150,158,166,196,208,220,232,244] ## central frequencies of subbands
band1=np.arange(5) # index for first band
band2=np.arange(5,10) #index for second band
bands=[band1,band2] #index
sub_deltas=np.array([8,8,8,8,8,12,12,12,12,12]) #delta_nu for integration for TOD
deltas=np.array([40,60]) #total bandwidth of all bands
Nbfreq=len(nus)
Nbsubfreq=len(sub_nus)

################
### Coverage ###
################
instruments = [QubicInstrument(filter_nu=nus[i]*10**9,detector_nep=2.7e-17) for i in range(Nbfreq)]
acqs = [QubicAcquisition(instruments[i], sampling, scene,photon_noise=True, effective_duration=10) for i in range(Nbfreq)]
covlim=0.1
coverage = np.array([acqs[i].get_coverage() for i in range(len(nus))])
observed = [(coverage[i] > covlim*np.max(coverage[i])) for i in range(Nbfreq)]
obs=reduce(logical_and,tuple(observed[i] for i in range(Nbfreq)))
pack = PackOperator(obs, broadcast='rightward')

###############
### Cmb map ###
###############

mapI,mapQ,mapU=hp.synfast(spectra[1:],nside,new=True, pixwin=True)
cmb=np.array([mapI,mapQ,mapU]).T

#########################################
### Dust power spectrum, map, scaling ###
#########################################

coef=1.39e-2
spectra_dust = [ell, np.zeros(len(ell)), coef*(ell/80)**(-0.42)/(fact*0.52), coef*(ell/80)**(-0.42)/fact, np.zeros(len(ell))]
dustT,dustQ,dustU= hp.synfast(spectra_dust[1:],nside,new=True, pixwin=True)
## dust map at 150GHz :
dust=np.array([dustT,dustQ,dustU]).T
###Frequency scaling for dust emission
def scaling_dust(freq1,freq2, sp_index=1.59): ## frequencies are in GHz
    freq1=float(freq1)
    freq2=float(freq2)
    x1=freq1/56.78
    x2=freq2/56.78
    S1=x1**2.*np.exp(x1)/(np.exp(x1)-1)**2.
    S2=x2**2.*np.exp(x2)/(np.exp(x2)-1)**2.
    vd=375.06/18.*19.6
    scaling_factor_dust=(np.exp(freq1/vd)-1)/(np.exp(freq2/vd)-1)*(freq2/freq1)**(sp_index+1)
    scaling_factor_termo=S1/S2*scaling_factor_dust
    return scaling_factor_termo

#################
### Input map ###
#################

x0=np.zeros((Nbsubfreq,Nbpixels,3))
for i in range(Nbsubfreq):
    #x0[i,:,0]=mapI+dustT*scaling_dust(150,sub_nus[i]e-9,1.59)
    x0[i,:,1]=mapQ+dustQ*scaling_dust(150,sub_nus[i],1.59)
    x0[i,:,2]=mapU+dustU*scaling_dust(150,sub_nus[i],1.59)

###################################################################################
### Convolution of the input map (only for comparison to the reconstructed map) ###
###################################################################################

sub_instruments=[QubicInstrument(filter_nu=sub_nus[i]*10**9,detector_nep=2.7e-17) for i in range(Nbsubfreq)]
sub_fwhms=np.array([sub_instruments[i].synthbeam.peak150.fwhm * (150 / (sub_nus[i])) for i in range(Nbsubfreq)])
C=np.array([BlockDiagonalOperator([HealpixConvolutionGaussianOperator(fwhm=fwhm) for fwhm in sub_fwhms[bands[i]]],new_axisin=0) for i in range(Nbfreq)])
x0_convolved=np.zeros((Nbfreq,Nbpixels,3))
x0_convolved=np.mean([C[i](x0[bands[i]]) for i in range(Nbfreq)],axis=1)

###############################
### Construction of the TOD ###
###############################

sub_acqs=[QubicAcquisition(sub_instruments[i], sampling, scene,photon_noise=True, effective_duration=10) for i in range(Nbsubfreq)]
sub_acqs_restricted=[sub_acqs[i][:,:,obs] for i in range(Nbsubfreq)]
operators=np.array([sub_acqs_restricted[i].get_operator() for i in range(Nbsubfreq)])
sub_fwhms_transfer = [np.sqrt(fwhm**2 - sub_fwhms[-1]**2) for fwhm in sub_fwhms] #Transfer fwhm from max resolution to lower resolution
T=np.array([HealpixConvolutionGaussianOperator(fwhm=fwhm) for fwhm in sub_fwhms_transfer]) # Transfer operator from maximum resolution to lower resolution
Convol=HealpixConvolutionGaussianOperator(fwhm=sub_fwhms[-1]) #Convolution by the maximum resolution kernel
noise=acqs[0].get_noise()*(np.sum(deltas))*10**9/(Nbfreq)
Y=np.sum([operators[i]*pack*T[i]*Convol*sub_deltas[i]*10**9*x0[i] for i in range(Nbsubfreq)],axis=0)+noise # Total TOD

######################
### Reconstruction ###
######################

K=np.array([np.sum([operators[j]*sub_deltas[j]*10**9 for j in bands[i]],axis=0) for i in range(Nbfreq)])
H=BlockRowOperator([K[i] for i in range(Nbfreq)], new_axisin=0)
invntt = acqs[0].get_invntt_operator()
A=H.T*invntt*H
b = (H.T * invntt)(Y)
preconditioner = BlockDiagonalOperator([DiagonalOperator(1 / coverage[0][obs], broadcast='rightward') for i in range(Nbfreq)], new_axisin=0) 
solution_qubic = pcg(A, b, M=preconditioner ,disp=True, tol=1e-3, maxiter=1000)
blockpack = BlockDiagonalOperator([pack for i in range(Nbfreq)], new_axisin=0)
maps =blockpack.T(solution_qubic['x'])  
maps[:,~obs] = 0
x0_convolved[:,~obs] = 0

###############
### Display ###
###############

cmb_convolved = Convol(cmb)
dustQ[~obs]=0
dustU[~obs]=0
dust[~obs]=0
res=maps-x0_convolved

reso=10

close("all")

#T=figure(1)
#T.canvas.set_window_title('Temperature')
#[(hp.gnomview((x0_convolved[i,:,0]).T,sub=(Nbfreq,3,3*i+1),title='convolved',rot=center,reso=reso,min=-200,max=200), hp.gnomview((x_transferred[i,:,0]).T, sub=(Nbfreq,3,3*i+2),title='reconstructed',rot=center,reso=reso,min=-200,max=200),\
#    hp.gnomview((res[i,:,0]).T, sub=(Nbfreq,3,3*i+3),title='residual',rot=center,reso=reso,min=-20,max=20)) for i in range(Nbfreq)]


Q=figure(1)
Q.canvas.set_window_title('Q polarization')
[(hp.gnomview((x0_convolved[i,:,1]).T,sub=(Nbfreq,3,3*i+1),title='convolved',rot=center,reso=reso,min=-3,max=3), hp.gnomview((maps[i,:,1]).T, sub=(Nbfreq,3,3*i+2),title='reconstructed',rot=center,reso=reso,min=-3,max=3),\
hp.gnomview((res[i,:,1]).T, sub=(Nbfreq,3,3*i+3),title='residual',rot=center,reso=reso,min=-3,max=3)) for i in range(Nbfreq)]

U=figure(2)
U.canvas.set_window_title('U polarization')
[(hp.gnomview((x0_convolved[i,:,2]).T,sub=(Nbfreq,3,3*i+1),title='convolved',rot=center,reso=reso,min=-2,max=2), hp.gnomview((maps[i,:,2]).T, sub=(Nbfreq,3,3*i+2),title='reconstructed',rot=center,reso=reso,min=-2,max=2),\
hp.gnomview((res[i,:,2]).T, sub=(Nbfreq,3,3*i+3),title='residual',rot=center,reso=reso,min=-2,max=2)) for i in range(Nbfreq)]

Dust=figure(3)
Dust.canvas.set_window_title('Dust')
[(hp.gnomview(dustT.T*scaling_dust(150,nus[i],1.59),sub=(Nbfreq,3,3*i+1),title='Tdust',rot=center,reso=reso,min=-2,max=2), hp.gnomview(dustQ.T*scaling_dust(150,nus[i],1.59), sub=(Nbfreq,3,3*i+2),title='Qdust',rot=center,reso=reso,min=-2,max=2),\
hp.gnomview(dustU.T*scaling_dust(150,nus[i],1.59), sub=(Nbfreq,3,3*i+3),title='Udust',rot=center,reso=reso,min=-2,max=2)) for i in range(Nbfreq)]


############
### Save ###
############

#k=1
#close("all")
#cartes=open('output_maps_'+str(k)+'.fits','w+r')
#save(cartes,x_transferred)
#cmb_file=open('cmb_'+str(k)+'.fits','w+r')
#save(cmb_file,cmb)
#dust150_file=open('dust150_'+str(k)+'.fits','w+r')
#save(dust150_file,dust)
#dust220_file=open('dust220_'+str(k)+'.fits','w+r')
#save(dust220_file,dust*scaling_dust(150,220,1.59))
#input_file=open('total_input_'+str(k)+'.fits','w+r')
#save(input_file,x0_nu)
#input_convolved=open('total_input_convolved_'+str(k)+'.fits','w+r')
#save(input_convolved,x0_convolved)
#cmb_convolved_file=open('cmb_convolved_'+str(k)+'.fits','w+r')
#save(cmb_convolved_file, cmb_convolved)
#residual_file=open('residual_'+str(k)+'.fits','w+r')
#save(residual_file,res)