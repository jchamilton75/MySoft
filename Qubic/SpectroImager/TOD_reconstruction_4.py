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


def TOD(subnu_min,subnu_max,subdelta_nu,cmb,dust,sampling,scene,effective_duration,verbose=True):

	###################
	### Frequencies ###
	###################

	Nbsubfreq=int(floor(log(subnu_max/subnu_min)/log(1+subdelta_nu)))+1
	sub_nus_edge=subnu_min*np.logspace(0,log(subnu_max/subnu_min)/log(1+subdelta_nu),Nbsubfreq,endpoint=True,base=subdelta_nu+1)
	sub_nus=np.array([(sub_nus_edge[i]+sub_nus_edge[i-1])/2 for i in range(1,Nbsubfreq)])
	sub_deltas=np.array([(sub_nus_edge[i]-sub_nus_edge[i-1]) for i in range(1,Nbsubfreq)])
	Delta=np.max(sub_nus)-np.min(sub_nus)
	Nbsubbands=len(sub_nus) ## = Nbsubfreq-1

	if verbose:
		print('Nombre de bandes utilisées pour la construction : '+str(Nbsubbands))
		print('Sous fréquences centrales utilisées pour la reconstruction : '+str(sub_nus))

	################
	### Coverage ###
	################
	sub_instruments=[QubicInstrument(filter_nu=sub_nus[i]*10**9,detector_nep=2.7e-17) for i in range(Nbsubbands)]
	sub_acqs=[QubicAcquisition(sub_instruments[i], sampling, scene,photon_noise=True, effective_duration=effective_duration) for i in range(Nbsubbands)]
	covlim=0.1
	coverage = np.array([sub_acqs[i].get_coverage() for i in range(Nbsubbands)])
	observed = [(coverage[i] > covlim*np.max(coverage[i])) for i in range(Nbsubbands)]
	obs=reduce(logical_and,tuple(observed[i] for i in range(Nbsubbands)))
	pack = PackOperator(obs, broadcast='rightward')

	#################
	### Input map ###
	#################

	x0=np.zeros((Nbsubbands,Nbpixels,3))
	for i in range(Nbsubbands):
	    #x0[i,:,0]=cmb.T[0]+dust.T[0]*scaling_dust(150,sub_nus[i]e-9,1.59)
	    x0[i,:,1]=cmb.T[1]+dust.T[1]*scaling_dust(150,sub_nus[i],1.59)
	    x0[i,:,2]=cmb.T[2]+dust.T[2]*scaling_dust(150,sub_nus[i],1.59)


	###############################
	### Construction of the TOD ###
	###############################
	Y=0
	for i in range(Nbsubbands):
		sub_acqs_restricted=sub_acqs[i][:,:,obs]
		operator=sub_acqs_restricted.get_operator()
		C=HealpixConvolutionGaussianOperator(fwhm=sub_instruments[i].synthbeam.peak150.fwhm * (150 / (sub_nus[i])))
		Y=Y+operator*pack*C*sub_deltas[i]*10**9*x0[i]

	dnu=sub_instruments[0].filter.bandwidth
	noise=sub_acqs[0].get_noise()*np.sum(sub_deltas)*10**9/dnu
	Y+=noise

	return Y,obs

def convolved_true_maps(nu_min,nu_max,delta_nu,subdelta_nu,cmb,dust,verbose=True):

	#frequencies to reconstruct
	Nbfreq=int(floor(log(nu_max/nu_min)/log(1+delta_nu)))+1 ## number of edge frequencies
	nus_edge=nu_min*np.logspace(0,log(nu_max/nu_min)/log(1+delta_nu),Nbfreq,endpoint=True,base=delta_nu+1) #edge frequencies of reconstructed bands
	nus=np.array([(nus_edge[i]+nus_edge[i-1])/2 for i in range(1,Nbfreq)])
	deltas=(delta_nu)*(nus)
	deltas=np.array([(nus_edge[i]-nus_edge[i-1]) for i in range(1,Nbfreq)])
	Nbbands=len(nus)

	#frequencies assumed to have been used for construction of TOD
	subnu_min=nu_min
	subnu_max=nu_max
	Nbsubfreq=int(floor(log(subnu_max/subnu_min)/log(1+subdelta_nu)))+1
	sub_nus_edge=subnu_min*np.logspace(0,log(subnu_max/subnu_min)/log(1+subdelta_nu),Nbsubfreq,endpoint=True,base=subdelta_nu+1)
	sub_nus=np.array([(sub_nus_edge[i]+sub_nus_edge[i-1])/2 for i in range(1,Nbsubfreq)])
	sub_deltas=np.array([(sub_nus_edge[i]-sub_nus_edge[i-1]) for i in range(1,Nbsubfreq)])	
	Nbsubbands=len(sub_nus)


	#Bands
	bands=[sub_nus[reduce(logical_and,(sub_nus<=nus_edge[i+1],sub_nus>=nus_edge[i]))] for i in range(Nbbands)]
	numbers=np.cumsum(np.array([len(bands[i]) for i in range(Nbbands)]))
	numbers=np.append(0,numbers)
	bands_numbers=np.array([(np.arange(numbers[i],numbers[i+1])) for i in range(Nbbands)])

	if verbose:
		print('Nombre de bandes utilisées pour la construction : '+str(Nbsubbands))
		print('Sous fréquences centrales utilisées pour la construction : '+str(sub_nus))
		print('Nombre de bandes reconstruites : '+str(Nbbands))
		print('Résolution spectrale : '+str(delta_nu))
		print ('Bandes reconstruites : ' + str(bands))	

	#################
	### Input map ###
	#################

	x0=np.zeros((Nbsubbands,Nbpixels,3))
	for i in range(Nbsubbands):
		#x0[i,:,0]=cmb.T[0]+dust.T[0]*scaling_dust(150,sub_nus[i]e-9,1.59)
		x0[i,:,1]=cmb.T[1]+dust.T[1]*scaling_dust(150,sub_nus[i],1.59)
		x0[i,:,2]=cmb.T[2]+dust.T[2]*scaling_dust(150,sub_nus[i],1.59)

	###################################################################################
	### Convolution of the input map (only for comparison to the reconstructed map) ###
	###################################################################################
	x0_convolved=np.zeros((Nbbands,Nbpixels,3))
	for i in range(Nbbands):
		for j in bands_numbers[i]:
			sub_instrument=QubicInstrument(filter_nu=sub_nus[i]*10**9,detector_nep=2.7e-17)
			C=HealpixConvolutionGaussianOperator(fwhm=sub_instrument.synthbeam.peak150.fwhm * (150 / (sub_nus[i])))
			x0_convolved[i]+=C(x0[j])/len(bands_numbers[i])

	return x0_convolved

def reconstruct(Y,nu_min,nu_max,delta_nu,subdelta_nu,sampling,scene,effective_duration, verbose=True):

	###################
	### Frequencies ###
	###################

	#frequencies to reconstruct
	Nbfreq=int(floor(log(nu_max/nu_min)/log(1+delta_nu)))+1 ## number of edge frequencies
	nus_edge=nu_min*np.logspace(0,log(nu_max/nu_min)/log(1+delta_nu),Nbfreq,endpoint=True,base=delta_nu+1) #edge frequencies of reconstructed bands
	nus=np.array([(nus_edge[i]+nus_edge[i-1])/2 for i in range(1,Nbfreq)])
	deltas=(delta_nu)*(nus)
	deltas=np.array([(nus_edge[i]-nus_edge[i-1]) for i in range(1,Nbfreq)])
	Nbbands=len(nus)

	#frequencies assumed to have been used for construction of TOD
	subnu_min=nu_min
	subnu_max=nu_max
	Nbsubfreq=int(floor(log(subnu_max/subnu_min)/log(1+subdelta_nu)))+1
	sub_nus_edge=subnu_min*np.logspace(0,log(subnu_max/subnu_min)/log(1+subdelta_nu),Nbsubfreq,endpoint=True,base=subdelta_nu+1)
	sub_nus=np.array([(sub_nus_edge[i]+sub_nus_edge[i-1])/2 for i in range(1,Nbsubfreq)])
	sub_deltas=np.array([(sub_nus_edge[i]-sub_nus_edge[i-1]) for i in range(1,Nbsubfreq)])
	Nbsubbands=len(sub_nus)


	#Bands
	bands=[sub_nus[reduce(logical_and,(sub_nus<=nus_edge[i+1],sub_nus>=nus_edge[i]))] for i in range(Nbbands)]
	numbers=np.cumsum(np.array([len(bands[i]) for i in range(Nbbands)]))
	numbers=np.append(0,numbers)
	bands_numbers=np.array([(np.arange(numbers[i],numbers[i+1])) for i in range(Nbbands)])

	if verbose:
		print('Nombre de bandes utilisées pour la reconstruction : '+str(Nbsubbands))
		print('Nombre de bandes reconstruites : '+str(Nbbands))
		print('Résolution spectrale : '+str(delta_nu))
		print ('Bandes reconstruites : ' + str(bands))


	################
	### Coverage ###
	################
	sub_instruments=[QubicInstrument(filter_nu=sub_nus[i]*10**9,detector_nep=2.7e-17) for i in range(Nbsubbands)]
	sub_acqs=[QubicAcquisition(sub_instruments[i], sampling, scene,photon_noise=True, effective_duration=effective_duration) for i in range(Nbsubbands)]
	covlim=0.1
	coverage = np.array([sub_acqs[i].get_coverage() for i in range(Nbsubbands)])
	observed = [(coverage[i] > covlim*np.max(coverage[i])) for i in range(Nbsubbands)]
	obs=reduce(logical_and,tuple(observed[i] for i in range(Nbsubbands)))
	pack = PackOperator(obs, broadcast='rightward')

	######################
	### Reconstruction ###
	######################
	sub_instruments=[QubicInstrument(filter_nu=sub_nus[i]*10**9,detector_nep=2.7e-17) for i in range(Nbsubbands)]
	sub_acqs=[QubicAcquisition(sub_instruments[i], sampling, scene,photon_noise=True, effective_duration=3) for i in range(Nbsubbands)]
	sub_acqs_restricted=[sub_acqs[i][:,:,obs] for i in range(Nbsubbands)]
	operators=np.array([sub_acqs_restricted[i].get_operator() for i in range(Nbsubbands)])
	K=np.array([np.sum([operators[j]*sub_deltas[j]*10**9 for j in bands_numbers[i]],axis=0) for i in range(Nbbands)])
	H=BlockRowOperator([K[i] for i in range(Nbbands)], new_axisin=0)
	invntt = sub_acqs[0].get_invntt_operator()
	A=H.T*invntt*H
	b = (H.T * invntt)(Y)
	preconditioner = BlockDiagonalOperator([DiagonalOperator(1 / coverage[0][obs], broadcast='rightward') for i in range(Nbbands)], new_axisin=0) 
	solution_qubic = pcg(A, b, M=preconditioner ,disp=True, tol=1e-3, maxiter=1000)
	blockpack = BlockDiagonalOperator([pack for i in range(Nbbands)], new_axisin=0)
	maps =blockpack.T(solution_qubic['x'])  
	maps[:,~obs] = 0

	return maps, bands

nu_min=150
nu_max=220
effective_duration=3000	
subdelta_construction=0.02

Y,obs=TOD(nu_min,nu_max,subdelta_construction,cmb,dust,sampling,scene,effective_duration,verbose=True)

delta_nu=0.2
subdelta_reconstruction=0.05
x0_convolved=convolved_true_maps(nu_min,nu_max,delta_nu,subdelta_construction,cmb,dust)
maps, bands=reconstruct(Y,nu_min,nu_max,delta_nu,subdelta_reconstruction,sampling,scene,effective_duration)
x0_convolved[:,~obs]=0
res=maps-x0_convolved
Nbbands=len(bands)

###############
### Display ###
###############

reso=10


close("all")

Q=figure(1)
Q.canvas.set_window_title('Q polarization')
[(hp.gnomview((x0_convolved[i,:,1]).T,sub=(Nbbands,3,3*i+1),title='convolved',rot=center,reso=reso,min=-3,max=3), hp.gnomview((maps[i,:,1]).T, sub=(Nbbands,3,3*i+2),title='reconstructed',rot=center,reso=reso,min=-3,max=3),\
hp.gnomview((res[i,:,1]).T, sub=(Nbbands,3,3*i+3),title='residual',rot=center,reso=reso,min=-1,max=1)) for i in range(Nbbands)]


U=figure(2)
U.canvas.set_window_title('U polarization')
[(hp.gnomview((x0_convolved[i,:,2]).T,sub=(Nbbands,3,3*i+1),title='convolved',rot=center,reso=reso,min=-3,max=3), hp.gnomview((maps[i,:,2]).T, sub=(Nbbands,3,3*i+2),title='reconstructed',rot=center,reso=reso,min=-3,max=3),\
hp.gnomview((res[i,:,2]).T, sub=(Nbbands,3,3*i+3),title='residual',rot=center,reso=reso,min=-1,max=1)) for i in range(Nbbands)]

Dust=figure(3)
Dust.canvas.set_window_title('Dust')
[(hp.gnomview(dustT.T*scaling_dust(150,np.mean(bands[i]),1.59),sub=(Nbbands,3,3*i+1),title='Tdust',rot=center,reso=reso,min=-2,max=2), hp.gnomview(dustQ.T*scaling_dust(150,np.mean(bands[i]),1.59), sub=(Nbbands,3,3*i+2),title='Qdust',rot=center,reso=reso,min=-2,max=2),\
hp.gnomview(dustU.T*scaling_dust(150,np.mean(bands[i]),1.59), sub=(Nbbands,3,3*i+3),title='Udust',rot=center,reso=reso,min=-2,max=2)) for i in range(Nbbands)]

## Remarques:
# il semble rester des residus de CMB sur Q et U
# le fait d'augmenter fortement effective duration ne semble pas changer le niveau de bruit sur les cartes de residus





