from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
#from Quad import qml
#from Quad import pyquad
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

maxiter = 100
tol = 5e-6
racenter = 0.0      # deg
deccenter = -57.0   # deg
angspeed = 1        # deg/sec
delta_az = 20.      # deg
angspeed_psi = 0.1  # deg/sec
maxpsi = 45.        # deg
nsweeps_el = 300
duration = 24      # hours
ts = 1000        # seconds
center = equ2gal(racenter, deccenter)
#sampling = create_sweeping_pointings([racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,angspeed_psi, maxpsi)

sampling = create_random_pointings([racenter, deccenter], 1000,30)


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
    Delta=subnu_max-subnu_min
    Nbsubbands=len(sub_nus) ## = Nbsubfreq-1

    if verbose:
        print('Nombre de bandes utilisées pour la construction : '+str(Nbsubbands))
        print('Sous fréquences centrales utilisées pour la construction : '+str(sub_nus))
        print('Edges : '+str(sub_nus_edge))

    ################
    ### Coverage ###
    ################
    sub_instruments=[QubicInstrument(filter_nu=sub_nus[i]*10**9,filter_relative_bandwidth=sub_deltas[i]/sub_nus[i],detector_nep=2.7e-17) for i in range(Nbsubbands)]
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
    dnu=sub_instruments[0].filter.bandwidth
    Y=0
    Y_noisy=0
   

    # Noiseless TOD
    for i in range(Nbsubbands):
        sub_acqs_restricted=sub_acqs[i][:,:,obs]
        operator=sub_acqs_restricted.get_operator()
        C=HealpixConvolutionGaussianOperator(fwhm=sub_instruments[i].synthbeam.peak150.fwhm * (150 / (sub_nus[i])))
        Y=Y+operator*pack*C*x0[i]
    

    # Global instrument creqted to get the noise over the entire instrument bandwidth
    Global_instrument=QubicInstrument(filter_nu=(subnu_max+subnu_min)/2,filter_relative_bandwidth=Delta/((subnu_max+subnu_min)/2),detector_nep=2.7e-17)
    Global_acquisition=QubicAcquisition(Global_instrument, sampling, scene,photon_noise=True, effective_duration=effective_duration)


    noise_instrument=Global_acquisition.get_noise()
    sigma=np.std(noise_instrument)
    mean=np.mean(noise_instrument)
    white_noise=np.random.normal(mean,sigma,shape(Y))
    Y_noisy=Y+white_noise


    

    #noise=sub_acqs[0].get_noise()*np.sum(sub_deltas)*10**9/dnu
    #0Y+=noise

    return Y_noisy,obs

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
        print('Edges : '+str(nus_edge)) 
        print('Sub Edges : '+str(sub_nus_edge))

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
            sub_instrument=QubicInstrument(filter_nu=sub_nus[j]*10**9,filter_relative_bandwidth=sub_deltas[j]/sub_nus[j],detector_nep=2.7e-17)
            C=HealpixConvolutionGaussianOperator(fwhm=sub_instrument.synthbeam.peak150.fwhm * (150 / (sub_nus[j])))
            x0_convolved[i]+=C(x0[j])*sub_deltas[j]/np.sum(sub_deltas[bands_numbers[i]])


    return x0_convolved

def reconstruct(Y,nu_min,nu_max,delta_nu,subdelta_nu,sampling,scene,effective_duration, verbose=True,return_mono=True):

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
    sub_instruments=[QubicInstrument(filter_nu=sub_nus[i]*10**9,filter_relative_bandwidth=sub_deltas[i]/sub_nus[i],detector_nep=2.7e-17) for i in range(Nbsubbands)]
    sub_acqs=[QubicAcquisition(sub_instruments[i], sampling, scene,photon_noise=True, effective_duration=effective_duration) for i in range(Nbsubbands)]
    covlim=0.1
    coverage = np.array([sub_acqs[i].get_coverage() for i in range(Nbsubbands)])
    observed = [(coverage[i] > covlim*np.max(coverage[i])) for i in range(Nbsubbands)]
    obs=reduce(logical_and,tuple(observed[i] for i in range(Nbsubbands)))
    pack = PackOperator(obs, broadcast='rightward')

    ######################
    ### Reconstruction ###
    ######################
    sub_acqs_restricted=[sub_acqs[i][:,:,obs] for i in range(Nbsubbands)]
    operators=np.array([sub_acqs_restricted[i].get_operator() for i in range(Nbsubbands)])
    K=np.array([np.sum([operators[j] for j in bands_numbers[i]],axis=0) for i in range(Nbbands)])
    H=BlockRowOperator([K[i] for i in range(Nbbands)], new_axisin=0)
    invntt = sub_acqs[0].get_invntt_operator()
    A=H.T*invntt*H
    b = (H.T * invntt)(Y)
    preconditioner = BlockDiagonalOperator([DiagonalOperator(1 / coverage[0][obs], broadcast='rightward') for i in range(Nbbands)], new_axisin=0) 
    solution_qubic = pcg(A, b, M=preconditioner ,disp=True, tol=1e-3, maxiter=1000)
    blockpack = BlockDiagonalOperator([pack for i in range(Nbbands)], new_axisin=0)
    maps =blockpack.T(solution_qubic['x'])  
    maps[:,~obs] = 0

    #################
    ### Input map ###
    #################

    x0=np.zeros((Nbsubbands,Nbpixels,3))
    for i in range(Nbsubbands):
        #x0[i,:,0]=cmb.T[0]+dust.T[0]*scaling_dust(150,sub_nus[i]e-9,1.59)
        x0[i,:,1]=cmb.T[1]+dust.T[1]*scaling_dust(150,sub_nus[i],1.59)
        x0[i,:,2]=cmb.T[2]+dust.T[2]*scaling_dust(150,sub_nus[i],1.59)

    maps_mono=np.zeros((Nbbands,Nbpixels,3))
    if return_mono:
        (m,n)=shape(Y)
        Y_mono=np.zeros((Nbbands,m,n))
        for i in range(Nbbands):
            for j in bands_numbers[i]:
                C=HealpixConvolutionGaussianOperator(fwhm=sub_instruments[j].synthbeam.peak150.fwhm * (150 / (sub_nus[j])))
                Y_mono[i]=Y_mono[i]+operators[j]*pack*C*x0[j]
            Global_instrument=QubicInstrument(filter_nu=nus[i],filter_relative_bandwidth=deltas[i]/nus[i],detector_nep=2.7e-17)
            Global_acquisition=QubicAcquisition(Global_instrument, sampling, scene,photon_noise=True, effective_duration=effective_duration)
            noise=Global_acquisition.get_noise()
            Y_mono[i]=Y_mono[i]+noise
            H_mono=np.sum([operators[j] for j in bands_numbers[i]],axis=0)
            A_mono=H_mono.T*invntt*H_mono
            b_mono = (H_mono.T * invntt)(Y_mono[i])
            preconditioner_mono = DiagonalOperator(1 / coverage[0][obs], broadcast='rightward') 
            solution_qubic_mono = pcg(A_mono, b_mono, M=preconditioner_mono ,disp=True, tol=1e-3, maxiter=1000)
            maps_mono[i] =pack.T(solution_qubic_mono['x'])  
            maps_mono[i,~obs] = 0
        return maps, maps_mono, bands, deltas

    return maps, bands, deltas

sampling = create_random_pointings([racenter, deccenter], 1000,20)

nu_min=150
nu_max=220
effective_duration=5000
subdelta_construction=0.05

Y,obs=TOD(nu_min,nu_max,subdelta_construction,cmb,dust,sampling,scene,effective_duration,verbose=True)


delta_nu=0.2
subdelta_reconstruction=0.05
x0_convolved=convolved_true_maps(nu_min,nu_max,delta_nu,subdelta_construction,cmb,dust)
maps, maps_mono, bands, deltas=reconstruct(Y,nu_min,nu_max,delta_nu,subdelta_reconstruction,
        sampling,scene,effective_duration)
x0_convolved[:,~obs]=0
res=maps-x0_convolved
res_mono=maps_mono-x0_convolved
noise_mono=np.std(res_mono[:,obs], axis=1)
noise_poly=np.std(res[:,obs], axis=1)

reso=30
Nbbands=len(bands)
Q=figure(1) 
Q.canvas.set_window_title('Q polarization')
[(hp.gnomview((x0_convolved[i,:,1]).T,sub=(Nbbands,3,3*i+1),title='convolved',rot=center,reso=reso,min=-3,max=3), 
    hp.gnomview((maps[i,:,1]).T, sub=(Nbbands,3,3*i+2),title='reconstructed',rot=center,reso=reso,min=-3,max=3),
    hp.gnomview((res[i,:,1]).T, sub=(Nbbands,3,3*i+3),title='residual',rot=center,reso=reso,min=-3,max=3)) 
    for i in range(Nbbands)]


clf()
hp.gnomview(res[1,:,1]+res[0,:,1], sub=(1,2,1), title='Sum', rot=center, reso=reso)
hp.gnomview(res[1,:,1]-res[0,:,1], sub=(1,2,2), title='Difference', rot=center, reso=reso)

clf()
plot(res[1,obs,0], res[0,obs,0],'.')
plot(res[1,obs,1], res[0,obs,1],'.')
plot(res[1,obs,2], res[0,obs,2],'.')




delta_nu=0.1
subdelta_reconstruction=0.05
x0_convolved=convolved_true_maps(nu_min,nu_max,delta_nu,subdelta_construction,cmb,dust)
maps, maps_mono, bands, deltas=reconstruct(Y,nu_min,nu_max,delta_nu,subdelta_reconstruction,
        sampling,scene,effective_duration)
x0_convolved[:,~obs]=0
res=maps-x0_convolved
res_mono=maps_mono-x0_convolved
noise_mono2=np.std(res_mono[:,obs], axis=1)
noise_poly2=np.std(res[:,obs], axis=1)


### Donne
In [128]: noise_poly
Out[128]: 
array([[ 1.0205025 ,  1.45255157,  1.46254105],
       [ 0.82467383,  1.20020626,  1.19310243]])

In [129]: noise_poly2
Out[129]: 
array([[  5.96020678,   8.18489821,   8.07006024],
       [  9.41543102,  12.81079761,  12.6226995 ],
       [  5.77843471,   7.83853192,   7.93794977],
       [  5.68650676,   7.68015309,   7.41917447]])

In [130]: noise_mono
Out[130]: 
array([[ 0.53816213,  0.75573923,  0.77059425],
       [ 0.42815919,  0.61663319,  0.6020461 ]])

In [131]: noise_mono2
Out[131]: 
array([[ 0.68251181,  0.96644686,  0.934959  ],
       [ 1.25707495,  1.76506751,  1.73168738],
       [ 0.63335   ,  0.90904965,  0.90106651],
       [ 0.69033911,  0.94693227,  0.95925736]])

### Comparer
np.sqrt(noise_mono2[0,:]**2+noise_mono2[1,:]**2)/2
np.sqrt(noise_mono2[2,:]**2+noise_mono2[3,:]**2)/2
### avec noise_mono: idéalement ça devrait être le même niveau de bruit
# ici le rapport est
np.sqrt(noise_mono2[0,:]**2+noise_mono2[1,:]**2)/2/noise_mono[0,:]
np.sqrt(noise_mono2[2,:]**2+noise_mono2[3,:]**2)/2/noise_mono[1,:]
## donc 1.3 pour la plus basse frequence et 1 pour la plus haute

### Pour poly c'est vraiment pas bon...
### Comparer
np.sqrt(noise_poly2[0,:]**2+noise_poly2[1,:]**2)/2
np.sqrt(noise_poly2[2,:]**2+noise_poly2[3,:]**2)/2
### avec noise_poly: idéalement ça devrait être le même niveau de bruit
# ici le rapport est
np.sqrt(noise_poly2[0,:]**2+noise_poly2[1,:]**2)/2/noise_poly[0,:]
np.sqrt(noise_poly2[2,:]**2+noise_poly2[3,:]**2)/2/noise_poly[1,:]
## donc tres mauvais... pourquoi ?

#### Incompréhensible pour moi. Est-ce que le fait que ts=1000 fout tout en l'air (surement) ?

#questions:
# qu'est ce que mono et poly ?


delta_nu=0.1
subdelta_reconstruction=0.02
x0_convolved=convolved_true_maps(nu_min,nu_max,delta_nu,subdelta_construction,cmb,dust)
maps, maps_mono, bands, deltas=reconstruct(Y,nu_min,nu_max,delta_nu,subdelta_reconstruction,
        sampling,scene,effective_duration)
x0_convolved[:,~obs]=0
res=maps-x0_convolved
res_mono=maps_mono-x0_convolved
noise_mono2=np.std(res_mono[:,obs], axis=1)
noise_poly2=np.std(res[:,obs], axis=1)






nu_min=135.
nu_max=165.
delta_nu=0.05
#Nbbands=2
effective_duration=5
subdelta_construction=0.025

Y,obs=TOD(nu_min,nu_max,subdelta_construction,cmb,dust,sampling,scene,effective_duration,verbose=True)



subdelta_reconstruction=0.025
x0_convolved=convolved_true_maps(nu_min,nu_max,delta_nu,subdelta_construction,cmb,dust)

maps, maps_mono, bands, deltas=reconstruct(Y,nu_min,nu_max,delta_nu,subdelta_reconstruction,
        sampling,scene,effective_duration)
x0_convolved[:,~obs]=0
res=maps-x0_convolved
res_mono=maps_mono-x0_convolved
noise_mono=np.std(res_mono[:,obs], axis=1)
noise_poly=np.std(res[:,obs], axis=1)



