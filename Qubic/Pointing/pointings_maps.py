from __future__ import division

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
import string

from Quad import pyquad

from pyoperators import pcg, DiagonalOperator, UnpackOperator
from pysimulators import ProjectionInMemoryOperator
from qubic import QubicAcquisition, QubicInstrument, create_random_pointings
from qubic import QubicInstrument, create_random_pointings

path = os.path.dirname('/Users/hamilton/idl/pro/Qubic/People/pierre/qubic/script/script_ga.py')

###############################################################################################
signoise=0.1
nbmc=10000
correc=True

a=np.loadtxt('/Users/hamilton/Qubic/cl_r=0.1bis2.txt')
ell=a[:,0]
ctt=np.concatenate([[0,0],a[:,1]*1e12*2*np.pi/(ell*(ell+1))])
cee=np.concatenate([[0,0],a[:,2]*1e12*2*np.pi/(ell*(ell+1))])
cte=np.concatenate([[0,0],a[:,4]*1e12*2*np.pi/(ell*(ell+1))])
cbb=np.concatenate([[0,0],a[:,7]*1e12*2*np.pi/(ell*(ell+1))])
ell=np.concatenate([[0,1],ell])
spectra=[ell,ctt,cte,cee,cbb]

def map2TOD(input_map, pointing,kmax=2):
    ns=hp.npix2nside(input_map.size)
    qubic = QubicInstrument('monochromatic,nopol',nside=ns)
    #### configure observation
    obs = QubicAcquisition(qubic, pointing)
    C = obs.get_convolution_peak_operator()
    P = obs.get_projection_peak_operator(kmax=kmax)
    H = P * C
    # Produce the Time-Ordered data
    tod = H(input_map)
    input_map_conv = C(input_map)
    return(tod,input_map_conv,P)

def TOD2map(tod,pointing,nside,kmax=2,disp=True,P=False,covmin=10):
    qubic = QubicInstrument('monochromatic,nopol',nside=nside)
    #### configure observation
    obs = QubicAcquisition(qubic, pointing)
    if not P:
        P = obs.get_projection_peak_operator(kmax=kmax)
    coverage = P.T(np.ones_like(tod))
    mask=coverage < covmin
    P.matrix.pack(mask)
    P_packed = ProjectionInMemoryOperator(P.matrix)
    unpack = UnpackOperator(mask)
    # data
    solution = pcg(P_packed.T * P_packed, P_packed.T(tod), M=DiagonalOperator(1/coverage[~mask]), disp=disp)
    output_map = unpack(solution['x'])
    output_map[mask] = np.nan
    coverage[mask]=np.nan
    return(output_map,mask,coverage)


#### test
nside=128
npoints=1000
ini=hp.synfast(spectra[4],nside,fwhm=0,pixwin=True)
true_pointings = create_random_pointings(npoints, 20)
tod,iniconv,P=map2TOD(ini,true_pointings)
out,mask,coverage=TOD2map(tod,true_pointings,128,P=P)

mapin=hp.ud_grade(iniconv,nside_out=128,order_in='RING',order_out='RING')
mapin[mask]=np.nan

hp.gnomview(out,rot=[0,90],reso=30,min=-0.5,max=0.5)
hp.gnomview(mapin,rot=[0,90],reso=30,min=-0.5,max=0.5)
hp.gnomview(out-mapin,rot=[0,90],reso=30,min=-0.05,max=0.05)
hp.gnomview(coverage,rot=[0,90],reso=30,min=0)



from Homogeneity import fitting
l=np.arange(3*nside)

def fct(x,pars):
    f=np.poly1d(pars[1:])
    return(exp(-x**2*pars[0]**2)+pars[1]*x**2+pars[2]*x**4)

nsbig=512
nsides=np.array([128])
nns=nsides.size
npoints=3000
nbmc=100
nbsigptg=10
#### in arcsec
mini=1.
maxi=3600
sigptg=np.logspace(np.log10(mini),np.log10(maxi),nbsigptg)
sigma=np.zeros((nns,nbsigptg))
dsigma=np.zeros((nns,nbsigptg))
sigrec=np.zeros((nns,nbmc,nbsigptg))

dictio={}
for n in np.arange(nns):
    nside=nsides[n]
    print(nside)    
    allratios=np.zeros((nbmc,nbsigptg,3*nside))
    for j in np.arange(nbsigptg):
        allressig=np.zeros(nbmc)
        print('Pointing Sigma = %0.5f'%sigptg[j])
        for i in np.arange(nbmc):
            print(str(nside)+' '+str(j)+' '+str('   Realization %0.f'%i)+str(' out of %0.f'%nbmc))
            # input map high resolution
            map_orig=hp.synfast(spectra[4],nsbig,fwhm=0,pixwin=True)
            # true pointings
            true_pointings = create_random_pointings(npoints, 20)
            # TOD with true pointings
            tod,mconv,Ptrue=map2TOD(map_orig,true_pointings)
            #spoiled pointings
            new_pointings=np.zeros_like(true_pointings)
            dtheta=np.random.randn(npoints)*sigptg[j]/3600
            dphi=np.random.randn(npoints)*sigptg[j]/3600/np.sin(true_pointings[:,0]*np.pi/180)
            new_pointings[:,0]=true_pointings[:,0]+dtheta
            new_pointings[:,1]=true_pointings[:,1]+dphi
            new_pointings[:,2]=true_pointings[:,2]+0
            # maps
            maptrueptg,masktrue,cctrue=TOD2map(tod,true_pointings,nside,disp=False,covmin=100)
            map,mask,cc=TOD2map(tod,new_pointings,nside,disp=False,covmin=100)
            mm=mask+masktrue
            mask[mm]=np.nan
            masktrue[mm]=np.nan
            allressig[i]=np.std(map[~mm]-maptrueptg[~mm])
            cltrueptg=hp.anafast(nan_to_num(maptrueptg))
            cl=hp.anafast(nan_to_num(map))
            allratios[i,j,:]=cl/cltrueptg
            l=np.arange(3*nside)
            fit=fitting.dothefit(l,cl/cltrueptg,np.ones(3*nside),[np.radians(sigptg[j]/3600),0.,0.],functname=fct,method='mpfit')
            clf()
            plot(l,cl/cltrueptg)
            plot(l,fct(l,fit[1]))
            draw()
            sigrec[n,i,j]=fit[1][0]
            
        sigma[n,j]=np.mean(allressig)
        dsigma[n,j]=np.std(allressig)
    dictio[nside]=allratios


    
clf()
xlim(np.min(sigptg)/2,np.max(sigptg)*2)
xscale('log')
yscale('log')
errorbar(sigptg,sigma[0,:],yerr=dsigma[0,:],fmt='ro',label='nside=128')
errorbar(sigptg,sigma[1,:],yerr=dsigma[1,:],fmt='bo',label='nside=256')
errorbar(sigptg,sigma[2,:],yerr=dsigma[2,:],fmt='go',label='nside=512')
xlabel('RMS pointing in arcmin')
ylabel('RMS residuals on map')
legend()

dict_avratio={}
for n in np.arange(nns):
    nside=nsides[n]
    print(nside)    
    avratio=np.zeros((nbsigptg,3*nside))
    for j in np.arange(nbsigptg):
        print('Pointing Sigma = %0.5f'%sigptg[j])
        for l in np.arange(3*nside):
            avratio[j,l]=np.mean(dictio[nside][:,j,l])
    dict_avratio[nside]=avratio




clf()
for i in np.arange(nbsigptg): 
    plot(dict_avratio[128][i,:],label=str(sigptg[i])+' arcsec')
legend(loc='lower right')
ylim(0,1.5)

l=np.arange(3*128)
clf()
for i in np.arange(nbsigptg): 
    plot(-np.exp(dict_avratio[128][i,:])/l**2,label=str(sigptg[i])+' arcsec')
legend(loc='lower right')
