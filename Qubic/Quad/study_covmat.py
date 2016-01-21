from __future__ import division

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os

from pyquad import pyquad

from pyoperators import pcg, DiagonalOperator, UnpackOperator
from pysimulators import ProjectionInMemoryOperator
from qubic import QubicConfiguration, QubicInstrument, create_random_pointings

path = os.path.dirname('/Users/hamilton/idl/pro/Qubic/People/pierre/qubic/script/script_ga.py')



#### Study covariance matrix
####################### restore stuff ###################
import pickle
from pysimulators import FitsArray
## Pointing
infile=open('saved_ptg.dat', 'rb')
data=pickle.load(infile)
infile.close()
pointings=data['pointings']
mask=data['mask']

signoise=0.5
nbmc=10000
data=0
## Covariance matrix
covmc=FitsArray('covmc'+str(signoise)+'_'+str(nbmc)+'.dat')
cormc=FitsArray('cormc'+str(signoise)+'_'+str(nbmc)+'.dat')
###########################################################

##### build coverage
a=np.loadtxt('./cl_r=0.1bis2.txt')
ell=a[:,0]
ctt=np.concatenate([[0,0],a[:,1]*1e12*2*np.pi/(ell*(ell+1))])
cee=np.concatenate([[0,0],a[:,2]*1e12*2*np.pi/(ell*(ell+1))])
cte=np.concatenate([[0,0],a[:,4]*1e12*2*np.pi/(ell*(ell+1))])
cbb=np.concatenate([[0,0],a[:,7]*1e12*2*np.pi/(ell*(ell+1))])
ell=np.concatenate([[0,1],ell])
spectra=[ell,ctt,cte,cee,cbb]
nside=128
map_orig=hp.synfast(spectra[4],nside,fwhm=0,pixwin=True)
input_map=map_orig.copy()
kmax = 2
qubic = QubicInstrument('monochromatic,nopol',nside=128)
obs = QubicConfiguration(qubic, pointings)
C = obs.get_convolution_peak_operator()
P = obs.get_projection_peak_operator(kmax=kmax)
H = P * C
tod = H(input_map)
coverage = P.T(np.ones_like(tod))





# study covariance matrix
# 1/ it is extremely diagonal => the map making does not correlate pixels significantly... weird
from Homogeneity import fitting

def ffunc(x,pars):
    return(pars[0]*x**pars[1])

npix=diag(covmc).size

clf()
plot(signoise**2/coverage[~mask],diag(covmc),'r.')
bla=fitting.dothefit(signoise**2/coverage[~mask],diag(covmc),ones(npix),[7500.,1.],functname=ffunc,method='minuit')
xlabel('naive diag covariance matrix')
ylabel('MC diag covariance matrix')
xx=linspace(0,0.001,10000)
plot(xx,xx,'k--')
plot(xx,ffunc(xx,bla[1]),lw=3)

clf()
plot(bla[1][0]*(signoise**2/coverage[~mask])**bla[1][1],diag(covmc),'r.')
xlabel('Model Diag Covariance Matrix from Coverage')
ylabel('MC Diag Covariance Matrix')
xx=linspace(0,0.06,10000)
plot(xx,xx,'g--',lw=3)



mapneff=np.zeros(12*nside**2)
mapneff[~mask]=1./diag(covmc)*signoise**2
#hp.gnomview(mapneff,rot=[0,90],reso=10,title='coveff')
cov=np.zeros(12*nside**2)
cov[~mask]=coverage[~mask]
#hp.gnomview(cov,rot=[0,90],reso=10,title='nhits')

clf()
plot(cov[~mask],mapneff[~mask],'r.')
ylim(0,max(mapneff[~mask]))
xlim(0,max(cov[~mask]))
xx=linspace(0,10000,10000)
plot(xx,xx,'k--')
plot(xx,xx/7,'k--')
ylabel('effective nhits')
xlabel('nhits')

iprings=np.arange(12*nside**2)
vecs=hp.pix2vec(int(nside),iprings[~mask])
cosangles=np.dot(np.transpose(vecs),vecs)
ang=np.arccos(cosangles)*180/np.pi


clf()
xn,yn,dxn,dyn=pyquad.profile(ang.flatten(),cormc.flatten(),0.001,50,200,plot=True,dispersion=False)

clf()
imshow(abs(cormc),interpolation='nearest',vmin=0,vmax=0.2)
title('Map Correlation Matrix')
colorbar()



clf()
errorbar(xn,yn,xerr=dxn,yerr=dyn,fmt='bo')
from Homogeneity import SplineFitting
spl=SplineFitting.MySplineFitting(xn,yn,dyn,100)
#plot(xn,spl(xn),'r',lw=3)
plot(xn,xn*0,'k--')
ylabel('Noise Correlation Matrix')
xlabel('Angle [degrees]')




sol2 = pcg(P_packed.T * P_packed, P_packed.T(tod*0+1), M=DiagonalOperator(1/coverage[~mask]), disp=True)
errmap = unpack(sol2['x'])
hp.gnomview(errmap,rot=[0,90],reso=14)
mm=np.zeros(12*nside**2)
mm[maskok]=sqrt(diag(covmc))
hp.gnomview(mm,rot=[0,90],reso=14)


