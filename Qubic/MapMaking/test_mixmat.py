
from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import pycamb
from pyoperators import (
    DenseOperator, DegreesOperator, DiagonalOperator, RadiansOperator,
    Cartesian2SphericalOperator, Spherical2CartesianOperator, pcg)
from pysimulators import (
    CartesianEquatorial2GalacticOperator, ProjectionOperator, FitsArray)
from pysimulators import (
    ProjectionOperator, SphericalEquatorial2GalacticOperator, FitsArray)
from qubic import QubicAcquisition, QubicInstrument, create_sweeping_pointings, create_random_pointings
from MapMaking import MapMaking as mm

###### Input Power spectrum ###################################
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
         'tensor_ratio':0.1,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False}

lmax = 1200
ell = np.arange(1,lmax+1)
T,E,B,X = pycamb.camb(lmax+1,**params)
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


def statstr(vec):
    m=np.mean(vec)
    s=np.std(vec)
    return '{0:.4f} +/- {1:.4f}'.format(m,s)


################### Make a MC: make various maps useful later
from pysimulators import FitsArray
dtheta=15.
npointings=5000
calerror=1e-2
nbmc=1000
for n in np.arange(nbmc):
    print(' ')
    print(' ')
    print(' ')
    print('#############################')
    print('####### mc '+str(n))
    print('#############################')
    mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
    rnd_pointing=create_random_pointings([racenter, deccenter],npointings,dtheta)
    tod=mm.map2tod(maps,rnd_pointing,qubic)
    tod_spoiled=tod.copy()
    rndcal=np.random.normal(loc=1.,scale=calerror,size=(len(detectors),2))
    for i in np.arange(len(detectors)):
        for j in np.arange(2):
            tod_spoiled[i,:,j]=rndcal[i,j]*tod[i,:,j]

    output_maps_all,call=mm.tod2map(tod,rnd_pointing,qubic,disp=False)
    output_maps_all_spoiled,call_spoiled=mm.tod2map(tod_spoiled,rnd_pointing,qubic,disp=False)
    output_maps_det,cdet=mm.tod2map_perdet(tod,rnd_pointing,qubic,disp=False)
    output_maps_det_spoiled,cdet_spoiled=mm.tod2map_perdet(tod_spoiled,rnd_pointing,qubic,disp=False)
    covmin=np.arange(12*nside**2)
    for i in np.arange(12*nside**2):
        covmin[i]=np.min([call[i],call_spoiled[i],cdet[i],cdet_spoiled[i]])

    strn = str(np.random.randint(1e6))
    FitsArray(call,copy=False).save('McMixMat/map'+strn+'_covall.fits')
    FitsArray(output_maps_all[:,0],copy=False).save('McMixMat/map'+strn+'_Iall.fits')
    FitsArray(output_maps_all[:,1],copy=False).save('McMixMat/map'+strn+'_Qall.fits')
    FitsArray(output_maps_all[:,2],copy=False).save('McMixMat/map'+strn+'_Uall.fits')
    FitsArray(cdet,copy=False).save('McMixMat/map'+strn+'_covdet.fits')
    FitsArray(output_maps_det[:,0],copy=False).save('McMixMat/map'+strn+'_Idet.fits')
    FitsArray(output_maps_det[:,1],copy=False).save('McMixMat/map'+strn+'_Qdet.fits')
    FitsArray(output_maps_det[:,2],copy=False).save('McMixMat/map'+strn+'_Udet.fits')

    FitsArray(call_spoiled,copy=False).save('McMixMat/map'+strn+'_covall_spoiled.fits')
    FitsArray(output_maps_all_spoiled[:,0],copy=False).save('McMixMat/map'+strn+'_Iall_spoiled.fits')
    FitsArray(output_maps_all_spoiled[:,1],copy=False).save('McMixMat/map'+strn+'_Qall_spoiled.fits')
    FitsArray(output_maps_all_spoiled[:,2],copy=False).save('McMixMat/map'+strn+'_Uall_spoiled.fits')
    FitsArray(cdet_spoiled,copy=False).save('McMixMat/map'+strn+'_covdet_spoiled.fits')
    FitsArray(output_maps_det_spoiled[:,0],copy=False).save('McMixMat/map'+strn+'_Idet_spoiled.fits')
    FitsArray(output_maps_det_spoiled[:,1],copy=False).save('McMixMat/map'+strn+'_Qdet_spoiled.fits')
    FitsArray(output_maps_det_spoiled[:,2],copy=False).save('McMixMat/map'+strn+'_Udet_spoiled.fits')





