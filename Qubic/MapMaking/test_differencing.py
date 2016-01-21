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

################# Input Power spectrum ###################################
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
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


################## Input Maps #############################################
nside=128
lmax=3*nside
ell=spectra[0]
maskl=ell<(lmax+1)
bl=np.ones(maskl.size)
fwhmrad=0.5*np.pi/180
mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
hp.mollview(mapi)
hp.mollview(mapq)
hp.mollview(mapu)
maps=np.transpose(np.array([mapi,mapq,mapu]))


################# Qubic Instrument #########################################
qubic = QubicInstrument('monochromatic',nside=nside)
detectors=qubic.detector.packed
ndet = len(detectors)
clf()
subplot(2,1,1)
plot(detectors.center[0:ndet/2,0],detectors.center[0:ndet/2,1],'ro')
subplot(2,1,2)
plot(detectors.center[ndet/2:2*ndet/2,0],detectors.center[ndet/2:2*ndet/2,1],'bo')





