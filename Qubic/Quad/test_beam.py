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

nside=1024
input_map=np.zeros(12*nside**2)
input_map[0]=1

#### QUBIC Instrument
kmax = 2
qubic = QubicInstrument('monochromatic,nopol',nside=nside)
pointings = create_random_pointings(10000, 20)

#### configure observation
obs = QubicConfiguration(qubic, pointings)
C = obs.get_convolution_peak_operator()
P = obs.get_projection_peak_operator(kmax=kmax)
H = P * C

# Produce the Time-Ordered data
tod = H(input_map)

# map-making
coverage = P.T(np.ones_like(tod))
mask = coverage < 10
P.matrix.pack(mask)
P_packed = ProjectionInMemoryOperator(P.matrix)
unpack = UnpackOperator(mask)
solution = pcg(P_packed.T * P_packed, P_packed.T(tod), M=DiagonalOperator(1/coverage[~mask]), disp=True)
output_map = unpack(solution['x'])

# some display
output_map[mask] = np.nan
hp.gnomview(input_map, rot=[0,90], reso=0.3, xsize=600,title='Input Map')
hp.gnomview(output_map, rot=[0,90], reso=0.3, xsize=600,title='Output Map')

# get theta
iprings=np.arange(12*nside**2)
vecs=hp.pix2vec(int(nside),iprings[~mask])
vec0=hp.pix2vec(int(nside),0)
angles=np.arccos(np.dot(np.transpose(vec0),vecs))
themap=output_map[~mask]
clf()
plot(angles*180/np.pi,themap,'b.')
xlim(0,10)
yscale('log')

from Homogeneity import fitting

def radgauss(x,pars):
    return(pars[0]/(2*np.pi*pars[1]**2)*np.exp(-0.5*(x/pars[1])**2))

angmax=1.6
ok=angles*180/np.pi < angmax
clf()
plot(angles*180/np.pi,themap,'b,')
#plot(angles[ok]*180/np.pi,themap[ok],'r,')
yscale('log')
xlabel('Angle w.r.t. point source position (deg.)')
ylabel('Map value')
bla=fitting.dothefit(angles[ok]*180/np.pi,themap[ok],themap[ok]*0+0.01,np.array([0.004,0.275]),functname=radgauss,method='mpfit')
title('Fit value : 37.1 arcmin FWHM')
xx=linspace(0,angmax,1000)
plot(xx,radgauss(xx,bla[1]),'r',lw=1)
