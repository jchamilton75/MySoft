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

nside=256
input_map=np.zeros(12*nside**2)
input_map[0]=1

#### QUBIC Instrument
kmax = 2
qubic = QubicInstrument('monochromatic,nopol',nside=nside)
pointings = create_random_pointings(1, 20)
pointings[0,:]=[0.1,0,0]


#### configure observation
obs = QubicConfiguration(qubic, pointings)
C = obs.get_convolution_peak_operator()
P = obs.get_projection_peak_operator(kmax=kmax)
H = P * C

# Produce the Time-Ordered data
tod = H(input_map)

plot(tod)
corner = qubic.pack(qubic.detector.corner).view(float).reshape((-1,4,2))

clf()
x=corner[:,0,0]
y=corner[:,0,1]
scatter(x,y,c=tod.ravel(),marker='s',s=60)
colorbar()


