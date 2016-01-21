from __future__ import division
from pyoperators import pcg
from pysimulators import profile
from qubic import (
    create_random_pointings, equ2gal, QubicAcquisition, PlanckAcquisition,
    QubicPlanckAcquisition)
from qubic.data import PATH
from qubic.io import read_map
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np

nside = 128
racenter = 0.0      # deg
deccenter = -57.0   # deg


fact = [1., 10]#, 10.]#, 100., 1000.]
sols = []


for i in xrange(len(fact)):
    #fix random seed:
    np.random.seed(42)

    if nside==256:
        sky = read_map(PATH + 'syn256_pol.fits')
    else:
        sky256 = read_map(PATH + 'syn256_pol.fits')
        skyI = hp.ud_grade(sky256[:,0], nside)
        skyQ = hp.ud_grade(sky256[:,1], nside)
        skyU = hp.ud_grade(sky256[:,2], nside)
        sky = np.array([skyI, skyQ, skyU]).T

    sampling = create_random_pointings([racenter, deccenter], 1000, 10)
    detector_nep = 4.7e-17/np.sqrt(365*86400 / len(sampling)*sampling.period)/fact[i]

    acq_qubic = QubicAcquisition(150, sampling, nside=nside,
                                 detector_nep=detector_nep)
    convolved_sky = acq_qubic.instrument.get_convolution_peak_operator()(sky)
    acq_planck = PlanckAcquisition(150, acq_qubic.scene, true_sky=convolved_sky)
    acq_fusion = QubicPlanckAcquisition(acq_qubic, acq_planck)

    H = acq_fusion.get_operator()
    invntt = acq_fusion.get_invntt_operator()
    obs = acq_fusion.get_observation()

    A = H.T * invntt * H
    b = H.T * invntt * obs

    solution_fusion = pcg(A, b, disp=True, maxiter=1000000, tol=1e-5)
    sols.append(solution_fusion)



# some display
center = equ2gal(racenter, deccenter)

def display(input, msg, iplot=1, reso=5, Trange=[100, 5, 5], xsize=800, subx=3, suby=3):
    out = []
    for i, (kind, lim) in enumerate(zip('IQU', Trange)):
        map = input[..., i]
        out += [hp.gnomview(map, rot=center, reso=reso, xsize=xsize, min=-lim,
                            max=lim, title=msg + ' ' + kind,
                            sub=(suby, subx, iplot + i), return_projected_map=True)]
    return out



reso=10
xsize=200
mp.figure()
mp.clf()
Trange=[0.1,0.1,0.1]
for i in xrange(len(sols)):
    solution_fusion = sols[i]
    res_fusion = display(solution_fusion['x'] - convolved_sky,
                     'Difference map', subx=3, suby=len(sols), iplot=3*i+1, reso=reso, Trange=Trange, xsize=xsize)


reso=15
xsize=200
clf()
Trange=[0.01,0.01,0.01]
for i in xrange(len(sols)):
    solution_fusion = sols[i]
    solution_fusion6 = sols6[i]
    res_fusion = display(solution_fusion['x'] - solution_fusion6['x'],
                     'Difference map', subx=3, suby=len(sols), iplot=3*i+1, reso=reso, Trange=Trange, xsize=xsize)


