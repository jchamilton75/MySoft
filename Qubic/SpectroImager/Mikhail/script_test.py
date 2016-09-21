from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
from qubic import create_random_pointings, gal2equ, QubicScene
#from instrument import QubicMultibandInstrument
#from acquisition import QubicMultibandAcquisition

import instrument
import acquisition

nside = 256

center_gal = 0, 90
center = gal2equ(center_gal[0], center_gal[1])

def compute_freq(nu_min, nu_max, delta_nu):
    '''
    Prepare frequency bands parameters
    '''
    _n = np.log(nu_max / nu_min) / np.log(1 + delta_nu)
    Nbfreq = int(np.floor(_n)) + 1
    nus_edge = nu_min * np.logspace(0, _n, Nbfreq,
                                    endpoint=True, base=delta_nu + 1)
    nus = np.array([(nus_edge[i] + nus_edge[i-1]) / 2 for i in range(1, Nbfreq)])
    deltas = np.array([(nus_edge[i] - nus_edge[i-1])  for i in range(1, Nbfreq)])
    Delta = nu_max - nu_min
    Nbbands = len(nus)
    return Nbfreq, nus_edge, nus, deltas, Delta, Nbbands

Nbfreq, nus_edge, nus, deltas, Delta, Nbbands = compute_freq(150. * (1 - 0.25 / 2.),
                                                             150. * (1 + 0.25 / 2.),
                                                             0.1)

p = create_random_pointings(center, 1000, 10)
q = instrument.QubicMultibandInstrument(filter_nus=nus * 1e9, 
                             filter_relative_bandwidths=nus / deltas)
s = QubicScene(nside=nside, kind='I')
a = acquisition.QubicMultibandAcquisition(q, p, s, effective_duration=2)


x0 = np.zeros((2, hp.nside2npix(nside)))
#hp.ang2pix(center)

x0[0, 20] = 1.
x0[1, 125] = 1.

TOD, maps_convolved = a.get_observation(x0)
maps_recon = a.tod2map(TOD)


for i, (inp, rec) in enumerate(zip(maps_convolved, maps_recon)):
    hp.gnomview(inp, rot=center_gal, sub=(2, 2, i + 1), title='Input, $\\nu = ${:.0f}'.format(nus[i]))
    hp.gnomview(rec, rot=center_gal, sub=(2, 2, i + 3), title='Reconstructed, $\\nu = ${:.0f}'.format(nus[i]))

mp.show()