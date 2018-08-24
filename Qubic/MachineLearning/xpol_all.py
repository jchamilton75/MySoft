from __future__ import division, print_function

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import pi
from qubic import (
    apodize_mask, equ2gal, plot_spectra, read_spectra, semilogy_spectra, Xpol)
from qubic.utils import progress_bar
from pysimulators import FitsArray



ns = 64
nmax = 100000
mask = FitsArray('mymask_ns{}.fits'.format(ns))
mymaps = FitsArray('mymaps_ns{}.fits'.format(ns))[0:nmax,:]
expcls = FitsArray('expcls_ns{}.fits'.format(ns))[0:nmax,:]
mycls = FitsArray('mycls_ns{}.fits'.format(ns))[0:nmax,:]

okpix = mask == 1

# Initialize Xpol
np.random.seed(0)
lmin = 1
lmax = 2*ns-1
delta_ell = 16
xpol = Xpol(mask, lmin, lmax, delta_ell)
ell_binned = xpol.ell_binned
nbins = len(ell_binned)

allclsout = np.zeros((nmax, nbins))
allclsin = np.zeros((nmax, nbins))
bar = progress_bar(nmax)
for i in xrange(nmax):
	bar.update()
	allclsin[i,:] = xpol.bin_spectra(mycls[i,:])
	maps = np.zeros((3,12*ns**2))
	maps[0,okpix] = mymaps[i,:]
	allclsout[i,:] = xpol.get_spectra(maps)[1][0]

clf()
num = np.random.randint(nmax)
plot(ell_binned*(ell_binned+1)*allclsin[num,:], label='in')
plot(ell_binned*(ell_binned+1)*allclsout[num,:],label='out')
title(num)
legend()

FitsArray(allclsout).save('myXpolCl_ns{}.fits'.format(ns))
FitsArray(ell_binned).save('myell_binned_ns{}.fits'.format(ns))
