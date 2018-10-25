import healpy as hp
import numpy as np
from pylab import *


### Converts a Covariance Matric int a Correlation Matrix
def cov2corr(mat):
    newmat = mat.copy()
    sh = np.shape(mat)
    for i in xrange(sh[0]):
        for j in xrange(sh[1]):
            newmat[i,j] = mat[i,j] / np.sqrt(mat[i,i] * mat[j,j])
    return newmat



def smallhpmap(map, seen):
    ns = hp.npix2nside(len(seen))
    allmap = np.zeros(12*ns**2) + hp.UNSEEN
    allmap[seen] = map
    return allmap







