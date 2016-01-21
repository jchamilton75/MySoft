from __future__ import division
import matplotlib.pyplot as mp
import numpy as np
from numpy import pi
import pycamb
from Homogeneity import SplineFitting
import scipy.interpolate as interp

## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
r = 0.05
H0 = 67.04
omegab = 0.022032
omegac = 0.12038
h2 = (H0/100.)**2
scalar_amp = np.exp(3.098)/1.E10
omegav = h2 - omegab - omegac
Omegab = omegab/h2
Omegac = omegac/h2
print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',omegav/h2


######### Make CAMB library
def rcamblib(rvalues,lmaxcamb):
	lll = np.arange(lmaxcamb+1)
	fact = (lll*(lll+1))/(2*np.pi)
	spec = np.zeros((lmaxcamb+1,len(rvalues)))
	i=0
	for r in rvalues:	
		print('i = {0:5.0f} over {1:5.0f}: Calling CAMB with r={2:10.8f}'.format(i,len(rvalues),r))	
		params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,'reion__use_optical_depth':True,'reion__optical_depth':0.0925,'tensor_ratio':r,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}
		T,E,B,X = pycamb.camb(lmaxcamb+1+150,**params)
		B=B[:lmaxcamb+1]/fact
		B[~np.isfinite(B)]=0
		spec[:,i] = B
		i = i + 1

	return [lll, rvalues, spec]



def get_Dlbb_fromlib(lvals, r, lib):
	lll = lib[0]
	bla = interp.RectBivariateSpline(lll,lib[1],lib[2])
	return np.ravel(bla(lll, r))*(lll*(lll+1))/(2*np.pi)





# ##########
# from Cosmo import interpol_camb as ic
# ### test
# rmin = 0.001
# rmax = 1
# nb =100
# lmaxcamb = 300
# rvalues = np.logspace(np.log10(rmin),np.log10(rmax),nb)
# lib = ic.rcamblib(rvalues, lmaxcamb)
# bla = interp.RectBivariateSpline(lib[0],lib[1],lib[2])
# lll = np.arange(1,lmaxcamb+1)
# fact = (lll*(lll+1))/(2*np.pi)

# ther=0.05
# # camb
# params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,'reion__use_optical_depth':True,'reion__optical_depth':0.0925,'tensor_ratio':ther,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}
# %timeit T,E,B,X = pycamb.camb(lmaxcamb+1+150,**params)
# B=B[:lmaxcamb]
# # interp
# %timeit dlint = ic.get_Dlbb_fromlib(lll, ther, lib)

# clf()
# plot(lll, dlint, lw=3)
# plot(lll, B, 'r--',lw=3)
# ##########









