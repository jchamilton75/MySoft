from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import pi
import pycamb
from qubic import QubicInstrument
from scipy.constants import c
from Sensitivity import qubic_sensitivity
from Homogeneity import SplineFitting
from pysimulators import FitsArray
from Cosmo import interpol_camb as ic
from Sensitivity import dualband_lib as db
import pymc
import pickle
from matplotlib import *
from pylab import *
from scipy.ndimage import gaussian_filter1d
import scipy
from McMc import mcmc


####### log likelihood #############
def log_likelihood(r=0, dldust_80_353=13.4, alphadust=-2.42, betadust=1.59, Tdust=19.6, data=None):
	rval = r
	dpars = np.array([dldust_80_353, alphadust, betadust, Tdust])

	specin = data['specin']
	instrument_info = data['inst_info']
	all_neq_nh = data['all_neq_nh']
	camblib = data['camblib']
	covmat, covmatnoise, covmatsample, allspec, allvars, ClBBcosmo, all_neq_nh = db.get_multiband_covariance(instrument_info, rval, dustParams=dpars, all_neq_nh=all_neq_nh, camblib=camblib)
	invcov = np.linalg.inv(covmat + np.random.randn(covmat.shape[0], covmat.shape[1])*1e-10)
	resid = specin - allspec
	chi2 = np.sum(np.dot(resid,invcov)*resid)
	chi2 = chi2 + (dldust_80_353 - 13.4)**2/0.26**2   #From Planck XXX
	chi2 = chi2 + (alphadust +2.42)**2/0.02**2        #From Planck XXX
	chi2 = chi2 + (betadust - 1.59)**2/0.11**2        #From Planck XXX section 6.2
	chi2 = chi2 + (Tdust - 19.6)**2/0.8**2            #From Planck XXII fig 2
	if np.isnan(chi2): chi2=1e10
	return -0.5 * chi2



def generic_model(data, variables=['r','dldust_80_353'], default=None):
	### Set default
	if default is None:
		dldust_80_353 = 13.4
		alphadust = -2.42
		betadust = 1.59
		Tdust = 19.6
		thervalue = 0.
		default = np.array([thervalue, dldust_80_353, alphadust, betadust, Tdust])

	### define distributions
	r = pymc.Uniform('r',0., 1., value=default[0], observed = 'r' not in variables)
	dldust_80_353 = pymc.Uniform('dldust_80_353',0, 30, value=default[1], observed = 'dldust_80_353' not in variables)
	alphadust = pymc.Uniform('alphadust',-4, -1, value=default[2], observed = 'alphadust' not in variables)
	betadust = pymc.Uniform('betadust',0.5, 2.5, value=default[3], observed = 'betadust' not in variables)
	Tdust = pymc.Uniform('Tdust',0., 50., value=default[4], observed = 'Tdust' not in variables)

	### Lo-Likelihood function
	@pymc.stochastic(trace=True,observed=True,plot=False)
	def loglikelihood(value=0, r=r, dldust_80_353=dldust_80_353, alphadust=alphadust, betadust=betadust, Tdust=Tdust, data=data):
		ll=log_likelihood(r=r, dldust_80_353=dldust_80_353, alphadust=alphadust, betadust=betadust, Tdust=Tdust, data=data)
		return(ll)
	return(locals())



def readchains(filename):
    pkl_file=open(filename,'rb')
    data=pickle.load(pkl_file)
    pkl_file.close()
    print(data.keys())
    bla={}
    dk=data.keys()
    for kk in data.keys(): 
        if ((kk != '_state_') and (kk != 'deviance')):
            bla[kk]=data[kk][0]
    return(bla)

