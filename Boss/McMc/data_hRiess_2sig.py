import numpy as np
import cosmolopy
import scipy
from astropy.io import fits
from pylab import *
from McMc import cosmo_utils
from McMc import mcmc

mycosmo=mcmc.fidcosmo.copy()

######### Log-Likelihood ####################################################
def log_likelihood(N_nu=np.array(mycosmo['N_nu']), Y_He=np.array(mycosmo['Y_He']), h=np.array(mycosmo['h']), n=np.array(mycosmo['n']), omega_M_0=np.array(mycosmo['omega_M_0']), omega_b_0=np.array(mycosmo['omega_b_0']), omega_lambda_0=np.array(mycosmo['omega_lambda_0']), omega_n_0=np.array(mycosmo['omega_n_0']), sigma_8=np.array(mycosmo['sigma_8']), t_0=np.array(mycosmo['t_0']), tau=np.array(mycosmo['tau']), w=np.array(mycosmo['w']), z_reion=np.array(mycosmo['z_reion']),nmassless=mycosmo['Num_Nu_massless'],nmassive=mycosmo['Num_Nu_massive'],library='astropy'):
    cosmo=mcmc.get_cosmology(N_nu,Y_He,h,n,omega_M_0,omega_b_0,omega_lambda_0,omega_n_0,sigma_8,t_0,tau,w,z_reion,nmassless,nmassive,library=library)
    chi2=(cosmo['h']-0.738)**2/(2*0.024)**2
    return(-0.5*chi2)
#############################################################################

