import numpy as np
import cosmolopy
import scipy
from astropy.io import fits
from pylab import *
from McMc import cosmo_utils
from McMc import mcmc

mycosmo=cosmolopy.fidcosmo.copy()
obh2=0.02205
sobh2=0.00028
och2=0.1199
soch2=0.0027
thetamc=0.01*1.04131
sthetamc=0.01*0.00063

######### Log-Likelihood ####################################################
def log_likelihood(N_nu=np.array(mycosmo['N_nu']), Y_He=np.array(mycosmo['Y_He']), h=np.array(mycosmo['h']), n=np.array(mycosmo['n']), omega_M_0=np.array(mycosmo['omega_M_0']), omega_b_0=np.array(mycosmo['omega_b_0']), omega_lambda_0=np.array(mycosmo['omega_lambda_0']), omega_n_0=np.array(mycosmo['omega_n_0']), sigma_8=np.array(mycosmo['sigma_8']), t_0=np.array(mycosmo['t_0']), tau=np.array(mycosmo['tau']), w=np.array(mycosmo['w']), z_reion=np.array(mycosmo['z_reion']),library='astropy'):
    cosmo=mcmc.get_cosmology(N_nu,Y_He,h,n,omega_M_0,omega_b_0,omega_lambda_0,omega_n_0,sigma_8,t_0,tau,w,z_reion,library=library)
    
    theobh2=cosmo['omega_b_0']*cosmo['h']**2
    theoch2=(cosmo['omega_M_0']-cosmo['omega_b_0']-cosmo['omega_n_0'])*cosmo['h']**2
    thethetamc=cosmo_utils.thetamc(**cosmo)
    chi2=(theobh2-obh2)**2/sobh2**2+(theoch2-och2)**2/soch2**2+(thethetamc-thetamc)**2/sthetamc**2
    return(-0.5*chi2)
#############################################################################

