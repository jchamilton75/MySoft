import numpy as np
import cosmolopy
import scipy
from astropy.io import fits
from pylab import *
from McMc import cosmo_utils
from McMc import mcmc
import pickle
import scipy.linalg

#fiducial cosmology
mycosmo=mcmc.fidcosmo.copy()

#restore Planck results for this Open Lambda CDM
pkl_file=open('/Users/hamilton/SDSS/Planck/stats_base_omegak_planck_lowl_lowLike.pkl','rb')
planck=pickle.load(pkl_file)
invcovar=scipy.linalg.inv(planck['covar'])
vals=planck['mean']
sigs=planck['sig']
pkl_file.close()

######### Log-Likelihood ####################################################
def log_likelihood(N_nu=np.array(mycosmo['N_nu']), Y_He=np.array(mycosmo['Y_He']), h=np.array(mycosmo['h']), n=np.array(mycosmo['n']), omega_M_0=np.array(mycosmo['omega_M_0']), omega_b_0=np.array(mycosmo['omega_b_0']), omega_lambda_0=np.array(mycosmo['omega_lambda_0']), omega_n_0=np.array(mycosmo['omega_n_0']), sigma_8=np.array(mycosmo['sigma_8']), t_0=np.array(mycosmo['t_0']), tau=np.array(mycosmo['tau']), w=np.array(mycosmo['w']), z_reion=np.array(mycosmo['z_reion']),nmassless=mycosmo['Num_Nu_massless'],nmassive=mycosmo['Num_Nu_massive'],library='astropy'):
    cosmo=mcmc.get_cosmology(N_nu,Y_He,h,n,omega_M_0,omega_b_0,omega_lambda_0,omega_n_0,sigma_8,t_0,tau,w,z_reion,nmassless,nmassive,library=library)
    
    theobh2=cosmo['omega_b_0']*cosmo['h']**2
    theoch2=(cosmo['omega_M_0']-cosmo['omega_b_0']-cosmo['omega_n_0'])*cosmo['h']**2
    thethetamc=100*cosmo_utils.thetamc(**cosmo)
    theok=cosmo['omega_k_0']
    thevals=[theobh2,theoch2,thethetamc,theok]
    chi2=np.dot(np.dot(thevals-vals,invcovar),thevals-vals)
    #chi2=np.sum((vals-thevals)**2/sigs**2)
    return(-0.5*chi2)
#############################################################################

