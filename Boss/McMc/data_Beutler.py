import numpy as np
import cosmolopy
import scipy
from astropy.io import fits
from pylab import *
from McMc import cosmo_utils
from McMc import mcmc
import astropy.cosmology

mycosmo=mcmc.fidcosmo.copy()
cc=2.99792458e5  # km/s
z=0.106
rs_dv_mes=0.336
drs_dv_mes=0.015
#correction to match Camb rs with E&H rs (Planck2013 XVI)
corr=1.0275    #rs(E&H) ~ 1.0275*rs(camb)

######### Log-Likelihood ####################################################
def log_likelihood(N_nu=mycosmo['N_nu'], Y_He=mycosmo['Y_He'], h=mycosmo['h'], n=mycosmo['n'], omega_M_0=mycosmo['omega_M_0'], omega_b_0=mycosmo['omega_b_0'], omega_lambda_0=mycosmo['omega_lambda_0'], omega_n_0=mycosmo['omega_n_0'], sigma_8=mycosmo['sigma_8'], t_0=mycosmo['t_0'], tau=mycosmo['tau'], w=mycosmo['w'], z_reion=mycosmo['z_reion'],nmassless=mycosmo['Num_Nu_massless'],nmassive=mycosmo['Num_Nu_massive'],library='astropy'):
    cosmo=mcmc.get_cosmology(N_nu,Y_He,h,n,omega_M_0,omega_b_0,omega_lambda_0,omega_n_0,sigma_8,t_0,tau,w,z_reion,nmassless,nmassive,library=library)
    
    if library == 'astropy': cosast=astropy.cosmology.wCDM(cosmo['h']*100,cosmo['omega_M_0'],cosmo['omega_lambda_0'],w0=cosmo['w'])

    try:
        if library == 'cosmolopy':
            #print('log_likelihood Beutler Da: cosmolopy')
            daval=cosmolopy.distance.angular_diameter_distance(z,**cosmo)
        elif library == 'astropy':
            #print('log_likelihood Beutler Da: astropy')
            daval=cosast.angular_diameter_distance(z)
        elif library == 'jc':
            #print('log_likelihood Beutler Da: jc')
            daval=cosmo_utils.angdist(z,**cosmo)
    except:
        daval=-1.
    try:
        if library == 'cosmolopy':
            #print('log_likelihood Beutler Hz: cosmolopy')
            hval=cosmolopy.distance.e_z(z,**cosmo)*cosmo['h']*100
        elif library == 'astropy':
            #print('log_likelihood Beutler Hz: astropy')
            hval=100*cosmo['h']/cosast.inv_efunc(z)
        elif library == 'jc':
            #print('log_likelihood Beutler Hz: jc')
            hval=cosmo_utils.e_z(z,**cosmo)*cosmo['h']*100
    except:
        hval=-1.
    dvval=((1+z)**2*daval**2*cc*z/hval)**(1./3)
    rsval=cosmo_utils.rs_zdrag_fast_camb(**cosmo)*corr
    rs_dv=rsval/dvval
    chi2=(rs_dv-rs_dv_mes)**2/drs_dv_mes**2
    return(-0.5*chi2)
#############################################################################
