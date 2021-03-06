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
z=0.35
dv_rs_mes=8.88
ddv_rs_mes=0.17
#correction to match Camb rs with E&H rs (Planck2013 XVI)
corr=1.0275    #rs(E&H) ~ 1.0275*rs(camb)

######### Log-Likelihood ####################################################
def log_likelihood(N_nu=mycosmo['N_nu'], Y_He=mycosmo['Y_He'], h=mycosmo['h'], n=mycosmo['n'], omega_M_0=mycosmo['omega_M_0'], omega_b_0=mycosmo['omega_b_0'], omega_lambda_0=mycosmo['omega_lambda_0'], omega_n_0=mycosmo['omega_n_0'], sigma_8=mycosmo['sigma_8'], t_0=mycosmo['t_0'], tau=mycosmo['tau'], w=mycosmo['w'], z_reion=mycosmo['z_reion'],nmassless=mycosmo['Num_Nu_massless'],nmassive=mycosmo['Num_Nu_massive'],library='astropy'):
    cosmo=mcmc.get_cosmology(N_nu,Y_He,h,n,omega_M_0,omega_b_0,omega_lambda_0,omega_n_0,sigma_8,t_0,tau,w,z_reion,nmassless,nmassive, library=library)
   
    if library == 'astropy': cosast=astropy.cosmology.wCDM(cosmo['h']*100,cosmo['omega_M_0'],cosmo['omega_lambda_0'],w0=cosmo['w'])
    
    try:
        if library == 'cosmolopy':
            #print('log_likelihood DR7 Da: cosmolopy')
            daval=cosmolopy.distance.angular_diameter_distance(z,**cosmo)
        elif library == 'astropy':
            #print('log_likelihood DR7 Da: astropy')
            daval=cosast.angular_diameter_distance(z)
        elif library == 'jc':
            #print('log_likelihood lDR7 Da: jc')
            daval=cosmo_utils.angdist(z,**cosmo)
    except:
        daval=-1.
    try:
        if library == 'cosmolopy':
            #print('log_likelihood DR7 Hz: cosmolopy')
            hval=cosmolopy.distance.e_z(z,**cosmo)*cosmo['h']*100
        elif library == 'astropy':
            #print('log_likelihood DR7 Hz: astropy')
            hval=100*cosmo['h']/cosast.inv_efunc(z)
        elif library == 'jc':
            #print('log_likelihood DR7 Hz: jc')
            hval=cosmo_utils.e_z(z,**cosmo)*cosmo['h']*100
    except:
        hval=-1.
    dvval=((1+z)**2*daval**2*cc*z/hval)**(1./3)
    rsval=cosmo_utils.rs_zdrag_fast_camb(**cosmo)*corr
    dv_rs=dvval/rsval
    chi2=(dv_rs-dv_rs_mes)**2/ddv_rs_mes**2
    return(-0.5*chi2)
#############################################################################
