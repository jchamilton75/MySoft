import pymc
import numpy as np
import cosmolopy
import astropy.cosmology
import random
import scipy
from pylab import *
from McMc import mcmc

############# input SNIa data #############################################
#read SNIa data: these assume h=0.7 but this is actually irrelevant for analysis as h also appears in luminosity distance, so finally, cosmology does not depend upon h.
filename='/Users/hamilton/Python/Boss/McMc/Data/SNData_selected_noascii.txt'
zsn,musn,dmusn=np.loadtxt(filename,usecols=(0, 7, 8),unpack=True)
tau = 1./dmusn**2
x = zsn

# do a plot
zvals=np.linspace(0.01,2,1000)
mycosmo=cosmolopy.fidcosmo.copy()
dlum=cosmolopy.distance.luminosity_distance(zvals,**mycosmo)*1e6
musn1a=5*np.log10(dlum)-5
clf()
xlabel('Redshift')
ylabel('Distance Modulus')
errorbar(zsn,musn,yerr=dmusn,fmt='ro')
xscale('log')
plot(zvals,musn1a)
#############################################################################


########## function of parameters that describes the data ###################
def themusn1a(z,**cosmo):
    try:
        dlum=cosmolopy.distance.luminosity_distance(z,**cosmo)*1e6
    except ValueError:
        dlum=1e-10
    if np.min(dlum) > 0:
        mu=5*np.log10(dlum)-5+5*np.log10(cosmo['h']/0.7)
    else:
        mu=np.zeros(z.size)-1
    return(mu)
#############################################################################


######### Log-Likelihood ####################################################
def log_likelihood(N_nu=mycosmo['N_nu'], Y_He=mycosmo['Y_He'], h=mycosmo['h'], n=mycosmo['n'], omega_M_0=mycosmo['omega_M_0'], omega_b_0=mycosmo['omega_b_0'], omega_lambda_0=mycosmo['omega_lambda_0'], omega_n_0=mycosmo['omega_n_0'], sigma_8=mycosmo['sigma_8'], t_0=mycosmo['t_0'], tau=mycosmo['tau'], w=mycosmo['w'], z_reion=mycosmo['z_reion']):
    cosmo=mcmc.get_cosmology(N_nu,Y_He,h,n,omega_M_0,omega_b_0,omega_lambda_0,omega_n_0,sigma_8,t_0,tau,w,z_reion)
    
    chi2=np.sum((musn-themusn1a(zsn,**cosmo))**2/dmusn**2)
    return(-0.5*chi2)
#############################################################################





