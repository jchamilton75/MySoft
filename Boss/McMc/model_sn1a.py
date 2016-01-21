import pymc
import numpy as np
import cosmolopy
import astropy.cosmology
import random
import scipy

#read SNIa data: these assume h=0.7 but this is actually irrelevant for analysis as h also appears in luminosity distance, so finally, cosmology does not depend upon h.
filename='/Users/hamilton/Python/Boss/McMc/Data/SNData_selected_noascii.txt'
zsn,musn,dmusn=np.loadtxt(filename,usecols=(0, 7, 8),unpack=True)
tau = 1./dmusn**2
x = zsn

#zvals=np.linspace(0.01,2,1000)
#mycosmo=cosmolopy.fidcosmo.copy()
#dlum=cosmolopy.distance.luminosity_distance(zvals,**mycosmo)*1e6
#musn1a=5*np.log10(dlum)-5
#
#clf()
#errorbar(zsn,musn,yerr=dmusn,fmt='ro')
#xscale('log')
#plot(zvals,musn1a)



#priors on unknown parameters
h = pymc.Uniform('h',0.,1.,value=0.7)
om = pymc.Uniform('om',0.,2.,value=0.3)
ol = pymc.Uniform('ol',0.,3.,value=0.7)

#function of parameters that describes the data
@pymc.deterministic
def themusn1a(x=x,h=h,om=om,ol=ol):
    cosmo=cosmolopy.fidcosmo.copy()
    cosmo['h']=h
    cosmo['omega_M_0']=om
    cosmo['omega_lambda_0']=ol
    cosmo['omega_k_0']=1.-om-ol
    dlum=cosmolopy.distance.luminosity_distance(x,**cosmo)*1e6
    if np.min(dlum) > 0:
        return(5*np.log10(dlum)-5+5*np.log10(h/0.7))
    else:
        return(np.zeros(x.size)-1)



# Astropy is very slow... too bad because it has many nice features (alternative models)
#@pymc.deterministic
#def themusn1a(x=x,h=h,om=om,ol=ol):
#    thez=np.linspace(np.min(x),np.max(x),50)
#    cosmo=astropy.cosmology.LambdaCDM(H0=h*100,Om0=om,Ode0=ol)
#    #thedlum=astropy.cosmology.luminosity_distance(thez,cosmo=cosmo)*1e6
#    dlum=scipy.interp(x,thez,thedlum)
#    if np.min(dlum) > 0:
#        return(5*np.log10(dlum)-5+5*np.log10(h/0.7))
#    else:
#        return(np.zeros(x.size)-1)


#statistics of the data
d=pymc.Normal('d',mu=themusn1a,tau=tau,value=musn,observed=True)









