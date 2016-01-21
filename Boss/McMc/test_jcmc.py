from pylab import *
import numpy as np
import cosmolopy



mycosmo=cosmolopy.fidcosmo.copy()
mycosmo['baryonic_effects']=True


filename='/Users/hamilton/SDSS/SNIa/SNData_selected_noascii.txt'
zsn,musn,dmusn=np.loadtxt(filename,usecols=(0, 7, 8),unpack=True)

zvals=np.linspace(0.01,2,1000)
dlum=cosmolopy.distance.luminosity_distance(zvals,**mycosmo)*1e6
musn1a=5*np.log10(dlum)-5

clf()
errorbar(zsn,musn,yerr=dmusn,fmt='ro')
xscale('log')
plot(zvals,musn1a)

def themusn1a(x,pars):
    cosmo=cosmolopy.fidcosmo.copy()
    if min(pars) <= 0:
        return(np.zeros(x.size)-1)
    cosmo['h']=pars[0]
    cosmo['omega_M_0']=pars[1]
    cosmo['omega_lambda_0']=pars[2]
    dlum=cosmolopy.distance.luminosity_distance(x,**cosmo)*1e6
    if min(dlum) > 0:
        return(5*np.log10(dlum)-5)
    else:
        return(np.zeros(x.size)-1)

from Homogeneity import fitting

### ne marche pas
a=fitting.dothefit(zsn,musn,dmusn,np.array([0.7,0.3,0.7]),themusn1a,method='mpfit')

### OK
a=fitting.dothefit(zsn,musn,dmusn,np.array([0.7,0.3,0.7]),themusn1a,method='minuit')

### ne marche pas...
a=fitting.dothefit(zsn,musn,dmusn,np.array([0.7,0.3,0.7]),themusn1a,method='mcmc')
chain=a[0]
clf()
ylim(0,2)
xlim(0,1)
plot(chain[:,1],chain[:,2],',')



#### essai avec pymc



