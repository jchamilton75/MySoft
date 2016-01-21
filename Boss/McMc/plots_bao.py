import numpy as np
import cosmolopy
import scipy
from astropy.io import fits
from pylab import *
from McMc import cosmo_utils
from McMc import mcmc
import pickle
import scipy.linalg

nbz=1000
zvals=linspace(0,3,nbz)

#### Lya DR11 
from McMc import data_lyaDR11 as L
reload(L)
print(L.rs_fidu)

######## TESTS #########################
reload(cosmo_utils)
mycosmo=L.mycosmo.copy()
mycosmo['baryonic_effects']=True
mycosmo['h']=0.7
mycosmo['omega_M_0']=0.27
mycosmo['omega_lambda_0']=0.73
mycosmo['omega_k_0']=0.
mycosmo['w']=-1.0
mycosmo['omega_b_0']=0.0227/mycosmo['h']**2
mycosmo['omega_n_0']=0
mycosmo['Num_Nu_massless']=3.046
mycosmo['Num_Nu_massive']=0

print(L.rs_fidu)
print(cosmo_utils.rs(**mycosmo))
print(cosmo_utils.rs_zstar_camb(**mycosmo))
print(cosmo_utils.rs_zdrag_camb(**mycosmo))
print(cosmo_utils.rs_zdrag_fast_camb(**mycosmo))

cosmoplanck=L.mycosmo.copy()
cosmoplanck['h']=0.6704
cosmoplanck['Y_He']=0.247710
obh2=0.022032
onh2=0.000645
och2=0.120376-onh2
cosmoplanck['omega_M_0']=(och2+obh2+onh2)/cosmoplanck['h']**2
cosmoplanck['omega_lambda_0']=1.-cosmoplanck['omega_M_0']
cosmoplanck['omega_k_0']=0
cosmoplanck['omega_b_0']=obh2/cosmoplanck['h']**2
cosmoplanck['omega_n_0']=onh2/cosmoplanck['h']**2
cosmoplanck['n']=0.9619123

print(cosmo_utils.rs_camb(**cosmoplanck))
print(cosmo_utils.rs(**cosmoplanck))
##########################################


tp=L.proba
invh2=np.zeros((len(L.X_hrs),len(L.Yda_rs)))
for i in np.arange(len(L.Yda_rs)): invh2[:,i]=(L.X_hrs*L.rs_fidu)
da2=np.zeros((len(L.X_hrs),len(L.Yda_rs)))
for i in np.arange(len(L.X_hrs)): da2[i,:]=L.Yda_rs*L.rs_fidu

minvhlya=np.sum(invh2*tp)/np.sum(tp)
mdalya=np.sum(da2*tp)/np.sum(tp)
minvh2lya=np.sum(invh2**2*tp)/np.sum(tp)
mda2lya=np.sum(da2**2*tp)/np.sum(tp)
sinvhlya=np.sqrt(minvh2lya-minvhlya**2)
sdalya=np.sqrt(mda2lya-mdalya**2)
mhlya=1./minvhlya
shlya=mhlya*(sinvhlya/minvhlya)

clf()
ylabel('$1/(H r_s)$')
xlabel('$D_A/r_s$')
imshow(L.proba/np.max(L.proba),extent=(np.min(L.Yda_rs)-L.newdy/2,np.max(L.Yda_rs)+L.newdy/2,np.min(L.X_hrs)-L.newdx/2,np.max(L.X_hrs)+L.newdx/2),origin='lower',interpolation='nearest',aspect='auto')
colorbar()
contour(L.Yda_rs,L.X_hrs,L.chi2vals,levels=L.chi2min+np.array([2.275,5.99]))
plot(L.Yda_rs*0+L.da_rs_fidu,L.X_hrs,'w--',lw=2)
plot(L.Yda_rs,L.X_hrs*0+1./L.hrs_fidu,'w--',lw=2)
plot(L.Yda_rs_min,L.X_hrs_min,'ro')

errorbar(mdalya/L.rs_fidu,minvhlya/L.rs_fidu,xerr=sdalya/L.rs_fidu,yerr=sinvhlya/L.rs_fidu,fmt='go')


#### testore Planck LambdaCDM Best fit
rep='/Volumes/Data/ChainsPlanck/PLA/base/planck_lowl_lowLike/'
planck_chains=np.loadtxt(rep+'base_planck_lowl_lowLike_1.txt')
names=np.loadtxt(rep+'base_planck_lowl_lowLike.paramnames',dtype='str',usecols=[0])
planck=dict([names[i],planck_chains[:,i+2]] for i in range(np.size(names)))
sz=len(planck['H0*'])

#### Calculate Da and H for Planck
davals=np.zeros((sz,nbz))
hvals=np.zeros((sz,nbz))
for i in np.arange(sz):
    print(i)
    cosmo=mycosmo.copy()
    cosmo['h']=planck['H0*'][i]/100
    cosmo['omega_M_0']=planck['omegam*'][i]
    cosmo['omega_lambda_0']=planck['omegal*'][i]
    cosmo['omega_k_0']=1.-cosmo['omega_M_0']-cosmo['omega_lambda_0']
    cosmo['w']=-1.
    davals[i,:]=cosmo_utils.angdist(zvals,**cosmo)
    hvals[i,:]=cosmo_utils.e_z(zvals,**cosmo)*cosmo['h']

### modele autre
cosmonew=cosmo.copy()
cosmonew['h']=planck['H0*'][i]/100
cosmonew['omega_M_0']=0.26
cosmonew['omega_k_0']=-0.007
cosmonew['omega_lambda_0']=1-cosmonew['omega_M_0']-cosmonew['omega_k_0']
cosmonew['w']=-1.
dazarb=cosmo_utils.angdist(zvals,**cosmonew)
hzarb=cosmo_utils.e_z(zvals,**cosmonew)*cosmonew['h']


    
mda=np.zeros(nbz)
sda=np.zeros(nbz)
mh=np.zeros(nbz)
sh=np.zeros(nbz)
for i in np.arange(nbz):
    mda[i]=np.mean(davals[:,i])
    sda[i]=np.std(davals[:,i])
    mh[i]=np.mean(hvals[:,i])
    sh[i]=np.std(hvals[:,i])

clf()
subplot(2,1,1)
plot(zvals,mda,'k')
plot(zvals,mda+sda,'k--')
plot(zvals,mda-sda,'k--')
xlabel('z')
ylabel('$D_A(z)$')
errorbar(L.zlya,mdalya,yerr=sdalya,fmt='ro')
plot(zvals,dazarb,'r')
subplot(2,1,2)
plot(zvals,mh/(1+zvals),'k')
plot(zvals,(mh+sh)/(1+zvals),'k--')
plot(zvals,(mh-sh)/(1+zvals),'k--')
xlabel('z')
ylabel('$H(z)/(1+z)$')
errorbar(L.zlya,mhlya/(1+L.zlya),yerr=shlya/(1+L.zlya),fmt='ro')
plot(zvals,hzarb/(1+zvals),'r')



