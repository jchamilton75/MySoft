import pymc
import numpy as np
import cosmolopy
import scipy
from astropy.io import fits
from McMc import cosmo_utils
from pylab import *

############# input data from Nicolas Busca ###################################
chi2file='/Users/hamilton/SDSS/LymanAlpha/Contours_Hrs_da_rs/Fit2DCosmo.fits'
chi2f=fits.open(chi2file)
#chi2f.info()
### Header with info on the min location of chi2
hdr1=chi2f[1].header
nd=hdr1['ndata']
ns=2043
corrective_factor=(ns-1)*1./(ns-nd-2)
chi2min=hdr1['chi2_min']/corrective_factor
aparmin=hdr1['apar_min']
apermin=hdr1['aper_min']
### chi2 array
c=chi2f[3].data
aparvals=c['apar_scan']
apervals=c['aper_scan']
chi2vals=c['chi2_scan']/corrective_factor
mask=apervals>1.24
chi2vals[mask]=np.max(chi2vals)
apar1d=aparvals[:,0]
aper1d=apervals[0,:]

dx=apar1d[1]-apar1d[0]
dy=aper1d[1]-aper1d[0]
proba=np.exp(-chi2vals/2)
proba=proba/np.sum(proba)/dx/dy
clf()
xlabel('alpha par')
ylabel('alpha per')
imshow(transpose(proba),extent=(np.min(apar1d)-dx/2,np.max(apar1d)+dx/2,np.min(aper1d)-dy/2,np.max(aper1d)+dy/2),origin='lower',interpolation='nearest',aspect='auto')
colorbar()
contour(apar1d,aper1d,transpose(chi2vals),levels=chi2min+np.array([2.275,5.99]))
plot(apar1d,aper1d*0+1,'k:')
plot(apar1d*0+1,aper1d,'k:')
plot(aparmin,apermin,'ro')

### fiducial cosmo
zlya=2.46
mycosmo=cosmolopy.fidcosmo.copy()
mycosmo['baryonic_effects']=True
mycosmo['h']=0.7
mycosmo['omega_M_0']=0.27
mycosmo['omega_lambda_0']=0.73
mycosmo['omega_k_0']=0.
mycosmo['w']=-1.0
mycosmo['omega_b_0']=0.0227/mycosmo['h']**2
da_fidu=cosmolopy.distance.angular_diameter_distance(zlya,**mycosmo)
h_fidu=cosmolopy.distance.e_z(zlya,**mycosmo)*mycosmo['h']
rs_fidu=cosmo_utils.rs(**mycosmo)
hrs_fidu=h_fidu*rs_fidu
da_rs_fidu=da_fidu/rs_fidu

X_hrs=apar1d/hrs_fidu
Yda_rs=aper1d*da_rs_fidu
X_hrs_min=aparmin/hrs_fidu
Yda_rs_min=apermin*da_rs_fidu
newdx=dx/hrs_fidu
newdy=dy*da_rs_fidu
proba=proba/np.sum(proba)/newdx/newdy

clf()
xlabel('$1/(H r_s)$')
ylabel('$D_A/r_s$')
imshow(transpose(proba/np.max(proba)),extent=(np.min(X_hrs)-newdx/2,np.max(X_hrs)+newdx/2,np.min(Yda_rs)-newdy/2,np.max(Yda_rs)+newdy/2),origin='lower',interpolation='nearest',aspect='auto')
colorbar()
contour(X_hrs,Yda_rs,transpose(chi2vals),levels=chi2min+np.array([2.275,5.99]))
plot(X_hrs,Yda_rs*0+da_rs_fidu,'k:')
plot(X_hrs*0+1./hrs_fidu,Yda_rs,'k:')
plot(X_hrs_min,Yda_rs_min,'ro')
#############################################################################




######### interpolating function ######################################
proba_interp=scipy.interpolate.interp2d(X_hrs,Yda_rs,np.transpose(proba),copy=True)
#############################################################################


######### priors on unknown parameters ######################################
# no prior on h
h = pymc.Uniform('h',0.,1.,value=0.7)
# Planck prior on h : this is Planck TT + WP + Highl + Planck Lensing: last column of table in page 22 from http://arxiv.org/pdf/1303.5076v1.pdf : Planck 2013 paper XVI
#h = pymc.Normal('h',0.678,1./0.077**2,value=mycosmo['h'])

om = pymc.Uniform('om',0.,2.,value=0.27)

ol = pymc.Uniform('ol',0.,3.,value=0.73)

# Fix ob
ob = pymc.Uniform('ob',0.,0.1,value=0.0463,observed=True)
# Leave ob free
#ob = pymc.Uniform('ob',0.,0.1,value=mycosmo['omega_b_0'])
#############################################################################



# function of parameters that calculates the proba
#@pymc.deterministic
def theproba(h=h,om=om,ol=ol,ob=ob):
    cosmo=cosmolopy.fidcosmo.copy()
    cosmo['h']=h.flatten()[0]
    cosmo['omega_M_0']=om.flatten()[0]
    cosmo['omega_lambda_0']=ol.flatten()[0]
    cosmo['omega_k_0']=1.-om.flatten()[0]-ol.flatten()[0]
    cosmo['omega_b_0']=ob.flatten()[0]
    try:
        daval=cosmolopy.distance.angular_diameter_distance(zlya,**cosmo)
    except ValueError:
        daval=-1.
    try:
        hval=cosmolopy.distance.e_z(zlya,**cosmo)*cosmo['h']
    except ValueError:
        hval=-1.
    rsval=cosmo_utils.rs(**cosmo)
    invhrs=1./(hval*rsval)
    da_rs=daval/rsval
    valprob=proba_interp(invhrs,da_rs)
    #print(h.flatten()[0],om.flatten()[0],ol.flatten()[0],ob.flatten()[0],'   ',invhrs,da_rs,valprob)
    return(valprob)

#clone to be called from outside, only used for debugging
def theproba_ext(h=h,om=om,ol=ol,ob=ob):
    cosmo=cosmolopy.fidcosmo.copy()
    cosmo['h']=h
    cosmo['omega_M_0']=om
    cosmo['omega_lambda_0']=ol
    cosmo['omega_k_0']=1.-om-ol
    cosmo['omega_b_0']=ob
    try:
        daval=cosmolopy.distance.angular_diameter_distance(zlya,**cosmo)
    except ValueError:
        daval=-1.
    try:
        hval=cosmolopy.distance.e_z(zlya,**cosmo)*cosmo['h']
    except ValueError:
        hval=-1.
    rsval=cosmo_utils.rs(**cosmo)
    invhrs=1./(hval*rsval)
    da_rs=daval/rsval
    valprob=proba_interp(invhrs,da_rs)
    #print(h.flatten()[0],om.flatten()[0],ol.flatten()[0],ob.flatten()[0],'   ',invhrs,da_rs,valprob)
    return(valprob,invhrs,da_rs)


@pymc.stochastic(trace=True,observed=True,plot=False)
def likelihood(value=0,h=h,om=om,ol=ol,ob=ob):
    return(np.log(theproba(h=h.flatten(),om=om.flatten(),ol=ol.flatten(),ob=ob.flatten())))












