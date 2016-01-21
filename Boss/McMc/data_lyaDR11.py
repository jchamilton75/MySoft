import numpy as np
import cosmolopy
import scipy
from astropy.io import fits
from pylab import *
from McMc import cosmo_utils
from McMc import mcmc
import astropy.cosmology

np.seterr(all='ignore')

############# input data from Nicolas Busca ###################################
# Old file from Nicolas
#chi2file='/Users/hamilton/SDSS/LymanAlpha/Contours_Hrs_da_rs/Fit2DCosmo.fits'
# New file from Nicolas 13/11/2013
#chi2file='/Users/hamilton/SDSS/LymanAlpha/Contours_Hrs_da_rs/FT_dr11_vac_dla_nobal_rr_np3_nb3_t123456.fits'
# New file by NG Busca on feb 10 2014
chi2file='/Users/hamilton/SDSS/LymanAlpha/Contours_Hrs_da_rs/FT_dr11_vac_dla_nobal_final_scan.fits'
chi2f=fits.open(chi2file)
#chi2f.info()
### Header with info on the min location of chi2
hdr1=chi2f[1].header
corrective_factor=1
chi2min=hdr1['chi2_min']/corrective_factor
aparmin=hdr1['apar_min']
apermin=hdr1['aper_min']
### chi2 array
c=chi2f[3].data
aparvals=c['apar_scan']
apervals=c['aper_scan']
chi2vals=c['chi2_scan']/corrective_factor-chi2min
mask=apervals>1.24
chi2vals[mask]=np.max(chi2vals)
apar1d=aparvals[:,0]
aper1d=apervals[0,:]

#####
#clf()
#dx=aper1d[1]-aper1d[0]
#dy=apar1d[1]-apar1d[0]
#imshow(chi2vals-np.min(chi2vals),interpolation='nearest',extent=(np.min(aper1d)-dx/2,np.max(aper1d)+dx/2,np.min(apar1d)-dy/2,np.max(apar1d)+dy/2),origin='lower')
#colorbar()
#contour(apar1d,aper1d,chi2vals,levels=chi2min+np.array([2.275,5.99]))
#xlabel('alpha perp (Da/rs)')
#ylabel('alpha par (H.rs)')
#plot(aper1d,apar1d*0+1,'g--',lw=3)
#plot(aper1d*0+1,apar1d,'g--',lw=3)


#####
dx=aper1d[1]-aper1d[0]
dy=apar1d[1]-apar1d[0]
proba=np.nan_to_num(np.exp(-chi2vals/2))
proba=proba/np.sum(proba)/dx/dy
#clf()
#xlabel('alpha perp (Da/rs)')
#ylabel('alpha par (H.rs)')
#xlim(0.8,1.2)
#ylim(0.7,1.25)
#imshow(proba,extent=(np.min(aper1d)-dx/2,np.max(aper1d)+dx/2,np.min(apar1d)-dy/2,np.max(apar1d)+dy/2),origin='lower',interpolation='nearest',aspect='auto')
#colorbar()
#contour(aper1d,apar1d,chi2vals,levels=chi2min+np.array([2.275,5.99]))
#plot(aper1d,apar1d*0+1,'g--',lw=3)
#plot(aper1d*0+1,apar1d,'g--',lw=3)
#plot(apermin,aparmin,'ro')

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
mycosmo['omega_n_0']=0
mycosmo['Num_Nu_massless']=3.046
mycosmo['Num_Nu_massive']=0
da_fidu=cosmolopy.distance.angular_diameter_distance(zlya,**mycosmo)
h_fidu=cosmolopy.distance.e_z(zlya,**mycosmo)*mycosmo['h']
rs_fidu=cosmo_utils.rs_zdrag_fast_camb(**mycosmo)
hrs_fidu=h_fidu*rs_fidu
da_rs_fidu=da_fidu/rs_fidu

X_hrs=apar1d/hrs_fidu
Yda_rs=aper1d*da_rs_fidu
X_hrs_min=aparmin/hrs_fidu
Yda_rs_min=apermin*da_rs_fidu
newdx=dx/hrs_fidu
newdy=dy*da_rs_fidu
proba=proba/np.sum(proba)/newdx/newdy

#clf()
#xlabel('$1/(H r_s)$')
#ylabel('$D_A/r_s$')
#imshow(transpose(proba/np.max(proba)),extent=(np.min(X_hrs)-newdx/2,np.max(X_hrs)+newdx/2,np.min(Yda_rs)-newdy/2,np.max(Yda_rs)+newdy/2),origin='lower',interpolation='nearest',aspect='auto')
#colorbar()
#contour(X_hrs,Yda_rs,transpose(chi2vals),levels=chi2min+np.array([2.275,5.99]))
#plot(X_hrs,Yda_rs*0+da_rs_fidu,'k:')
#plot(X_hrs*0+1./hrs_fidu,Yda_rs,'k:')
#plot(X_hrs_min,Yda_rs_min,'ro')

#clf()
#ylabel('$1/(H r_s)$')
#xlabel('$D_A/r_s$')
#imshow(proba/np.max(proba),extent=(np.min(Yda_rs)-newdy/2,np.max(Yda_rs)+newdy/2,np.min(X_hrs)-newdx/2,np.max(X_hrs)+newdx/2),origin='lower',interpolation='nearest',aspect='auto')
#colorbar()
#contour(Yda_rs,X_hrs,chi2vals,levels=chi2min+np.array([2.275,5.99]))
#plot(Yda_rs*0+da_rs_fidu,X_hrs,'g--',lw=3)
#plot(Yda_rs,X_hrs*0+1./hrs_fidu,'g--',lw=3)
#plot(Yda_rs_min,X_hrs_min,'ro')
#stop
#############################################################################


######### interpolating function ############################################
likelihood_interp=scipy.interpolate.interp2d(X_hrs,Yda_rs,np.transpose(proba),copy=True)
#############################################################################



######### Log-Likelihood ####################################################
def log_likelihood(N_nu=np.array(mycosmo['N_nu']), Y_He=np.array(mycosmo['Y_He']), h=np.array(mycosmo['h']), n=np.array(mycosmo['n']), omega_M_0=np.array(mycosmo['omega_M_0']), omega_b_0=np.array(mycosmo['omega_b_0']), omega_lambda_0=np.array(mycosmo['omega_lambda_0']), omega_n_0=np.array(mycosmo['omega_n_0']), sigma_8=np.array(mycosmo['sigma_8']), t_0=np.array(mycosmo['t_0']), tau=np.array(mycosmo['tau']), w=np.array(mycosmo['w']), z_reion=np.array(mycosmo['z_reion']),nmassless=mycosmo['Num_Nu_massless'],nmassive=mycosmo['Num_Nu_massive'],library='astropy'):
    cosmo=mcmc.get_cosmology(N_nu,Y_He,h,n,omega_M_0,omega_b_0,omega_lambda_0,omega_n_0,sigma_8,t_0,tau,w,z_reion,nmassless,nmassive,library=library)

    #print(cosmo['h'],cosmo['omega_M_0'],cosmo['omega_lambda_0'],cosmo['omega_M_0']+cosmo['omega_lambda_0'],cosmo['omega_k_0'],cosmo['w'])

    if library == 'astropy': cosast=astropy.cosmology.wCDM(cosmo['h']*100,cosmo['omega_M_0'],cosmo['omega_lambda_0'],w0=cosmo['w'])
    
    try:
        if library == 'cosmolopy':
            #print('log_likelihood lyaDR11 Da: cosmolopy')
            daval=cosmolopy.distance.angular_diameter_distance(zlya,**cosmo)
        elif library == 'astropy':
            #print('log_likelihood lyaDR11 Da: astropy')
            daval=cosast.angular_diameter_distance(zlya)
        elif library == 'jc':
            #print('log_likelihood lyaDR11 Da: jc')
            daval=cosmo_utils.angdist(zlya,**cosmo)
    except:
        daval=-1.
    try:
        if library == 'cosmolopy':
            #print('log_likelihood lyaDR11 Hz: cosmolopy')
            hval=cosmolopy.distance.e_z(zlya,**cosmo)*cosmo['h']
        elif library == 'astropy':
            #print('log_likelihood lyaDR11 Hz: astropy')
            hval=cosmo['h']/cosast.inv_efunc(zlya)
        elif library == 'jc':
            #print('log_likelihood lyaDR11 Hz: jc')
            hval=cosmo_utils.e_z(zlya,**cosmo)*cosmo['h']
    except:
        hval=-1.

    rsval=cosmo_utils.rs_zdrag_fast_camb(**cosmo)
    invhrs=1./(hval*rsval)
    da_rs=daval/rsval
    valprob=likelihood_interp(invhrs,da_rs)
    return(np.log(valprob))
#############################################################################




