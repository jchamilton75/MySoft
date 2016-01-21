## use with EPD
from __future__ import division
import numpy as np
import pymc
from McMc import mcmc
from Homogeneity import fitting
import pycamb

################# BICEP2 Data
lmin, lcenter, lmax, TT, TE, EE, BB, TB, EB, dTT, dTE, dEE, dBB, dTB, dEB = np.loadtxt('/Users/hamilton/CMB/Interfero/Bicep2/B2_3yr_bandpowers_20140314.txt').T
lll, clTTl, clTEl, clEEl, clBBl, clTBl, clEBl = np.loadtxt('/Users/hamilton/CMB/Interfero/Bicep2/B2_3yr_camb_planck_lensed_uK_20140314.txt')[0:1000,:].T
ll, clTT, clTE, clEE, clBB, clTB, clEB = np.loadtxt('/Users/hamilton/CMB/Interfero/Bicep2/B2_3yr_camb_planck_withB_uK_20140314.txt')[0:1000,:].T
lminthl, lthl, lmaxthl, clthTTl, clthTEl, clthEEl, clthBBl, clthTBl, clthEBl = np.loadtxt('/Users/hamilton/CMB/Interfero/Bicep2/B2_3yr_cl_expected_lensed_20140314.txt').T
lminth, lth, lmaxth, clthTT, clthTE, clthEE, clthBB, clthTB, clthEB = np.loadtxt('/Users/hamilton/CMB/Interfero/Bicep2/B2_3yr_cl_expected_withB_20140314.txt').T


clf()
xlim(0,500)
ylim(-0.01,0.055)
errorbar(lcenter,BB,fmt='ro',xlolims=lmin,xuplims=lmax,yerr=dBB)
plot(lll,clBBl,'g',alpha=0.5)
plot(ll,clBB*2,'b',alpha=0.5)
plot(ll,clBBl+clBB*2,'r')

#### adding noise residuals and try fitting
def functsum(x,pars):
	bla = pars[0] * np.interp(x, ll, clBB) + pars[1] * np.interp(x, ll, clBBl) + pars[2]**2 * x * (x + 1) / 2 / np.pi
	return bla

def functsum2(x,pars):
	bla = pars[0] * np.interp(x, ll, clBB) + pars[1] * np.interp(x, ll, clBBl)
	return bla


guess = [2., 1., np.sqrt(1e6)]
fit = fitting.dothefit(lcenter, BB, dBB, guess, functname = functsum, method = 'minuit')
res = fit[1]

guess2 = [2., 1.]
fit2 = fitting.dothefit(lcenter, BB, dBB, guess2, functname = functsum2, method = 'minuit')
res2 = fit2[1]


#### adding noise
clnoise = res[2]**2
pownoise = clnoise*ll*(ll+1)/2/np.pi
clf()
xlim(0,500)
ylim(-0.01,0.055)
errorbar(lcenter,BB,fmt='ro',xlolims=lmin,xuplims=lmax,yerr=dBB)
plot(lll,clBBl * res[1],'g',alpha=0.5)
plot(ll,clBB * res[0],'b',alpha=0.5)
plot(ll,pownoise,'k',alpha=0.5)
plot(ll, res[1] * clBBl + clBB * res[0] + pownoise,'r')

plot(lll,clBBl * res2[1],'g--',alpha=0.5)
plot(ll,clBB * res2[0],'b--',alpha=0.5)
plot(ll, res2[1] * clBBl + clBB * res2[0],'r--')


######################### Get B power spectrum from Camb
###### Input Power spectrum ###################################
################# Input Power spectrum ###################################
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
H0 = 67.04
omegab = 0.022032
omegac = 0.12038
h2 = (H0/100.)**2
scalar_amp = np.exp(3.098)/1.E10
omegav = h2 - omegab - omegac
Omegab = omegab/h2
Omegac = omegac/h2
rvalue = 0.2
print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',omegav/h2

params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':rvalue,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False,
         'tensor_index':-rvalue/8}

params_l = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
        	 'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
        	 'tensor_ratio':rvalue,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True,
        	 'tensor_index':-rvalue/8}

lmax = 1000
ell = np.arange(1,600)


Tprim,Eprim,Bprim,Xprim = pycamb.camb(lmax+1,**params)
Tprim = Tprim[0:599]
Eprim = Eprim[0:599]
Bprim = Bprim[0:599]
Xprim = Xprim[0:599]
Tpl,Epl,Bpl,Xpl = pycamb.camb(lmax+1,**params_l)
Tpl = Tpl[0:599]
Epl = Epl[0:599]
Bpl = Bpl[0:599]
Xpl = Xpl[0:599]
Bl = Bpl - Bprim

clf()
plot(ell,Bprim,label='Primordial $C_\ell^{BB}$'+'$ (r={0:.2f})$'.format(rvalue))
plot(ell,Bl,label='Lensing $C_\ell^{BB}$')
plot(ell,Bpl,label='Total $C_\ell^{BB}$'+'$ (r={0:.2f})$'.format(rvalue))
yscale('log')
xlim(0,600)
ylim(0.0005,0.1)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$'+'    '+'$[\mu K]^2$ ')
legend(loc='upper left',frameon=False)




##### Take one model and calculate bins and error bars from both noise and sample variance
def th_errors(ell, cell, fsky, mukarcmin, bl_or_fwhmdeg, deltal=1):
	if np.array(bl_or_fwhmdeg).size == 1:
		fwhmdeg = bl_or_fwhmdeg
		bl = exp(-0.5*ell*(ell+1)*np.radians(fwhmdeg/2.35)**2)
	else:
		bl = bl_or_fwhmdeg
	factor = np.sqrt(2./((2*ell+1)*fsky*deltal))
	noiseterm = factor*2*(np.radians(mukarcmin/60))**2/bl**2
	sampleterm = factor*cell
	return [sampleterm, noiseterm, sampleterm+noiseterm]

def th_errors_bins(ellcenter, ell, cell, fsky, mukarcmin, bl_or_fwhmdeg, deltal=1):
	st,nt,tt = th_errors(ell, cell, fsky, mukarcmin, bl_or_fwhmdeg, deltal=deltal)
	ist = np.interp(ellcenter,ell,st)
	int = np.interp(ellcenter,ell,nt)
	itt = np.interp(ellcenter,ell,tt)
	return [ist, int, itt]


### Check with BICEP2
lmin, lcenter, lmax, TT, TE, EE, BB, TB, EB, dTT, dTE, dEE, dBB, dTB, dEB = np.loadtxt('/Users/hamilton/CMB/Interfero/Bicep2/B2_3yr_bandpowers_20140314.txt').T
lll, clTTl, clTEl, clEEl, clBBl, clTBl, clEBl = np.loadtxt('/Users/hamilton/CMB/Interfero/Bicep2/B2_3yr_camb_planck_lensed_uK_20140314.txt')[0:1000,:].T
ll, clTT, clTE, clEE, clBB, clTB, clEB = np.loadtxt('/Users/hamilton/CMB/Interfero/Bicep2/B2_3yr_camb_planck_withB_uK_20140314.txt')[0:1000,:].T
lminthl, lthl, lmaxthl, clthTTl, clthTEl, clthEEl, clthBBl, clthTBl, clthEBl = np.loadtxt('/Users/hamilton/CMB/Interfero/Bicep2/B2_3yr_cl_expected_lensed_20140314.txt').T
lminth, lth, lmaxth, clthTT, clthTE, clthEE, clthBB, clthTB, clthEB = np.loadtxt('/Users/hamilton/CMB/Interfero/Bicep2/B2_3yr_cl_expected_withB_20140314.txt').T
clBB = clBB*2 #in order to have r=0.2
clBBtot = clBBl + clBB
clthBB = clthBB*2 #in order to have r=0.2
clthBBtot = clthBBl + clthBB


nbsqdeg = 380
fsky = nbsqdeg/41000
nkdeg = 87
mukarcmin = nkdeg/1000*60
beamsigma = 0.221
fwhmdeg = beamsigma*2.35
deltal = 35

fact = lll * (lll + 1) / (2 * np.pi)
factbin = lcenter * (lcenter + 1) / (2 * np.pi)
dcls, dcln, dcl = th_errors(lll, clBBtot/fact, fsky, mukarcmin, fwhmdeg, deltal=deltal)
dclsb, dclnb, dclb = th_errors_bins(lcenter, lll, clBBtot/fact, 
	fsky, mukarcmin, fwhmdeg, deltal=deltal)

clf()
plot(lll,dcln*fact,'k--',label='Noise variance (Knox formula)')
plot(lll,dcls*fact,'g--',label='Sample variance (Knox formula)')
plot(lll,dcl*fact,'r--',label='Total variance (Knox formula)')
plot(lcenter,dBB,'bo',label='BICEP2 errorbars')
xlabel('$\ell$')
ylabel('Error bar on Band Power')
ylim(0,0.02)
xlim(0,300)
title('Beam $\sigma={0:.3f}$ deg. - Area={1:.0f} deg$^2$ - Noise={2:.0f} $nK.degree$ - $\Delta\ell=${3:.0f}'.format(beamsigma,nbsqdeg,nkdeg,deltal))
legend(loc='upper left')
savefig('check_bicep2_errors.png')

clf()
plot(lcenter,dclnb*factbin,'k--',label='Noise variance (Knox formula)')
plot(lcenter,dclsb*factbin,'g--',label='Sample variance (Knox formula)')
plot(lcenter,dclb*factbin,'r--',label='Total variance (Knox formula)')
plot(lcenter,dBB,'bo',label='BICEP2 errorbars')
xlabel('$\ell$')
ylabel('Error bar on Band Power')
ylim(0,0.02)
xlim(0,300)
title('Beam $\sigma={0:.3f}$ deg. - Area={1:.0f} deg$^2$ - Noise={2:.0f} $nK.degree$ - $\Delta\ell=${3:.0f}'.format(beamsigma,nbsqdeg,nkdeg,deltal))
legend(loc='upper left')
savefig('check_bicep2_errors_binned.png')


clf()
plot(lll,clBBtot,label='r=0.2 + Lensing')
plot(lth,clthBBtot,label='r=0.2 + Lensing')
ylim(0,0.08)
xlim(0,400)
errorbar(lcenter-2,BB,fmt='ro',xlolims=lmin,xuplims=lmax,yerr=dBB,label = 'BICEP2 Error Bars')
errorbar(lcenter+2,BB,fmt='bo',xlolims=lmin,xuplims=lmax,yerr=dclb*factbin,label = 'Knox Approx. Error Bars')
xlabel('$\ell$')
ylabel('$\ell (\ell+1) C_\ell /2\pi$')
legend(loc='upper left')
savefig('new_bicep2_errors.png')




####### Chi2 w.r.t. no tensor model
chi2init = (((BB-clthBBl)/dBB)**2).sum()
ndfinit = len(dBB)
print(chi2init,ndfinit,chi2init/ndfinit,np.sqrt(chi2init/ndfinit))
chi2init = (((BB-clthBBl)/(dclb*factbin))**2).sum()
ndfinit = len(dBB)
print(chi2init,ndfinit,chi2init/ndfinit,np.sqrt(chi2init/ndfinit))





####### Fit with initial r=0.1 and initial lensing provided by BICEP2
clBBinit = clBB/2
def functsum(x,pars):
	bla = pars[0] / 0.1 * np.interp(x, ll, clBBinit) + pars[1] * np.interp(x, ll, clBBl)
	return bla

mask = lcenter < 1000

guess = [2., 1.]
fit = fitting.do_emcee(lcenter, BB, dBB, guess, functname = functsum,nbmc=10000,nburn=10000)
chains = fit[0]
res = fit[1]
err = fit[2]
cov = fit[3]
print(res[0]/err[0])

guess = [2., 1.]
factbin = lcenter * (lcenter + 1) / (2 * np.pi)
dclsb, dclnb, dclb = th_errors_bins(lcenter, lll, clBBtot/fact, 
	fsky, mukarcmin, fwhmdeg, deltal=deltal)
newerrors = dclb*factbin
fit2 = fitting.do_emcee(lcenter, BB, newerrors, guess, functname = functsum,nbmc=10000,nburn=10000)
chains2 = fit2[0]
res2 = fit2[1]
err2 = fit2[2]
cov2 = fit2[3]
print(res2[0]/err2[0])

clf()
errorbar(lcenter-2,BB,fmt='bo',xlolims=lmin,xuplims=lmax,yerr=dBB,label = 'BICEP2 Error Bars r = {0:.3f} +/- {1:.3f} <=> {2:.1f}$\sigma$'.format(res[0],err[0],res[0]/err[0]))
errorbar(lcenter+2,BB,fmt='ro',xlolims=lmin,xuplims=lmax,yerr=dclb*factbin,label = 'Knox Approx. Error Bars r = {0:.3f} +/- {1:.3f} <=> {2:.1f}$\sigma$'.format(res2[0],err2[0],res2[0]/err2[0]))
ylim(0,0.08)
xlim(0,400)
plot(lll,functsum(lll,res),'b',label='Fit')
plot(lll,functsum(lll,np.array([res[0],0])),'b--')
plot(lll,functsum(lll,np.array([0,res[1]])),'b:')
plot(lll,functsum(lll,res2),'r',label='Fit')
plot(lll,functsum(lll,np.array([res2[0],0])),'r--')
plot(lll,functsum(lll,np.array([0,res2[1]])),'r:')
xlabel('$\ell$')
ylabel('$\ell (\ell+1) C_\ell /2\pi$')
legend(loc='upper left')
savefig('newfit_bicep2.png')

from McMc import mcmc

clf()
thechain = {'r':chains[:,0],'A_l':chains[:,1]}
thechain2 = {'r':chains2[:,0],'A_l':chains2[:,1]}
a0=mcmc.matrixplot(thechain2,['r','A_l'],'blue',4,limits=[[0,0.4],[-0.5,3.5]])
a1=mcmc.matrixplot(thechain,['r','A_l'],'red',4,limits=[[0,0.4],[-0.5,3.5]])
subplot(2,2,2)
axis('off')
legend([a1,a0],['BICEP2 Error Bars r = {0:.3f} +/- {1:.3f} <=> {2:.1f}$\sigma$'.format(res[0],err[0],res[0]/err[0]), 'Knox Approx. Error Bars r = {0:.3f} +/- {1:.3f} <=> {2:.1f}$\sigma$'.format(res2[0],err2[0],res2[0]/err2[0])])
savefig('matrixplot_bicep2.png')




####### Fit with Camb models
H0 = 67.04
omegab = 0.022032
omegac = 0.12038
h2 = (H0/100.)**2
scalar_amp = np.exp(3.098)/1.E10
omegav = h2 - omegab - omegac
Omegab = omegab/h2
Omegac = omegac/h2
rvalue = 0.2
print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',omegav/h2

params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':rvalue,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False}

params_l = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
        	 'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
        	 'tensor_ratio':0,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}

lmax = 1000
ell = np.arange(1,600)

#### lensing / no tensors
Tpl,Epl,Bpl,Xpl = pycamb.camb(lmax+1,**params_l)
Tpl = Tpl[0:599]
Epl = Epl[0:599]
Bpl = Bpl[0:599]
Xpl = Xpl[0:599]

def getbbsum(pars, x):
	thepars = params.copy()
	thepars['tensor_ratio'] = pars[0]
	Tprim,Eprim,Bprim,Xprim = pycamb.camb(lmax+1,**thepars)
	Bprim = Bprim[0:599]
	bla = np.interp(x, ell, Bprim) + pars[1] * np.interp(x, ell, Bpl)
	return bla

def residuals(pars, x, data, errors):
	bla = getbbsum(pars, x)
	residuals = (data-bla)/errors	
	print(pars,(residuals**2).sum())
	#plot(x,bla)
	return residuals


import scipy.optimize as spo
guess = [.2, 1.]
res, cov_x, aa, aa, aa= spo.leastsq(residuals, guess, args=(lcenter, BB, dBB), epsfcn=0.01, full_output=True)
err = np.sqrt(np.diag(cov_x))

res2, cov_x2, aa, aa, aa= spo.leastsq(residuals, guess, args=(lcenter, BB, dclb*factbin), epsfcn=0.01, full_output=True)
err2 = np.sqrt(np.diag(cov_x2))


clf()
errorbar(lcenter-2,BB,fmt='bo',xlolims=lmin,xuplims=lmax,yerr=dBB,label = 'BICEP2 Error Bars r = {0:.3f} +/- {1:.3f} <=> {2:.1f}$\sigma$'.format(res[0],err[0],res[0]/err[0]))
errorbar(lcenter+2,BB,fmt='ro',xlolims=lmin,xuplims=lmax,yerr=dclb*factbin,label = 'Knox Approx. Error Bars r = {0:.3f} +/- {1:.3f} <=> {2:.1f}$\sigma$'.format(res2[0],err2[0],res2[0]/err2[0]))
ylim(0,0.08)
xlim(0,400)
plot(ell,getbbsum(res,ell),'b',label='Fit')
plot(ell,getbbsum(np.array([res[0],0]),ell),'b--')
plot(ell,getbbsum(np.array([0,res[1]]),ell),'b:')
plot(ell,getbbsum(res2,ell),'r',label='Fit')
plot(ell,getbbsum(np.array([res2[0],0]),ell),'r--')
plot(ell,getbbsum(np.array([0,res2[1]]),ell),'r:')
xlabel('$\ell$')
ylabel('$\ell (\ell+1) C_\ell /2\pi$')
legend(loc='upper left')
title('Using CAMB and fixed error bars')
savefig('fit_withcamb_fixed_errors.png')


#### dBB does not include sample variance, so should be added
def getbbsum_samplevar(pars, x):
	thepars = params.copy()
	thepars['tensor_ratio'] = pars[0]
	Tprim,Eprim,Bprim,Xprim = pycamb.camb(lmax+1,**thepars)
	Bprim = Bprim[0:599]
	totBB = Bprim + pars[1] * Bpl
	dclsb, dclnb, dclb = th_errors_bins(x, ell, totBB/(ell*(ell+1)/(2*np.pi)), fsky, mukarcmin, fwhmdeg, deltal=deltal)
	bla = np.interp(x, ell, Bprim) + pars[1] * np.interp(x, ell, Bpl)
	return bla, dclsb, dclnb, dclb

def residuals_samplevar(pars, x, data, errors):
	bla, dclsb, dclnb, dclb = getbbsum_samplevar(pars, x)
	residuals = (data-bla)/(errors + dclsb*x*(x+1)/(2*np.pi))	
	return residuals


mask = lcenter < 2000
guess = [.2, 1.]
resf, cov_x, aa, aa, aa= spo.leastsq(residuals_samplevar, guess, args=(lcenter[mask], BB[mask], dBB[mask]), epsfcn=0.01, full_output=True)
errf = np.sqrt(np.diag(cov_x))

clf()
errorbar(lcenter-2,BB,fmt='bo',yerr=dBB,alpha=0.3)
errorbar(lcenter[mask]-2,BB[mask],fmt='bo',yerr=dBB[mask],label = 'BICEP2 Error Bars r = {0:.3f} +/- {1:.3f} <=> {2:.1f}$\sigma$'.format(resf[0],errf[0],resf[0]/errf[0]))
ylim(0,0.08)
xlim(0,400)
btot, dclsb, dclnb, dclb = getbbsum_samplevar(resf, ell)
bprim, aa, bb, cc = getbbsum_samplevar([resf[0],0], ell)
blens, aa, bb, cc = getbbsum_samplevar([0,resf[1]], ell)
plot(ell,bprim,'b--')
plot(ell,blens,'b:')
plot(ell,btot,'b')
plot(ell,btot-dclsb*ell*(ell+1)/(2*np.pi),'b',alpha=0.7)
plot(ell,btot+dclsb*ell*(ell+1)/(2*np.pi),'b',alpha=0.7)
xlabel('$\ell$')
ylabel('$\ell (\ell+1) C_\ell /2\pi$')
legend(loc='upper left')
title('Using CAMB and adding sample variance while fitting')
savefig('correct_fit_allpts.png')



mask = lcenter < 200
guess = [.2, 1.]
resf, cov_x, aa, aa, aa= spo.leastsq(residuals_samplevar, guess, args=(lcenter[mask], BB[mask], dBB[mask]), epsfcn=0.01, full_output=True)
errf = np.sqrt(np.diag(cov_x))

clf()
errorbar(lcenter-2,BB,fmt='bo',yerr=dBB,alpha=0.3)
errorbar(lcenter[mask]-2,BB[mask],fmt='bo',yerr=dBB[mask],label = 'BICEP2 Error Bars r = {0:.3f} +/- {1:.3f} <=> {2:.1f}$\sigma$'.format(resf[0],errf[0],resf[0]/errf[0]))
ylim(0,0.08)
xlim(0,400)
btot, dclsb, dclnb, dclb = getbbsum_samplevar(resf, ell)
bprim, aa, bb, cc = getbbsum_samplevar([resf[0],0], ell)
blens, aa, bb, cc = getbbsum_samplevar([0,resf[1]], ell)
plot(ell,bprim,'b--')
plot(ell,blens,'b:')
plot(ell,btot,'b')
plot(ell,btot-dclsb*ell*(ell+1)/(2*np.pi),'b',alpha=0.7)
plot(ell,btot+dclsb*ell*(ell+1)/(2*np.pi),'b',alpha=0.7)
xlabel('$\ell$')
ylabel('$\ell (\ell+1) C_\ell /2\pi$')
legend(loc='upper left')
title('Using CAMB and adding sample variance while fitting')
savefig('correct_fit_5pts.png')









