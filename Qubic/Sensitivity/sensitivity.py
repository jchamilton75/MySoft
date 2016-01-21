from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import pi
import pycamb
from qubic import QubicInstrument
from scipy.constants import c
from Sensitivity import qubic_sensitivity
from Homogeneity import SplineFitting


#### Get input Power spectra
################# Input Power spectrum ###################################
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
r = 0.05
H0 = 67.04
omegab = 0.022032
omegac = 0.12038
h2 = (H0/100.)**2
scalar_amp = np.exp(3.098)/1.E10
omegav = h2 - omegab - omegac
Omegab = omegab/h2
Omegac = omegac/h2
print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',omegav/h2

params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':r,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}
lmaxcamb = 1000
T,E,B,X = pycamb.camb(lmaxcamb+1+150,**params)
lll = np.arange(1,lmaxcamb+1)
T=T[:lmaxcamb]
E=E[:lmaxcamb]
B=B[:lmaxcamb]
X=X[:lmaxcamb]
fact = (lll*(lll+1))/(2*np.pi)
spectra = [lll, T/fact, E/fact, B/fact, X/fact]

params_nl = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':r,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False}
lmaxcamb = 1000
Tnl,Enl,Bnl,Xnl = pycamb.camb(lmaxcamb+1+150,**params_nl)
lll = np.arange(1,lmaxcamb+1)
Tnl=Tnl[:lmaxcamb]
Enl=Enl[:lmaxcamb]
Bnl=Bnl[:lmaxcamb]
Xnl=Xnl[:lmaxcamb]
fact = (lll*(lll+1))/(2*np.pi)
spectra_nl = [lll, Tnl/fact, Enl/fact, Bnl/fact, Xnl/fact]

clf()
mp.plot(lll, np.sqrt(spectra[1]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{TT}$')
mp.plot(lll, np.sqrt(abs(spectra[4])*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{TE}$')
mp.plot(lll,np.sqrt(spectra[2]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{EE}$')
mp.plot(lll,np.sqrt(spectra[3]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{BB}$')
mp.yscale('log')
mp.xlim(0,lmaxcamb+1)
#ylim(0.0001,100)
mp.xlabel('$\ell$')
mp.ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
mp.legend(loc='lower right',frameon=False)


########## Qubic Instrument
inst = QubicInstrument()
NET150 = (220+314)/2*sqrt(2)*sqrt(2)  ### Average between witner and summer
NET220 = (520+906)/2*sqrt(2)*sqrt(2)  ### Average between witner and summer

### Binning
deltal=100.
lmin=25
ellbins = np.array([25, 45, 65, 95, 140, 200, 300])
ellmin = ellbins[:len(ellbins)-1]
ellmax = ellbins[1:len(ellbins)]

### Get baselines and errors
epsilon = 0.7 * 0.7  #focal plane integration + optical efficiency)

clf()
subplot(2,1,1)
xlim(-300,300)
ylim(-300,300)
xlabel('$u$')
ylabel('$v$')
subplot(2,1,2)
ylim(0,1)
ylabel('$N_{eq}(\ell)/N_h$')
xlabel('$\ell$')

#### 150 GHz
ellav, deltacl150, noisevar150, samplevar150, neq_nh150, nsig150= qubic_sensitivity.give_qubic_errors(inst, ellbins, lll, spectra[3], nu=150e9, epsilon=epsilon,net_polar=NET150, plot_baselines=True, symplot='ro')

#### 220 GHz
ellav, deltacl220, noisevar220, samplevar220, neq_nh220, nsig220= qubic_sensitivity.give_qubic_errors(inst, ellbins, lll, spectra[3], nu=220e9, epsilon=epsilon,net_polar=NET220, plot_baselines=True, symplot='bo')

#### 150*220
samplevar150x220 = np.sqrt(samplevar150*samplevar220)
noisevar150x220 = np.sqrt(noisevar150*noisevar220)
deltacl150x220 = samplevar150x220 + noisevar150x220



#### S/N
spec = np.interp(ellav, lll, spectra[3]*(lll*(lll+1))/(2*np.pi))
nbsig150 = np.sqrt(np.sum(spec**2/(deltacl150*(ellav*(ellav+1))/(2*np.pi))**2))
nbsig220 = np.sqrt(np.sum(spec**2/(deltacl220*(ellav*(ellav+1))/(2*np.pi))**2))
nbsig150x220 = np.sqrt(np.sum(spec**2/(deltacl150x220*(ellav*(ellav+1))/(2*np.pi))**2))


clf()
#yscale('log')
xlim(0,300)
ylim(-0.02,0.05)
title('r = {0:3.2f}'.format(r))
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$'+'    '+'$[\mu K^2]$ ')
plot(lll,lll*0,'k:')
plot(lll,spectra[3]*(lll*(lll+1))/(2*np.pi),'k',lw=3)
plot(lll,spectra_nl[3]*(lll*(lll+1))/(2*np.pi),'k--')
plot(lll,(spectra[3]-spectra_nl[3])*(lll*(lll+1))/(2*np.pi),'k:')
errorbar(ellav-2, spec, yerr=deltacl220*(ellav*(ellav+1))/(2*np.pi), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='bo', label='220 GHz: {0:3.1f} $\sigma$'.format(nbsig220))
errorbar(ellav-2, spec, yerr=noisevar220*(ellav*(ellav+1))/(2*np.pi), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='bo')
errorbar(ellav+2, spec, yerr=deltacl150*(ellav*(ellav+1))/(2*np.pi), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='ro', label='150 GHz: {0:3.1f} $\sigma$'.format(nbsig150))
errorbar(ellav+2, spec, yerr=noisevar150*(ellav*(ellav+1))/(2*np.pi), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='ro')
errorbar(ellav, spec, yerr=deltacl150x220*(ellav*(ellav+1))/(2*np.pi), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='go', label='150x220 GHz: {0:3.1f} $\sigma$'.format(nbsig150x220))
errorbar(ellav, spec, yerr=noisevar150x220*(ellav*(ellav+1))/(2*np.pi), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='go')
legend()




########### explore (r, fsky) plane
fskyvals = linspace(0.005, 0.05,30)
rvals = linspace(0., 0.2, 30)
neq_nh150 = False

sig150 = np.zeros((len(fskyvals), len(rvals)))
for i in xrange(len(rvals)):
	pars = params.copy()
	pars['tensor_ratio'] = rvals[i]
	print('Doing r={0:4.3f}'.format(rvals[i]))
	T,E,B,X = pycamb.camb(lmaxcamb+1+150,**pars)
	lll = np.arange(1,lmaxcamb+1)
	T=T[:lmaxcamb]
	E=E[:lmaxcamb]
	B=B[:lmaxcamb]
	X=X[:lmaxcamb]
	fact = (lll*(lll+1))/(2*np.pi)
	spectra = [lll, T/fact, E/fact, B/fact, X/fact]
	for j in xrange(len(fskyvals)):
		ellav, deltacl150, noisevar150, samplevar150, neq_nh150, nbsig= qubic_sensitivity.give_qubic_errors(inst, ellbins, lll, spectra[3], nu=150e9, epsilon=epsilon,net_polar=NET150, fsky=fskyvals[j], neq_nh=neq_nh150)
		sig150[j,i] = nbsig


clf()
y,x = meshgrid(rvals, fskyvals)
cs=contour(x,y,sig150,levels=[1.65, 1.96, 2.58, 3.29], colors=['k','k','k'])
fmt = {}
strs = ['90%','95%','99%','99.9%']
for l,s in zip( cs.levels, strs ):
    fmt[l] = s
clabel(cs,fmt=fmt)
contourf(x,y, sig150, levels=linspace(0,5,21))
xlabel('$f_{sky}$')
ylabel('Tensor to Scalar ratio')
title('B-mode signal (prim+lens) detection\n with QUBIC 150 GHz channel')
cb=colorbar()
cb.set_label('number of $\sigma$')
savefig('nbsig_rvals_fsky.png')



######### Explore dust
import scipy.constants

def Bnu(nuGHz, temp):
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k
	nu = nuGHz*1e9
	return 2 * h * nu**3 / c**2 / (np.exp(h * nu / k / temp) - 1)

def dBnu_dT(nuGHz, temp):
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k
	nu = nuGHz*1e9
	theBnu = Bnu(nuGHz, temp)
	return (theBnu * c / nu / temp)**2 / 2 * np.exp(h * nu / k / temp) / k

def mbb(nuGHz, beta, temp):
	return (nuGHz/353)**beta * Bnu(nuGHz, temp)

def KCMB2MJy_sr(nuGHz):
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k
	T = 2.725
	nu = nuGHz*1e9
	x = h * nu / k / T
	ex = np.exp(x)
	fac_in = dBnu_dT(nuGHz, T)
	fac_out = 1e20
	return fac_in * fac_out

def freq_conversion(nuGHz_in, nuGHz_out, betadust, Tdust):
	val_in = KCMB2MJy_sr(nuGHz_in) / mbb(nuGHz_in, betadust, Tdust)
	val_out = KCMB2MJy_sr(nuGHz_out) / mbb(nuGHz_out, betadust, Tdust)
	return val_in / val_out

### plot to compare with Planck XXX fig. 6
betadust = 1.59
Tdust = 19.6
clf()
yscale('log')
nu=linspace(100,400,1000)
plot(nu,(mbb(nu, betadust, Tdust)/mbb(353, betadust, Tdust))**2,'b')
plot(nu,nu*0+1,'k:')

# should be 0.0453
freq_conversion(353, 150, betadust, Tdust)
# should be 0.1347
freq_conversion(353, 220, betadust, Tdust)

Dl_353 = 13.4 # +/- 0.26 muK^2 at l=80 (section 6.2)
Dl_150 = Dl_353 * (freq_conversion(353, 150, betadust, Tdust))**2
Dl_220 = Dl_353 * (freq_conversion(353, 220, betadust, Tdust))**2

def Dl_BB_dust(ell, freqGHz1, freqGHz2):
	Dl_353_ell80 = 13.4
	alpha_bb = -2.42
	return Dl_353_ell80 * (freq_conversion(353, freqGHz1, betadust, Tdust) * freq_conversion(353, freqGHz2, betadust, Tdust)) * (ell/80)**(alpha_bb+2)

def Dl_BB_dust_bins(ellbins, freqGHz1, freqGHz2):
	res = np.zeros(len(ellbins)-1)
	for i in xrange(len(ellbins)-1):
		lvals = ellbins[i]+np.arange(ellbins[i+1]-ellbins[i])
		dlvals = Dl_BB_dust(lvals, freqGHz1, freqGHz2)
		res[i] = np.mean(dlvals)
	return res


clf()
yscale('log')
xscale('log')
xlim(10,500)
ylim(1e-4,100)
title('r = {0:3.2f}'.format(r))
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$'+'    '+'$[\mu K^2]$ ')
plot(lll,lll*0,'k:')
plot(lll,spectra[3]*(lll*(lll+1))/(2*np.pi),'k',lw=3)
#plot(lll, Dl_BB_dust(lll, 150),'r', label='150 GHz')
#plot(lll, Dl_BB_dust(lll, 220),'g', label='220 GHz')
#plot(lll, Dl_BB_dust(lll, 353),'b', label='353 GHz')
errorbar(ellav, Dl_BB_dust_bins(ellbins, 150, 150), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='ro', label='150x150 GHz')
errorbar(ellav, Dl_BB_dust_bins(ellbins, 220, 220), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='go', label='220x220 GHz')
errorbar(ellav, Dl_BB_dust_bins(ellbins, 353, 353), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='bo', label='353x353 GHz')
errorbar(ellav, Dl_BB_dust_bins(ellbins, 150, 220), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='mo', label='150x220 GHz')
errorbar(ellav, Dl_BB_dust_bins(ellbins, 150, 353), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='ko', label='150x353 GHz')
errorbar(ellav, Dl_BB_dust_bins(ellbins, 220, 353), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='yo', label='220x353 GHz')
legend(loc='upper left')

















