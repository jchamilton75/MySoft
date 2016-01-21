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
from pysimulators import FitsArray

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
ellav, deltacl150, noisevar150, samplevar150, neq_nh150, nsig150, bs150= qubic_sensitivity.give_qubic_errors(inst, ellbins, lll, spectra[3], nu=150e9, epsilon=epsilon,net_polar=NET150, plot_baselines=True, symplot='ro')

#### 220 GHz
ellav, deltacl220, noisevar220, samplevar220, neq_nh220, nsig220, bs220= qubic_sensitivity.give_qubic_errors(inst, ellbins, lll, spectra[3], nu=220e9, epsilon=epsilon,net_polar=NET220, plot_baselines=True, symplot='bo')

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


######### Explore dust
from Sensitivity import dualband_lib as db
### plot to compare with Planck XXX fig. 6
betadust = 1.59
Tdust = 19.6
clf()
yscale('log')
nu=linspace(100,400,1000)
plot(nu,(db.mbb(nu, betadust, Tdust)/db.mbb(353, betadust, Tdust))**2,'b')
plot(nu,nu*0+1,'k:')

# should be 0.0453
db.freq_conversion(353, 150, betadust, Tdust)
# should be 0.1347
db.freq_conversion(353, 220, betadust, Tdust)

Dl_353 = 13.4 # +/- 0.26 muK^2 at l=80 (section 6.2)
Dl_150 = Dl_353 * (db.freq_conversion(353, 150, betadust, Tdust))**2
Dl_220 = Dl_353 * (db.freq_conversion(353, 220, betadust, Tdust))**2

clf()
yscale('log')
xscale('log')
xlim(10,500)
ylim(1e-4,100)
title('r = {0:3.2f}'.format(r))
xlabel('$\ell$')
#ylabel('$\ell(\ell+1)C_\ell/2\pi$'+'    '+'$[\mu K^2]$ ')
ylabel(r'$\frac{\ell(\ell+1)}{2\pi}\,C_\ell$'+'    '+'$[\mu K^2]$ ')
plot(lll,lll*0,'k:')
plot(lll,spectra[3]*(lll*(lll+1))/(2*np.pi),'k',lw=3)
plot(lll, db.Dl_BB_dust(lll, 150),'r', label='150 GHz')
plot(lll, db.Dl_BB_dust(lll, 220),'g', label='220 GHz')
plot(lll, db.Dl_BB_dust(lll, 353),'b', label='353 GHz')
plot(lll, db.Dl_BB_dust(lll, 150, 220),'m', label='150 x 353 GHz')
plot(lll, db.Dl_BB_dust(lll, 150, 353),'k', label='150 x 353 GHz')
plot(lll, db.Dl_BB_dust(lll, 220, 353),'y', label='220 x 353 GHz')
errorbar(ellav, db.Dl_BB_dust_bins(ellbins, 150), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='ro')
errorbar(ellav, db.Dl_BB_dust_bins(ellbins, 220), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='go')
errorbar(ellav, db.Dl_BB_dust_bins(ellbins, 353), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='bo')
errorbar(ellav, db.Dl_BB_dust_bins(ellbins, 150, 220), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='mo')
errorbar(ellav, db.Dl_BB_dust_bins(ellbins, 150, 353), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='ko')
errorbar(ellav, db.Dl_BB_dust_bins(ellbins, 220, 353), xerr=np.array([ellav-ellmin,ellmax-ellav]), fmt='yo')
legend(loc='upper left')



########## Now put both
lmax = 500
ell = np.arange(lmax+1)
fact = (ell*(ell+1))/(2*np.pi)

freqs = [150, 220, 353]
ncross = np.int(len(freqs) + len(freqs)*(len(freqs)-1)/2)
col = get_cmap('jet')(np.linspace(0, 1.0, ncross)[::-1])

r=0.05
dldust_80_353 = 13.4
alphadust = -2.42
betadust = 1.59
Tdust = 19.6
params = [dldust_80_353, alphadust, betadust, Tdust]
allspec = []
allfreqs = []
for i in np.arange(len(freqs)):
	for j in np.arange(i, len(freqs)):
		thecross = np.str(freqs[i])+'x'+np.str(freqs[j])
		print('Doing '+thecross)
		spec = db.get_ClBB_cross_th(ell, freqs[i], freqGHz2=freqs[j], rvalue=r, dustParams=params)
		allspec.append(spec)
		allfreqs.append(thecross)

clf()
yscale('log')
xscale('log')
xlim(10,500)
ylim(1e-4,100)
title('r = {0:3.2f}'.format(r))
xlabel('$\ell$')
ylabel(r'$\frac{\ell(\ell+1)}{2\pi}\,C_\ell$'+'    '+'$[\mu K^2]$ ')
for i in np.arange(len(allspec)):
	plot(ell, allspec[i][0] * fact, label = allfreqs[i], color=col[i])
	plot(ell, allspec[i][1] * fact, '--', color=col[i])
	plot(ell, allspec[i][2] * fact, ':', color=col[i])
legend()





###### Now with the error bars and binning
r=0.05
dldust_80_353 = 13.4
alphadust = -2.42
betadust = 1.59
Tdust = 19.6
dustParams = [dldust_80_353, alphadust, betadust, Tdust]


### QUBIC instrument
inst = QubicInstrument()

### Binning (as BICEP2)
ellbins = np.array([21, 56, 91, 126, 161, 196, 231, 266, 301, 335])
ellmin = ellbins[:len(ellbins)-1]
ellmax = ellbins[1:len(ellbins)]


#### Build a Cl Library with CAMB
from Cosmo import interpol_camb as ic
#rmin = 0.001
#rmax = 1
#nb =100
#lmaxcamb = np.max(ellbins)
#rvalues = np.concatenate((np.zeros(1),np.logspace(np.log10(rmin),np.log10(rmax),nb)))
#camblib = ic.rcamblib(rvalues, lmaxcamb)
#FitsArray(camblib[0], copy=False).save('camblib_ell.fits')
#FitsArray(camblib[1], copy=False).save('camblib_r.fits')
#FitsArray(camblib[2], copy=False).save('camblib_cl.fits')
### Restore it
ellcamblib = FitsArray('camblib600_ell.fits')
rcamblib = FitsArray('camblib600_r.fits')
clcamblib = FitsArray('camblib600_cl.fits')
camblib = [ellcamblib, rcamblib, clcamblib]

################## Find Planck noise at 353 GHz in BB in BICEP2 Field
#### From Fig 9 Planck XXX
fskyplanck = 0.01
errors_fig = np.array([0.008/2, 0.015/2, 0.032/2])
ell_fig = np.array([40,120, 250, 400])
ellmin_fig = ell_fig[0:-1]
ellmax_fig = ell_fig[1:]
ellav_fig = 0.5 * (ellmin_fig + ellmax_fig)
deltal_fig = ellmax_fig - ellmin_fig
conversion_150_to_353 = 1./db.freq_conversion(353, 150, betadust, Tdust)**2
errors_353 = errors_fig * conversion_150_to_353

ellav_fig * (ellav_fig +1) / (2*pi) * np.sqrt(2./((2*ellav_fig+1)*deltal_fig*fskyplanck)) * 0.0151
errors_353

##### Checking Planck 353 GHz noise ###########################################
ellav = 0.5 * (ellmin + ellmax)
deltal = ellmax - ellmin
fskyplanck = 0.01
dlnoise = ellav * (ellav +1) / (2*pi) * np.sqrt(2./((2*ellav+1)*deltal*fskyplanck)) * 0.0151

duration = 1
epsilon =1
freq = [353]
type = ['im']
NET = [850]
name = ['353']
col = 'r'
instinfo = [inst, ellbins, freq, type, NET, fskyplanck, duration, epsilon, name, col]
bla = db.get_multiband_covariance(instinfo, r, doplot=True, dustParams=dustParams, verbose=True, camblib=camblib)
plot(ellav, dlnoise,'ro')
###############################################################################


def dotheplot(r, instruments ,damp=1, marginalized=True, nn=30, rrange=[0.,0.3], dustrange = [0.,25.], saveplot=False):
	dldust_80_353 = 13.4*damp
	alphadust = -2.42
	betadust = 1.59
	Tdust = 19.6
	ThedustParams = np.array([dldust_80_353, alphadust, betadust, Tdust])

	clf()
	paramsdefault = np.array([r, dldust_80_353, alphadust, betadust, Tdust])
	xlabel('r')
	ylabel('Likelihood')

	likelihoods = []
	maxlike = []
	for instinfo in instruments:
		### Calculate input spectra
		bla = db.get_multiband_covariance(instinfo, r, doplot=False, dustParams=ThedustParams, verbose=True, camblib=camblib)
		spec = bla[3]

		if marginalized:
			### Marginalizing over Dust Amplitude
			title('Marginalized over Dust (Amplitude = {0:1.0f} ) ; r = {1:4.2f}'.format(damp,r))
			# dust amplitude
			nvalsmarg = nn
			valsmarg = linspace(dustrange[0],dustrange[1], nvalsmarg)
			indexmarg = 1
			# r
			nvals = nn
			valsamp = linspace(rrange[0], rrange[1], nvals)
			index = 0
			thelike = db.like_1d_marginalize(spec, index, valsamp, indexmarg, valsmarg, instinfo, camblib=camblib, paramsdefault=paramsdefault)
		else:
			### r
			nvals = nn
			valsamp = linspace(rrange[0], rrange[1], nvals)
			index = 0
			#### Likelihoods 1D
			title('Fixed Dust = {0:1.0f} ; r = {1:4.2f}'.format(damp, r))
			thelike = db.like_1d(spec, index, valsamp, instinfo, camblib=camblib, paramsdefault=paramsdefault)
		maxlike.append(np.max(thelike[0]))
		likelihoods.append(thelike)
		ylim(0,np.max(np.array(maxlike))*1.2)
		draw()


	if saveplot:
		if marginalized:
			savefig('db_marginalized_dust={0:1.0f}_r={1:4.2f}.png'.format(damp,r))
		else:
			savefig('db_fixed_dust={0:1.0f}_r={1:4.2f}.png'.format(damp,r))

	return likelihoods

### Instruments
netplanck_353 = 850
net150 = 550.
net220 = 1450.


freqsA = [150]
typeA = ['bi']
NETsA = [net150/sqrt(2)]
nameA = ['150x2']
colA = 'k'
fskyA = 0.01
instrumentA = [inst, ellbins, freqsA, typeA, NETsA, fskyA, nameA, colA]

freqsB = [150, 220]
typeB = ['bi', 'bi']
NETsB = [net150, net220]
nameB = ['150, 220']
colB = 'm'
fskyB = 0.01
instrumentB = [inst, ellbins, freqsB, typeB, NETsB, fskyB, nameB, colB]

freqsC = [150, 353]
typeC = ['bi', 'im']
NETsC = [net150/sqrt(2), netplanck_353]
nameC = ['150x2, 353']
colC = 'b'
fskyC = 0.01
instrumentC = [inst, ellbins, freqsC, typeC, NETsC, fskyC, nameC, colC]

freqsD = [150, 220, 353]
typeD = ['bi', 'bi', 'im']
NETsD = [net150, net220, netplanck_353]
nameD = ['150, 220, 353']
colD = 'r'
fskyD = 0.01
instrumentD = [inst, ellbins, freqsD, typeD, NETsD, fskyD, nameD, colD]

instruments = [instrumentA, instrumentB, instrumentC, instrumentD]



############# RUN
saveplot=False

like_fixed_00 = dotheplot(0,instruments,damp=0, marginalized=False, rrange=[0,0.4], nn=100, saveplot=saveplot)
like_fixed_r0 = dotheplot(0.05,instruments,damp=0, marginalized=False, rrange=[0,0.4], nn=100, saveplot=saveplot)
like_fixed_0dust = dotheplot(0,instruments,damp=1, marginalized=False, rrange=[0,0.4], nn=100, saveplot=saveplot)
like_fixed_rdust = dotheplot(0.05,instruments,damp=1, marginalized=False, rrange=[0,0.4], nn=100, saveplot=saveplot)

like_marg_00 = dotheplot(0,instruments,damp=0, marginalized=True, rrange=[0,0.4], nn=100, saveplot=saveplot)
like_marg_r0 = dotheplot(0.05,instruments,damp=0, marginalized=True, rrange=[0,0.4], nn=100, saveplot=saveplot)
like_marg_0dust = dotheplot(0,instruments,damp=1, marginalized=True, rrange=[0,0.4], nn=100, saveplot=saveplot)
like_marg_rdust = dotheplot(0.05,instruments,damp=1, marginalized=True, rrange=[0,0.4], nn=100, saveplot=saveplot)








