

mukarcmin = 1.
#fwhmdeg = np.array([1., 5., 30.])/60
#fsky = np.array([0.05, 0.1, 0.5, 1.])
fwhmdeg = np.linspace(1.,30., 100)/60
fsky = 1
lens_res = 0

sigs_r = np.zeros(len(fwhmdeg))
sigs_r_svl = np.zeros(len(fwhmdeg))
sigs_rt = np.zeros(len(fwhmdeg))
sigs_rt_svl = np.zeros(len(fwhmdeg))
for k in np.arange(len(fwhmdeg)):
	print('Num:',k)
	s, s_svl, der, spectra = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg[k], lens_res, 
		der=der, spectra=spectra, min_ell=1)
	sigs_r[k] = s[0]
	sigs_r_svl[k] = s_svl[0]
	sigs_rt[k] = s[1]
	sigs_rt_svl[k] = s_svl[1]

clf()
plot(fwhmdeg*60, sigs_rt)

lens_res = 0
s, s_svl, der, spectra = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, 0.01, lens_res, 
		der=der, spectra=spectra, min_ell=1, plotcl=True)






mukarcmin = 1.
#fwhmdeg = np.array([1., 5., 30.])/60
#fsky = np.array([0.05, 0.1, 0.5, 1.])
fwhmdeg = np.linspace(1.,30., 30)/60
fsky = np.array([0.05, 0.1, 0.5, 1.])
lens_res = np.array([0., 0.01, 0.1, 1.])

sigs_r = np.zeros((len(lens_res), len(fsky), len(fwhmdeg)))
sigs_r_svl = np.zeros((len(lens_res), len(fsky), len(fwhmdeg)))
sigs_rt = np.zeros((len(lens_res), len(fsky), len(fwhmdeg)))
sigs_rt_svl = np.zeros((len(lens_res), len(fsky), len(fwhmdeg)))
for i in np.arange(len(lens_res)):
	for j in np.arange(len(fsky)):
		for k in np.arange(len(fwhmdeg)):
			print('Num:',i,j,k)
			s, s_svl, der, spectra = Fisher.get_tratio_accuracy(params, fsky[j], mukarcmin, fwhmdeg[k], lens_res[i], 
				der=der, spectra=spectra, min_ell=1)
			sigs_r[i,j,k] = s[0]
			sigs_r_svl[i,j,k] = s_svl[0]
			sigs_rt[i,j,k] = s[1]
			sigs_rt_svl[i,j,k] = s_svl[1]


clf()
imshow(sigs_rt[0,:,:],interpolation='nearest')
colorbar()

clf()
imshow(sigs_rt_svl[0,:,:],interpolation='nearest')
colorbar()


clf()
imshow(sigs_r[1,:,:],interpolation='nearest')
colorbar()













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
        	 'tensor_ratio':0,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}

lmax = 2200
ell = np.arange(1,2000)


Tprim,Eprim,Bprim,Xprim = pycamb.camb(lmax+1,**params)
Tprim = Tprim[0:1999]
Eprim = Eprim[0:1999]
Bprim = Bprim[0:1999]
Xprim = Xprim[0:1999]
Tl,El,Bl,Xl = pycamb.camb(lmax+1,**params_l)
Tl = Tl[0:1999]
El = El[0:1999]
Bl = Bl[0:1999]
Xl = Xl[0:1999]
Btot = Bl + Bprim

clf()
plot(ell,Bprim,label='Primordial $C_\ell^{BB}$'+'$ (r={0:.2f})$'.format(rvalue))
plot(ell,Bl,label='Lensing $C_\ell^{BB}$')
plot(ell,Btot,label='Total $C_\ell^{BB}$'+'$ (r={0:.2f})$'.format(rvalue))
yscale('log')
xlim(0,2000)
ylim(0.0005,0.1)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$'+'    '+'$[\mu K]^2$ ')
legend(loc='upper left',frameon=False)




############## Base parameters
rval = 0.2
trval = -8
aval = 1.
pars = [rval, trval, aval] 
parnames = ['$r$', '$T_{ratio}$', '$A_L$']
centervals = np.array([rval, trval, aval])

##### stage 4 1muk.arcmin 50% - 1arcmin
##### SPT pol 1st season now 10,uk.arcmin- 1arcmin - 2500 sqdeg ?
##### stage III 4muk.arcmin 6% - 1 arcmin

#### sample experimental configuration  ##################################
totsky = (4*np.pi*(180/np.pi)**2)
nbsqdeg = 380.
fsky = nbsqdeg/totsky
nkdeg = 87
mukarcmin = nkdeg/1000*60
beamsigma = 0.221
fwhmdeg = beamsigma*2.35
deltal = 35
min_ell = 20
nbins = 9
lmin = min_ell+(deltal)*np.arange(nbins)
lmax = deltal-1+min_ell+(deltal)*np.arange(nbins)
ell = np.arange(np.max(lmax)+100)+1
lcenter = (lmax+lmin)/2
allargs = [params, lmax, ell, Bl, fsky, mukarcmin, fwhmdeg, deltal, lmin, lmax]

btot_binned, clprim_binned, cllens_binned, dclsb, dclnb, dclb, Bprim, Blensed, totBB = get_spectrum_bins(pars, lcenter, allargs)
bins = lcenter

clf()
plot(ell, Bprim, 'k--')
plot(ell, Blensed, 'k:')
plot(ell, totBB, 'k')
xlim(0,max(lmax))
errorbar(lcenter, btot_binned, yerr=dclb, xerr = (lmax+1-lmin)/2, fmt='ro')
errorbar(lcenter, btot_binned, yerr=dclsb, xerr = (lmax+1-lmin)/2, fmt='ro')

fmBicep, der = fishermatrix(centervals, bins, dclb, get_spectrum_bins, allargs)
fmBicepsvl, der = fishermatrix(centervals, bins, dclsb, get_spectrum_bins, allargs, der=der)

clf()
a1=plot_fisher(fmBicepsvl, centervals, parnames,'r')
a0=plot_fisher(fmBicep, centervals, parnames,'b')
subplot(3,3,3)
axis('off')
legend([a0,a1],['BICEP2 Current','BICEP2 s.v. limited'])

clf()
a1=plot_fisher(fmBicepsvl, centervals, parnames,'r',fixed=1)
a0=plot_fisher(fmBicep, centervals, parnames,'b',fixed=1)
subplot(3,3,3)
axis('off')
legend([a0,a1],['BICEP2 Current','BICEP2 s.v. limited'])
############################################################################


#### sample experimental configuration  QUBIC ##################################
totsky = (4*np.pi*(180/np.pi)**2)
nbsqdeg = 400.
fsky = nbsqdeg/totsky
mukarcmin = 3.5 
nkdeg = mukarcmin*1000/60
beamsigma = 0.23
fwhmdeg = beamsigma*2.35
deltal = 35
min_ell = 20
nbins = 10
lmin = min_ell+(deltal)*np.arange(nbins)
lmax = deltal-1+min_ell+(deltal)*np.arange(10)
ell = np.arange(np.max(lmax)+100)+1
lcenter = (lmax+lmin)/2
allargs = [params, lmax, ell, Bl, fsky, mukarcmin, fwhmdeg, deltal, lmin, lmax]

btot_binned, clprim_binned, cllens_binned, dclsb, dclnb, dclb, Bprim, Blensed, totBB = get_spectrum_bins(pars, lcenter, allargs)
bins = lcenter
fmQubic, der = fishermatrix(centervals, bins, dclb, get_spectrum_bins, allargs)
fmQubicsvl, der = fishermatrix(centervals, bins, dclsb, get_spectrum_bins, allargs, der=der)

clf()
plot(ell, Bprim, 'k--')
plot(ell, Blensed, 'k:')
plot(ell, totBB, 'k')
xlim(0,max(lmax))
errorbar(lcenter, btot_binned, yerr=dclb, xerr = (lmax+1-lmin)/2, fmt='ro')
errorbar(lcenter, btot_binned, yerr=dclsb, xerr = (lmax+1-lmin)/2, fmt='ro')

clf()
a1=plot_fisher(fmQubicsvl, centervals, parnames,'r')
a0=plot_fisher(fmQubic, centervals, parnames,'b')
subplot(3,3,3)
axis('off')
legend([a0,a1],['QUBIC 1 year','QUBIC s.v. limited'])

clf()
a1=plot_fisher(fmQubicsvl, centervals, parnames,'r',fixed=1)
a0=plot_fisher(fmQubic, centervals, parnames,'b',fixed=1)
subplot(3,3,3)
axis('off')
legend([a0,a1],['QUBIC 1 year','QUBIC s.v. limited'])
############################################################################

clf()
a1=plot_fisher(fmQubic, centervals, parnames,'r',fixed=1)
a0=plot_fisher(fmBicep, centervals, parnames,'b',fixed=1)
subplot(3,3,3)
axis('off')
legend([a0,a1],['BICEP2 Current','QUBIC 1 year'])

clf()
a1=plot_fisher(fmQubic, centervals, parnames,'r')
a0=plot_fisher(fmBicep, centervals, parnames,'b')
subplot(3,3,3)
axis('off')
legend([a0,a1],['BICEP2 Current','QUBIC 1 year'])

clf()
a1=plot_fisher(fmQubicsvl, centervals, parnames,'r',fixed=1)
a0=plot_fisher(fmBicepsvl, centervals, parnames,'b',fixed=1)
subplot(3,3,3)
axis('off')
legend([a0,a1],['BICEP2 s.v.l.','QUBIC s.v.l.'])

clf()
a1=plot_fisher(fmQubicsvl, centervals, parnames,'r')
a0=plot_fisher(fmBicepsvl, centervals, parnames,'b')
subplot(3,3,3)
axis('off')
legend([a0,a1],['BICEP2 s.v.l.','QUBIC s.v.l.'])



#### sample experimental configuration High Resolution change lensing amplitude ##################################
totsky = (4*np.pi*(180/np.pi)**2)
nbsqdeg = totsky
fsky = nbsqdeg/totsky
mukarcmin = 1. 
nkdeg = mukarcmin*1000/60
beamsigma = 1./60./2.35
fwhmdeg = beamsigma*2.35
deltal = 20
min_ell = 2
nbins = 100
lmin = min_ell+(deltal)*np.arange(nbins)
lmax = deltal-1+min_ell+(deltal)*np.arange(nbins)
maxellbin = 1. / np.radians(fwhmdeg / 2.35)
mask = lmax < maxellbin
lmin = lmin[mask]
lmax = lmax[mask]
ell = np.arange(np.max(lmax)+100)+1
lcenter = (lmax+lmin)/2
allargs = [params, lmax, ell, Bl, fsky, mukarcmin, fwhmdeg, deltal, lmin, lmax]

pars = np.array([0.2, -8, 0.0])
btot_binned, clprim_binned, cllens_binned, dclsb, dclnb, dclb, Bprim, Blensed, totBB = get_spectrum_bins(pars, lcenter, allargs)
bins = lcenter

clf()
plot(ell, Bprim, 'k--')
plot(ell, Blensed, 'k:')
plot(ell, totBB, 'k')
#errorbar(lcenter, btot_binned, yerr=dclb, xerr = (lmax+1-lmin)/2, fmt='ro')
errorbar(lcenter, btot_binned, yerr=dclsb, xerr = (lmax+1-lmin)/2, fmt='ro')
yscale('log')

fmsvl, der = fishermatrix(pars, bins, dclsb, get_spectrum_bins, allargs)
fm, der = fishermatrix(pars, bins, dclb, get_spectrum_bins, allargs, der=der)
############################################################################



sigmas = np.sqrt(np.diag(np.linalg.inv(submatrix(fm,[0,1]))))
sigmas_svl = np.sqrt(np.diag(np.linalg.inv(submatrix(fmsvl,[0,1]))))

clf()
a1=plot_fisher(fmsvl, centervals, parnames,'r',fixed=2)
a0=plot_fisher(fm, centervals, parnames,'b',fixed=2)
subplot(3,3,3)
axis('off')
legend([a1,a0],['s.v.l. $\sigma_r$={0:.4f} - $\sigma_\kappa$={1:.1f}'.format(sigmas_svl[0],sigmas_svl[1]),'Noisy $\sigma_r$={0:.4f} - $\sigma_\kappa$={1:.1f}'.format(sigmas[0],sigmas[1])])

clf()
a1=plot_fisher(submatrix(fmsvl,[0,1]), centervals[0:2], parnames[0:2],'r')
a0=plot_fisher(submatrix(fm,[0,1]), centervals[0:2], parnames[0:2],'b')
subplot(3,3,3)
axis('off')
legend([a1,a0],['s.v.l. $\sigma_r$={0:.4f} - $\sigma_\kappa$={1:.1f}'.format(sigmas_svl[0],sigmas_svl[1]),'Noisy $\sigma_r$={0:.4f} - $\sigma_\kappa$={1:.1f}'.format(sigmas[0],sigmas[1])])

##### Def configuration
mukarcmin = 1.
fwhmdeg = 1./60 
fsky = 1. 
deltal = 20
min_ell = 2
nbins = 100
residual_lensing = np.array([0.01, 0.1, 1])

sig = np.zeros(len(residual_lensing))
sig_svl = np.zeros(len(residual_lensing))
for i in np.arange(len(residual_lensing)):
	s, s_svl = get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, deltal, min_ell, nbins, residual_lensing[i])
	sig[i] = s[1]
	sig_svl[i] = s_svl[1]

clf()
plot(residual_lensing, sig, 'b')
plot(residual_lensing, sig_svl, 'r')
xscale('log')

##### degraded resolution 5 arcmin
mukarcmin = 1.
fwhmdeg = 5./60 
fsky = 1. 
deltal = 20
min_ell = 2
nbins = 100
residual_lensing = np.array([0.01, 0.1, 1])

sig_5arcmin = np.zeros(len(residual_lensing))
sig_svl_5arcmin = np.zeros(len(residual_lensing))
for i in np.arange(len(residual_lensing)):
	s, s_svl = get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, deltal, min_ell, nbins,  residual_lensing[i])
	sig_5arcmin[i] = s[1]
	sig_svl_5arcmin[i] = s_svl[1]


##### degraded resolution 30 arcmin
mukarcmin = 1.
fwhmdeg = 30./60 
fsky = 1. 
deltal = 20
min_ell = 2
nbins = 100
residual_lensing = np.array([0.01, 0.1, 1])

sig_30arcmin = np.zeros(len(residual_lensing))
sig_svl_30arcmin = np.zeros(len(residual_lensing))
for i in np.arange(len(residual_lensing)):
	s, s_svl = get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, deltal, min_ell, nbins,  residual_lensing[i])
	sig_30arcmin[i] = s[1]
	sig_svl_30arcmin[i] = s_svl[1]

##### degraded fsky
mukarcmin = 1.
fwhmdeg = 1./60 
fsky = 0.5 
deltal = 20
min_ell = 10
nbins = 100
residual_lensing = np.array([0.01, 0.1, 1])

sig_halfsky = np.zeros(len(residual_lensing))
sig_svl_halfsky = np.zeros(len(residual_lensing))
for i in np.arange(len(residual_lensing)):
	s, s_svl = get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, deltal, min_ell, nbins,  residual_lensing[i])
	sig_halfsky[i] = s[1]
	sig_svl_halfsky[i] = s_svl[1]


##### small patch
mukarcmin = 1.
fwhmdeg = 1./60 
fsky = 0.05 
deltal = 20
min_ell = 20
nbins = 100
residual_lensing = np.array([0.01, 0.1, 1])

sig_small = np.zeros(len(residual_lensing))
sig_svl_small = np.zeros(len(residual_lensing))
for i in np.arange(len(residual_lensing)):
	s, s_svl = get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, deltal, min_ell, nbins, residual_lensing[i])
	sig_small[i] = s[1]
	sig_svl_small[i] = s_svl[1]



clf()
plot(residual_lensing, sig, 'b',lw=3,label='FWHM=1arcmin - Noise=1$\mu$k.arcmin - fsky=1')
plot(residual_lensing, sig_svl, 'b--',lw=3)
plot(residual_lensing, sig_halfsky, 'g',lw=3,label='FWHM=1arcmin - Noise=1$\mu$k.arcmin - fsky=0.5')
plot(residual_lensing, sig_svl_halfsky, 'g--',lw=3)
plot(residual_lensing, sig_small, 'r',lw=3,label='FWHM=1arcmin - Noise=1$\mu$k.arcmin - fsky=0.05')
plot(residual_lensing, sig_svl_small, 'r--',lw=3)
xscale('log')
#yscale('log')
xlabel('Residual fraction of Lensing B-mode')
ylabel('$\sigma ( r/n_T )$')
legend(loc='upper left')


clf()
plot(residual_lensing, sig, 'b',lw=3,label='FWHM=1arcmin - Noise=1$\mu$k.arcmin - fsky=1')
plot(residual_lensing, sig_svl, 'b--',lw=3)
plot(residual_lensing, sig_5arcmin, 'g',lw=3,label='FWHM=5arcmin - Noise=1$\mu$k.arcmin - fsky=1')
plot(residual_lensing, sig_svl_5arcmin, 'g--',lw=3)
plot(residual_lensing, sig_30arcmin, 'r',lw=3,label='FWHM=30arcmin - Noise=1$\mu$k.arcmin - fsky=1')
plot(residual_lensing, sig_svl_30arcmin, 'r--',lw=3)
xscale('log')
#yscale('log')
xlabel('Residual fraction of Lensing B-mode')
ylabel('$\sigma ( r/n_T )$')
legend(loc='upper left')


#################### Very general
mukarcmin = 1.
fwhmdeg = np.array([1., 5., 10., 30.])/60
fsky = np.array([0.05, 0.1, 0.5, 1.])
lens_res = np.array([0., 0.01, 0.1, 1.])
nbins = 1000
delta_ell = 10

der = None
spectra = None
s, ssvl, der, spectra = get_tratio_accuracy(params, 1., 1., 0.5, deltal, min_ell, nbins, lens_res[i], der=der, spectra = spectra)

sigs_r = np.zeros((len(lens_res), len(fsky), len(fwhmdeg)))
sigs_r_svl = np.zeros((len(lens_res), len(fsky), len(fwhmdeg)))
sigs_rt = np.zeros((len(lens_res), len(fsky), len(fwhmdeg)))
sigs_rt_svl = np.zeros((len(lens_res), len(fsky), len(fwhmdeg)))
for i in np.arange(len(lens_res)):
	for j in np.arange(len(fsky)):
		for k in np.arange(len(fwhmdeg)):
			print('Num:',i,j,k)
			min_ell = np.int(180/np.degrees((sqrt(2*fsky[j]))))
			s, s_svl, der, spectra = get_tratio_accuracy(params, fsky[j], mukarcmin, fwhmdeg[k], deltal, min_ell, nbins, lens_res[i], der=der, spectra=spectra)
			sigs_r[i,j,k] = s[0]
			sigs_r_svl[i,j,k] = s_svl[0]
			sigs_rt[i,j,k] = s[1]
			sigs_rt_svl[i,j,k] = s_svl[1]

