from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import pycamb
#from qubic.utils import progress_bar
from Homogeneity import fitting
from Cosmo import FisherNew
from Cosmo import CMBspectra

##### Base Cosmology
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
rvalue = 0.1
print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',omegav/h2
params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':rvalue,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False,
         'tensor_index':-rvalue/8}


################## First run to calculate derivatives
fsky = 384/41000
mukarcminT = 87/1000*60
fwhmdeg = 0.52
pars = {'tensor_ratio':0.2, 'tensor_index':-0.025, 'lensing_residual':1, 'scalar_index':0.9624}
varnames = {'tensor_ratio':'$r$', 'tensor_index':'$n_T$', 'lensing_residual':'$a_L$', 'scalar_index':'$n_S$'}
fm, basic_spec, der, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, deltal=25)

clf()
nbins = len(data)/4
yscale('log')
ylim(1e-4,1e4)
#xlim(0,np.max(lmax[mask]))
plot(basic_spec[0], basic_spec[1],'k', lw=2)
plot(basic_spec[0], basic_spec[2],'g', lw=2)
plot(basic_spec[0], np.abs(basic_spec[3]),'b', lw=2)
plot(basic_spec[0], basic_spec[4],'m', lw=2)
plot(basic_spec[0], basic_spec[5],'y', lw=2)
plot(basic_spec[0], basic_spec[4]+pars['lensing_residual']*basic_spec[5],'r', lw=2)
errorbar(lcenter, data[0*nbins:1*nbins], yerr=error[0*nbins:1*nbins], fmt='ko', alpha=0.6)
errorbar(lcenter, data[1*nbins:2*nbins], yerr=error[1*nbins:2*nbins], fmt='go', alpha=0.6)
errorbar(lcenter, np.abs(data[2*nbins:3*nbins]), yerr=error[2*nbins:3*nbins], fmt='bo', alpha=0.6)
errorbar(lcenter, data[3*nbins:4*nbins], yerr=error[3*nbins:4*nbins], fmt='ro', alpha=0.6)

############# BICEP2 COnfiguration ######################
fsky = 384/41000
mukarcminT = 87/1000*60
fwhmdeg = 0.52
pars['lensing_residual'] = 1
fmbicep2, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, noT=True, noE=True, noTE=True, deltal=25, varnames=varnames, maxell = 1./np.radians(fwhmdeg / 2.35))


############# QUBIC COnfiguration ######################
fsky = 0.01
mukarcminT = 4.
pars['lensing_residual'] = 1
fwhmdeg = 0.52
fmqubic, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, noT=True, noE=True, noTE=True, deltal=25, varnames=varnames, maxell = 1./np.radians(fwhmdeg / 2.35))

# QUBIC reoptimized
fsky = 0.04
mukarcminT = 4.*(fsky/0.01)**0.5
pars['lensing_residual'] = 1
fwhmdeg = 0.52
fmqubicnew, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, noT=True, noE=True, noTE=True, deltal=25, varnames=varnames, maxell = 1./np.radians(fwhmdeg / 2.35))

# QUBIC reoptimized
fsky = 0.04
mukarcminT = 4.*(fsky/0.01)**0.5/sqrt(6)
pars['lensing_residual'] = 1
fwhmdeg = 0.52
fmqubicnew6, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, noT=True, noE=True, noTE=True, deltal=25, varnames=varnames, maxell = 1./np.radians(fwhmdeg / 2.35))



clf()
a1,leg1 = FisherNew.plot_fisher(fmqubic, pars,'b', fixed=['tensor_index','scalar_index'], varnames=varnames, onesigma=True)
a2,leg2 = FisherNew.plot_fisher(fmqubicnew, pars,'g', fixed=['tensor_index','scalar_index'], varnames=varnames, onesigma=True)
a3,leg3 = FisherNew.plot_fisher(fmqubicnew6, pars,'r', fixed=['tensor_index','scalar_index'], varnames=varnames, onesigma=True)
a0,leg0 = FisherNew.plot_fisher(fmbicep2, pars,'k', fixed=['tensor_index','scalar_index'], varnames=varnames, onesigma=True)
subplot(4,4,4)
axis('off')
legend([a0, a1, a2, a3],[leg0, leg1, leg2, leg3])


################# satellites
pars = {'tensor_ratio':0.01, 'tensor_index':-0.025, 'lensing_residual':1, 'scalar_index':0.9624}
varnames = {'tensor_ratio':'$r$', 'tensor_index':'$n_T$', 'lensing_residual':'$a_L$', 'scalar_index':'$n_S$'}
fm, basic_spec, der, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, deltal=25)
nbins=len(data)/4
noT=False
noE=False
noTE=False

#### Assume a CMB stage 4 experiment
fsky = 0.2
mukarcminT = 1.
pars['lensing_residual'] = 0.1
fwhmdeg = 3./60
fm_stage4, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)


#### Assume a full sky low resolution satellite
fsky = 0.5
mukarcminT = 2.
pars['lensing_residual'] = 1
fwhmdeg = 30./60
fm_lrs, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)

## CORE+ / PRISM
fsky = 0.75
mukarcminT = 1.
pars['lensing_residual'] = 0.1
fwhmdeg = 4./60
fm_hrs, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)
errhrs = np.reshape(error,(4,nbins))


## CORE++ ?
fsky = 0.75
mukarcminT = 0.1
pars['lensing_residual'] = 0.1
fwhmdeg = 3./60
fm_hrsplus, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)
errhrsplus = np.reshape(error,(4,nbins))


clf()
limits = [[0.953,0.973], [0.090, 0.11], [-0.15,0.1], [0.16,0.24]]
a1,leg1 = FisherNew.plot_fisher(fm_stage4, pars,'b', varnames=varnames, onesigma=True, limits=limits)
a2,leg2 = FisherNew.plot_fisher(fm_lrs, pars,'g', varnames=varnames, onesigma=True, limits=limits)
a4,leg4 = FisherNew.plot_fisher(fm_lrs+fm_stage4, pars,'m', varnames=varnames, onesigma=True, limits=limits)
a3,leg3 = FisherNew.plot_fisher(fm_hrs, pars,'r', varnames=varnames, onesigma=True, limits=limits)
a0,leg0 = FisherNew.plot_fisher(fm_hrsplus, pars,'k', varnames=varnames, onesigma=True, limits=limits)
subplot(4,4,4)
axis('off')
legend([a1, a2, a4, a3, a0],['Stage 4: '+leg1, 'LRS: '+leg2, 'Stage 4 + LRS: '+leg4, 'HRS: '+leg3, 'HRS++: '+leg0],loc='upper right',title='TT+EE+TE+BB')
savefig('allspectra.png')


clf()
limits = [[0.090, 0.11], [-0.15,0.1], [0.16,0.24]]
a1,leg1 = FisherNew.plot_fisher(fm_stage4, pars,'b', varnames=varnames, fixed=['scalar_index'], onesigma=True, limits=limits)
a2,leg2 = FisherNew.plot_fisher(fm_lrs, pars,'g', varnames=varnames, fixed=['scalar_index'], onesigma=True, limits=limits)
a4,leg4 = FisherNew.plot_fisher(fm_lrs+fm_stage4, pars,'m', varnames=varnames, fixed=['scalar_index'], onesigma=True, limits=limits)
a3,leg3 = FisherNew.plot_fisher(fm_hrs, pars,'r', varnames=varnames, fixed=['scalar_index'], onesigma=True, limits=limits)
a0,leg0 = FisherNew.plot_fisher(fm_hrsplus, pars,'k', varnames=varnames, fixed=['scalar_index'], onesigma=True, limits=limits)
subplot(4,4,4)
axis('off')
legend([a1, a2, a4, a3, a0],['Stage 4: '+leg1, 'LRS: '+leg2, 'Stage 4 + LRS: '+leg4, 'HRS: '+leg3, 'HRS++: '+leg0],loc='upper right',title='TT+EE+TE+BB - Fixed $n_S$')
savefig('allspectra_nons.png')

#### get the correct matrix for r and ns
## fix nt
sub_fm_stage4 = FisherNew.submatrix(fm_stage4,[0,1,3])
sub_fm_lrs = FisherNew.submatrix(fm_lrs,[0,1,3])
sub_fm_lrs_stage4 = FisherNew.submatrix(fm_stage4+fm_lrs,[0,1,3])
sub_fm_hrs = FisherNew.submatrix(fm_hrs,[0,1,3])
sub_fm_hrsplus = FisherNew.submatrix(fm_hrsplus,[0,1,3])

## Marginalize over al and reverse order for having ns, r
cov_nsr_stage4 = FisherNew.submatrix(np.linalg.inv(sub_fm_stage4), [0,2])
cov_nsr_lrs = FisherNew.submatrix(np.linalg.inv(sub_fm_lrs), [0,2])
cov_nsr_lrs_stage4 = FisherNew.submatrix(np.linalg.inv(sub_fm_lrs_stage4), [0,2])
cov_nsr_hrs = FisherNew.submatrix(np.linalg.inv(sub_fm_hrs), [0,2])
cov_nsr_hrsplus = FisherNew.submatrix(np.linalg.inv(sub_fm_hrsplus), [0,2])

fm_nsr_lrs = np.linalg.inv(cov_nsr_lrs)
fm_nsr_stage4 = np.linalg.inv(cov_nsr_stage4)
fm_nsr_lrs_stage4 = np.linalg.inv(cov_nsr_lrs_stage4)
fm_nsr_hrs = np.linalg.inv(cov_nsr_hrs)
fm_nsr_hrsplus = np.linalg.inv(cov_nsr_hrsplus)

fm_nsr_margnt_stage4 = np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_stage4), [0,3]))
fm_nsr_margnt_lrs = np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_lrs), [0,3]))
fm_nsr_margnt_lrs_stage4 = np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_lrs+fm_stage4), [0,3]))
fm_nsr_margnt_hrs = np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_hrs), [0,3]))
fm_nsr_margnt_hrsplus = np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_hrsplus), [0,3]))


clf()
xlim(0.8,1.1)
ylim(0,0.6)
xlabel('$n_s$',fontsize=20)
ylabel('$r$',fontsize=20)
al = 0.3
aa = fill_between([0.8, 1], [0,0], [0.45, 0],color='b', alpha=al)
text(0.82,0.18,'Small Field',fontsize=20)
bb = fill_between([0.8, 1], [0.45, 0], [1.4,0],color='r', alpha=al)
text(0.85,0.4,'Large Field',fontsize=20)
cc = fill_between([0.8, 1.1], [1.4,-0.7], [1e8,0],color='y', alpha=al)
text(1.02,0.3,'Hybrid',fontsize=20)
nsval = pars['scalar_index']
rval = pars['tensor_ratio']
a1=FisherNew.cont_from_fisher2d(fm_nsr_margnt_stage4, [nsval, rval] ,color='b', onesigma=True)
a2=FisherNew.cont_from_fisher2d(fm_nsr_margnt_lrs, [nsval, rval] ,color='g', onesigma=True)
a4=FisherNew.cont_from_fisher2d(fm_nsr_margnt_lrs_stage4, [nsval, rval] ,color='m', onesigma=True)
a3=FisherNew.cont_from_fisher2d(fm_nsr_margnt_hrs, [nsval, rval] ,color='r', onesigma=True)
a0=FisherNew.cont_from_fisher2d(fm_nsr_margnt_hrsplus, [nsval, rval] ,color='k', onesigma=True)
plot([nsval-0.0054,nsval-0.0054],[0,1],'k:')
plot([nsval+0.0054,nsval+0.0054],[0,1],'k:')
legend([a1, a2, a4, a3, a0],['Stage 4', 'LRS', 'Stage 4 + LRS', 'HRS: ', 'HRS++'],loc='upper right',title='TT+EE+TE+BB (marginalising over $n_T, a_L$)', fontsize=15, frameon=0)
savefig('inflationplot_0.1.png')

clf()
xlim(0.956,0.97)
ylim(0.,0.3)
xlabel('$n_s$',fontsize=20)
ylabel('$r$',fontsize=20)
al = 0.3
aa = fill_between([0.8, 1], [0,0], [0.45, 0],color='b', alpha=al)
#text(0.82,0.18,'Small Field',fontsize=20)
bb = fill_between([0.8, 1], [0.45, 0], [1.4,0],color='r', alpha=al)
text(0.961,0.12,'Large Field',fontsize=20)
cc = fill_between([0.8, 1.1], [1.4,-0.7], [1e8,0],color='y', alpha=al)
text(0.961,0.28,'Hybrid',fontsize=20)
nsval = params['scalar_index']
rval = params['tensor_ratio']
a1=FisherNew.cont_from_fisher2d(fm_nsr_margnt_stage4, [nsval, rval] ,color='b', onesigma=True)
a2=FisherNew.cont_from_fisher2d(fm_nsr_margnt_lrs, [nsval, rval] ,color='g', onesigma=True)
a4=FisherNew.cont_from_fisher2d(fm_nsr_margnt_lrs_stage4, [nsval, rval] ,color='m', onesigma=True)
a3=FisherNew.cont_from_fisher2d(fm_nsr_margnt_hrs, [nsval, rval] ,color='r', onesigma=True)
a0=FisherNew.cont_from_fisher2d(fm_nsr_margnt_hrsplus, [nsval, rval] ,color='k', onesigma=True)
plot([nsval-0.0054,nsval-0.0054],[0,1],'k:')
plot([nsval+0.0054,nsval+0.0054],[0,1],'k:')
legend([a1, a2, a4, a3, a0],['Stage 4', 'LRS', 'Stage 4 + LRS', 'HRS: ', 'HRS++'],loc='upper right',title='TT+EE+TE+BB \n(marginalising over $n_T, a_L$)', fontsize=15, frameon=0)
savefig('inflationplot_zoom_0.1.png')

clf()
xlim(0.956,0.97)
ylim(0.,0.3)
xlabel('$n_s$',fontsize=20)
ylabel('$r$',fontsize=20)
al = 0.3
aa = fill_between([0.8, 1], [0,0], [0.45, 0],color='b', alpha=al)
#text(0.82,0.18,'Small Field',fontsize=20)
bb = fill_between([0.8, 1], [0.45, 0], [1.4,0],color='r', alpha=al)
text(0.961,0.12,'Large Field',fontsize=20)
cc = fill_between([0.8, 1.1], [1.4,-0.7], [1e8,0],color='y', alpha=al)
text(0.961,0.28,'Hybrid',fontsize=20)
nsval = params['scalar_index']
rval = params['tensor_ratio']
a1=FisherNew.cont_from_fisher2d(fm_nsr_stage4, [nsval, rval] ,color='b', onesigma=True)
a2=FisherNew.cont_from_fisher2d(fm_nsr_lrs, [nsval, rval] ,color='g', onesigma=True)
a4=FisherNew.cont_from_fisher2d(fm_nsr_lrs_stage4, [nsval, rval] ,color='m', onesigma=True)
a3=FisherNew.cont_from_fisher2d(fm_nsr_hrs, [nsval, rval] ,color='r', onesigma=True)
a0=FisherNew.cont_from_fisher2d(fm_nsr_hrsplus, [nsval, rval] ,color='k', onesigma=True)
plot([nsval-0.0054,nsval-0.0054],[0,1],'k:')
plot([nsval+0.0054,nsval+0.0054],[0,1],'k:')
legend([a1, a2, a4, a3, a0],['Stage 4', 'LRS', 'Stage 4 + LRS', 'HRS: ', 'HRS++'],loc='upper right',title='TT+EE+TE+BB \n(marginalising over $a_L$, fixing $n_T$)', fontsize=15, frameon=0)
savefig('inflationplot_zoom_fixednt_0.1.png')



#### Same with BB only
noT=True
noE=True
noTE=True

#### Assume a CMB stage 4 experiment
fsky = 0.2
mukarcminT = 1.
pars['lensing_residual'] = 0.1
fwhmdeg = 3./60
fm_stage4, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)


#### Assume a full sky low resolution satellite
fsky = 0.5
mukarcminT = 2
pars['lensing_residual'] = 1
fwhmdeg = 30./60
fm_lrs, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)

## CORE+ / PRISM
fsky = 0.75
mukarcminT = 1.5
pars['lensing_residual'] = 0.1
fwhmdeg = 4./60
fm_hrs, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)

## CORE++ ?
fsky = 0.75
mukarcminT = 0.1
pars['lensing_residual'] = 0.1
fwhmdeg = 3./60
fm_hrsplus, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)


clf()
limits = [[0.090, 0.11], [-0.15,0.1], [0.16,0.24]]
a1,leg1 = FisherNew.plot_fisher(fm_stage4, pars,'b', varnames=varnames, fixed=['scalar_index'], onesigma=True, limits=limits)
a2,leg2 = FisherNew.plot_fisher(fm_lrs, pars,'g', varnames=varnames, fixed=['scalar_index'], onesigma=True, limits=limits)
a4,leg4 = FisherNew.plot_fisher(fm_lrs+fm_stage4, pars,'m', varnames=varnames, fixed=['scalar_index'], onesigma=True, limits=limits)
a3,leg3 = FisherNew.plot_fisher(fm_hrs, pars,'r', varnames=varnames, fixed=['scalar_index'], onesigma=True, limits=limits)
a0,leg0 = FisherNew.plot_fisher(fm_hrsplus, pars,'k', varnames=varnames, fixed=['scalar_index'], onesigma=True, limits=limits)
subplot(4,4,4)
axis('off')
legend([a1, a2, a4, a3, a0],['Stage 4: '+leg1, 'LRS: '+leg2, 'Stage 4 + LRS: '+leg4, 'HRS: '+leg3, 'HRS++: '+leg0],loc='upper right',title='BB only - Fixed $n_S$')
savefig('BBonly_nons.png')

clf()
limits = [[0.953,0.973], [0.090, 0.11], [-0.15,0.1], [0.16,0.24]]
a1,leg1 = FisherNew.plot_fisher(fm_stage4, pars,'b', varnames=varnames, onesigma=True, limits=limits)
a2,leg2 = FisherNew.plot_fisher(fm_lrs, pars,'g', varnames=varnames, onesigma=True, limits=limits)
a4,leg4 = FisherNew.plot_fisher(fm_lrs+fm_stage4, pars,'m', varnames=varnames, onesigma=True, limits=limits)
a3,leg3 = FisherNew.plot_fisher(fm_hrs, pars,'r', varnames=varnames, onesigma=True, limits=limits)
a0,leg0 = FisherNew.plot_fisher(fm_hrsplus, pars,'k', varnames=varnames, onesigma=True, limits=limits)
subplot(4,4,4)
axis('off')
legend([a1, a2, a4, a3, a0],['Stage 4: '+leg1, 'LRS: '+leg2, 'Stage 4 + LRS: '+leg4, 'HRS: '+leg3, 'HRS++: '+leg0],loc='upper right',title='BB only')
savefig('BBonly.png')



#### get the correct matrix for r and ns
## fix nt
sub_fm_stage4 = FisherNew.submatrix(fm_stage4,[0,1,3])
sub_fm_lrs = FisherNew.submatrix(fm_lrs,[0,1,3])
sub_fm_lrs_stage4 = FisherNew.submatrix(fm_stage4+fm_lrs,[0,1,3])
sub_fm_hrs = FisherNew.submatrix(fm_hrs,[0,1,3])
sub_fm_hrsplus = FisherNew.submatrix(fm_hrsplus,[0,1,3])

## Marginalize over al and reverse order for having ns, r
cov_nsr_stage4 = FisherNew.submatrix(np.linalg.inv(sub_fm_stage4), [0,2])
cov_nsr_lrs = FisherNew.submatrix(np.linalg.inv(sub_fm_lrs), [0,2])
cov_nsr_lrs_stage4 = FisherNew.submatrix(np.linalg.inv(sub_fm_lrs_stage4), [0,2])
cov_nsr_hrs = FisherNew.submatrix(np.linalg.inv(sub_fm_hrs), [0,2])
cov_nsr_hrsplus = FisherNew.submatrix(np.linalg.inv(sub_fm_hrsplus), [0,2])

fm_nsr_lrs = np.linalg.inv(cov_nsr_lrs)
fm_nsr_stage4 = np.linalg.inv(cov_nsr_stage4)
fm_nsr_lrs_stage4 = np.linalg.inv(cov_nsr_lrs_stage4)
fm_nsr_hrs = np.linalg.inv(cov_nsr_hrs)
fm_nsr_hrsplus = np.linalg.inv(cov_nsr_hrsplus)

fm_nsr_margnt_stage4 = np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_stage4), [0,3]))
fm_nsr_margnt_lrs = np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_lrs), [0,3]))
fm_nsr_margnt_lrs_stage4 = np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_lrs+fm_stage4), [0,3]))
fm_nsr_margnt_hrs = np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_hrs), [0,3]))
fm_nsr_margnt_hrsplus = np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_hrsplus), [0,3]))


clf()
xlim(0.8,1.1)
ylim(0,0.6)
xlabel('$n_s$',fontsize=20)
ylabel('$r$',fontsize=20)
al = 0.3
aa = fill_between([0.8, 1], [0,0], [0.45, 0],color='b', alpha=al)
text(0.82,0.18,'Small Field',fontsize=20)
bb = fill_between([0.8, 1], [0.45, 0], [1.4,0],color='r', alpha=al)
text(0.85,0.4,'Large Field',fontsize=20)
cc = fill_between([0.8, 1.1], [1.4,-0.7], [1e8,0],color='y', alpha=al)
text(1.02,0.3,'Hybrid',fontsize=20)
nsval = params['scalar_index']
rval = params['tensor_ratio']
a1=FisherNew.cont_from_fisher2d(fm_nsr_margnt_stage4, [nsval, rval] ,color='b', onesigma=True)
a2=FisherNew.cont_from_fisher2d(fm_nsr_margnt_lrs, [nsval, rval] ,color='g', onesigma=True)
a4=FisherNew.cont_from_fisher2d(fm_nsr_margnt_lrs_stage4, [nsval, rval] ,color='m', onesigma=True)
a3=FisherNew.cont_from_fisher2d(fm_nsr_margnt_hrs, [nsval, rval] ,color='r', onesigma=True)
a0=FisherNew.cont_from_fisher2d(fm_nsr_margnt_hrsplus, [nsval, rval] ,color='k', onesigma=True)
legend([a1, a2, a4, a3, a0],['Stage 4', 'LRS', 'Stage 4 + LRS', 'HRS: ', 'HRS++'],loc='upper right',title='BB Only \n(marginalising over $n_T, a_L$)', fontsize=15, frameon=0)
savefig('inflationplot_BBonly.png')


clf()
xlim(0.8,1.1)
ylim(0,0.6)
xlabel('$n_s$',fontsize=20)
ylabel('$r$',fontsize=20)
al = 0.3
aa = fill_between([0.8, 1], [0,0], [0.45, 0],color='b', alpha=al)
text(0.82,0.18,'Small Field',fontsize=20)
bb = fill_between([0.8, 1], [0.45, 0], [1.4,0],color='r', alpha=al)
text(0.85,0.4,'Large Field',fontsize=20)
cc = fill_between([0.8, 1.1], [1.4,-0.7], [1e8,0],color='y', alpha=al)
text(1.02,0.3,'Hybrid',fontsize=20)
nsval = params['scalar_index']
rval = params['tensor_ratio']
a1=FisherNew.cont_from_fisher2d(fm_nsr_stage4, [nsval, rval] ,color='b', onesigma=True)
a2=FisherNew.cont_from_fisher2d(fm_nsr_lrs, [nsval, rval] ,color='g', onesigma=True)
a4=FisherNew.cont_from_fisher2d(fm_nsr_lrs_stage4, [nsval, rval] ,color='m', onesigma=True)
a3=FisherNew.cont_from_fisher2d(fm_nsr_hrs, [nsval, rval] ,color='r', onesigma=True)
a0=FisherNew.cont_from_fisher2d(fm_nsr_hrsplus, [nsval, rval] ,color='k', onesigma=True)
legend([a1, a2, a4, a3, a0],['Stage 4', 'LRS', 'Stage 4 + LRS', 'HRS: ', 'HRS++'],loc='upper right',title='BB Only (marginalising over $n_T, a_L$)', fontsize=15, frameon=0)
savefig('inflationplot_BBonly_ntfixed.png')

clf()
xlim(0.957,0.97)
ylim(0.1,0.3)
xlabel('$n_s$',fontsize=20)
ylabel('$r$',fontsize=20)
al = 0.3
aa = fill_between([0.8, 1], [0,0], [0.45, 0],color='b', alpha=al)
#text(0.82,0.18,'Small Field',fontsize=20)
bb = fill_between([0.8, 1], [0.45, 0], [1.4,0],color='r', alpha=al)
text(0.961,0.12,'Large Field',fontsize=20)
cc = fill_between([0.8, 1.1], [1.4,-0.7], [1e8,0],color='y', alpha=al)
text(0.961,0.28,'Hybrid',fontsize=20)
nsval = params['scalar_index']
rval = params['tensor_ratio']
a1=FisherNew.cont_from_fisher2d(fm_nsr_margnt_stage4, [nsval, rval] ,color='b', onesigma=True)
a2=FisherNew.cont_from_fisher2d(fm_nsr_margnt_lrs, [nsval, rval] ,color='g', onesigma=True)
a4=FisherNew.cont_from_fisher2d(fm_nsr_margnt_lrs_stage4, [nsval, rval] ,color='m', onesigma=True)
a3=FisherNew.cont_from_fisher2d(fm_nsr_margnt_hrs, [nsval, rval] ,color='r', onesigma=True)
a0=FisherNew.cont_from_fisher2d(fm_nsr_margnt_hrsplus, [nsval, rval] ,color='k', onesigma=True)
legend([a1, a2, a4, a3, a0],['Stage 4', 'LRS', 'Stage 4 + LRS', 'HRS: ', 'HRS++'],loc='upper right',title='TT+EE+TE+BB \n(marginalising over $n_T, a_L$)', fontsize=15, frameon=0)
savefig('inflationplot_BBonly_zoom.png')

clf()
xlim(0.957,0.97)
ylim(0.1,0.3)
xlabel('$n_s$',fontsize=20)
ylabel('$r$',fontsize=20)
al = 0.3
aa = fill_between([0.8, 1], [0,0], [0.45, 0],color='b', alpha=al)
#text(0.82,0.18,'Small Field',fontsize=20)
bb = fill_between([0.8, 1], [0.45, 0], [1.4,0],color='r', alpha=al)
text(0.961,0.12,'Large Field',fontsize=20)
cc = fill_between([0.8, 1.1], [1.4,-0.7], [1e8,0],color='y', alpha=al)
text(0.961,0.28,'Hybrid',fontsize=20)
nsval = params['scalar_index']
rval = params['tensor_ratio']
a1=FisherNew.cont_from_fisher2d(fm_nsr_stage4, [nsval, rval] ,color='b', onesigma=True)
a2=FisherNew.cont_from_fisher2d(fm_nsr_lrs, [nsval, rval] ,color='g', onesigma=True)
a4=FisherNew.cont_from_fisher2d(fm_nsr_lrs_stage4, [nsval, rval] ,color='m', onesigma=True)
a3=FisherNew.cont_from_fisher2d(fm_nsr_hrs, [nsval, rval] ,color='r', onesigma=True)
a0=FisherNew.cont_from_fisher2d(fm_nsr_hrsplus, [nsval, rval] ,color='k', onesigma=True)
legend([a1, a2, a4, a3, a0],['Stage 4', 'LRS', 'Stage 4 + LRS', 'HRS: ', 'HRS++'],loc='upper right',title='TT+EE+TE+BB \n(marginalising over $a_L$, fixing $n_T$)', fontsize=15, frameon=0)
savefig('inflationplot_BBonly_zoom_fixednt.png')



##### Try to reproduce Josquin's plots
noT = False
noE = False
noTE = False

# Stage IV
fsky_s4 = 0.2
mukarcminT_s4 = 1.
fwhmdeg_s4 = 3./60
pars['lensing_residual'] = 0.1
fm_stage4, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky_s4, mukarcminT_s4, fwhmdeg_s4, basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)

# COrE+
fsky = 0.75
fwhmdeg = np.array([1., 2., 5., 10.])/60
mukarcminT = np.linspace(0.1,10,100)
lensing_residuals = np.array([0., 0.1, 1])
sigs = np.zeros((len(pars), len(lensing_residuals), len(fwhmdeg), len(mukarcminT)))
sigs_withstage4 = np.zeros((len(pars), len(lensing_residuals), len(fwhmdeg), len(mukarcminT)))
for k in np.arange(len(lensing_residuals)):
	for j in np. arange(len(fwhmdeg)):
		for i in np.arange(len(mukarcminT)):
			print(i,j,k)
			#### Just COrE+
			pars['lensing_residual'] = lensing_residuals[k]
			fm_hrs, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT[i], fwhmdeg[j], basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)
			s,c = FisherNew.give_sigmas(fm_hrs, pars)
			sigs[:, k, j, i] = s
			#### COrE+ and S4 with overlap on the sky
			pars['lensing_residual'] = lensing_residuals[k]
			fm_hrs2, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky-fsky_s4, mukarcminT[i], fwhmdeg[j], basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)
			fm_stage4, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky_s4, mukarcminT_s4, fwhmdeg_s4, basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)
			s2,c2 = FisherNew.give_sigmas(fm_hrs2 + fm_stage4, pars)
			for l in np.arange(len(pars)):
				sigs_withstage4[l, k, j, i] = np.min([s2[l],s[l]])

clf()
parnum = 3
for i in np.arange(4):
	subplot(1,4,i+1)
	title('FWHM = {0:.2g}'.format(fwhmdeg[i]*60))
	xscale('log')
	if i==0: ylabel('$\sigma$('+varnames[pars.keys()[parnum]]+')')
	ylim(0,np.max(sigs[parnum, 2, :, :]))
	fill_between(mukarcminT, sigs[parnum, 0, i, :], y2=sigs[parnum, 2, i, :],color='blue',alpha=0.1)
	fill_between(mukarcminT, sigs_withstage4[parnum, 0, i, :], y2=sigs_withstage4[parnum, 2, i, :],color='orange',alpha=0.1)
	plot(mukarcminT, sigs[parnum, 0, i, :],'r', lw=2)
	plot(mukarcminT, sigs[parnum, 1, i, :],'b', lw=2)
	plot(mukarcminT, sigs[parnum, 2, i, :],'g', lw=2)
	plot(mukarcminT, sigs_withstage4[parnum, 0, i, :],'r--', lw=2)
	plot(mukarcminT, sigs_withstage4[parnum, 1, i, :],'b--', lw=2)
	plot(mukarcminT, sigs_withstage4[parnum, 2, i, :],'g--', lw=2)
	plot(mukarcminT, mukarcminT*0+np.abs(pars[pars.keys()[parnum]])/2,'k:', lw=2)



clf()
parnum = 2
for i in np.arange(4):
	subplot(1,4,i+1)
	title('FWHM = {0:.2g}'.format(fwhmdeg[i]*60))
	xscale('log')
	if i==0: ylabel(varnames[pars.keys()[parnum]]+'/$\sigma$('+varnames[pars.keys()[parnum]]+')')
	ylim(0,np.max(np.abs(pars[pars.keys()[parnum]])/sigs[parnum, :, :, :]))
	fill_between(mukarcminT, np.abs(pars[pars.keys()[parnum]])/sigs[parnum, 0, i, :], y2=np.abs(pars[pars.keys()[parnum]])/sigs[parnum, 2, i, :],color='blue',alpha=0.1)
	fill_between(mukarcminT, np.abs(pars[pars.keys()[parnum]])/sigs_withstage4[parnum, 0, i, :], y2=np.abs(pars[pars.keys()[parnum]])/sigs_withstage4[parnum, 2, i, :],color='orange',alpha=0.1)
	plot(mukarcminT, np.abs(pars[pars.keys()[parnum]])/sigs[parnum, 0, i, :],'r', lw=2)
	plot(mukarcminT, np.abs(pars[pars.keys()[parnum]])/sigs[parnum, 1, i, :],'b', lw=2)
	plot(mukarcminT, np.abs(pars[pars.keys()[parnum]])/sigs[parnum, 2, i, :],'g', lw=2)
	plot(mukarcminT, np.abs(pars[pars.keys()[parnum]])/sigs_withstage4[parnum, 0, i, :],'r--', lw=2)
	plot(mukarcminT, np.abs(pars[pars.keys()[parnum]])/sigs_withstage4[parnum, 1, i, :],'b--', lw=2)
	plot(mukarcminT, np.abs(pars[pars.keys()[parnum]])/sigs_withstage4[parnum, 2, i, :],'g--', lw=2)
	#plot(mukarcminT, mukarcminT*0+np.abs(pars[pars.keys()[parnum]])/2,'k:', lw=2)







############# Explore ground experiment
noT=True
noE=True
noTE=True
pars = {'tensor_ratio':0.2, 'tensor_index':-0.025, 'lensing_residual':1}
varnames = {'tensor_ratio':'$r$', 'tensor_index':'$n_T$', 'lensing_residual':'$a_L$'}
fm, basic_spec, der, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, deltal=25)

#### Assume a CMB stage 4 experiment
fsky = 0.5*0.8
mukarcminT = 1.
pars['lensing_residual'] = 0.1
fwhmdeg = 3./60
fm_stage4, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)

clf()
a0,leg0 = FisherNew.plot_fisher(fm_stage4, pars,'k', varnames=varnames, onesigma=True)
s,c = FisherNew.give_sigmas(fm_stage4, pars)



###### calcul plus réaliste des paramètres
NET_T = 40 #muK.sqrt(s) Optimistic ?
nbols = 10000
fsky = 1.
fwhmdeg = 3./60
duration = 1.
CMBspectra.muKarcmin(NET_T, nbols, fsky, fwhmdeg)


### explore fsky
nbols = 10000
NET_T = 40.
allfsky = linspace(0.01, 1,100)
pars['lensing_residual'] = 0.1
fwhmdeg = 30./60
allsvals = np.zeros((len(pars), len(allfsky)))
for i in np.arange(len(allfsky)):
	print(i)
	mukarcminT = CMBspectra.muKarcmin(NET_T, nbols, allfsky[i], fwhmdeg)
	fm, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, allfsky[i], mukarcminT, fwhmdeg, basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)
	s, c = FisherNew.give_sigmas(fm, pars)
	allsvals[:,i] = s

clf()
for i in np.arange(len(pars)):
	subplot(len(pars),1,i+1)
	ylabel('Nb $\sigma$ '+pars.keys()[i])
	plot(allfsky, np.abs(pars.values()[i]/allsvals[i,:]))
	xlabel('fsky')



#### explore fsky and fwhm
nbols = 10000
NET_T = 40.
allfsky = linspace(0.01, 1,30)
pars['lensing_residual'] = 0.1
allfwhmdeg = linspace(3./60, 60./60, 10)
allsvals = np.zeros((len(pars), len(allfsky), len(allfwhmdeg)))
allmukarcminT = np.zeros((len(allfsky), len(allfwhmdeg)))
for i in np.arange(len(allfsky)):
	for j in np.arange(len(allfwhmdeg)):
		print(i,j)
		mukarcminT = CMBspectra.muKarcmin(NET_T, nbols, allfsky[i], allfwhmdeg[j])
		allmukarcminT[i,j] = mukarcminT
		fm, dum, dum, lcenter, data, error = CMBspectra.fisher_analysis(pars, allfsky[i], mukarcminT, allfwhmdeg[j], basic_spectra=basic_spec, der=der, deltal=25, varnames=varnames, noT=noT, noE=noE, noTE=noTE)
		s, c = FisherNew.give_sigmas(fm, pars)
		allsvals[:,i,j] = s








