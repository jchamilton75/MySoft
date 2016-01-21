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



rvalues = [0.01,0.05, 0.1, 0.2]

allder = []
allbasic_spec = []
for rr in rvalues:
	################# satellites
	pars = {'tensor_ratio':rr, 'tensor_index':-0.025, 'lensing_residual':1, 'scalar_index':0.9624}
	varnames = {'tensor_ratio':'$r$', 'tensor_index':'$n_T$', 'lensing_residual':'$a_L$', 'scalar_index':'$n_S$'}
	fsky=0.01
	mukarcminT=1
	fwhmdeg=0.1
	fm, basic_spec, der, lcenter, data, error = CMBspectra.fisher_analysis(pars, fsky, mukarcminT, fwhmdeg, deltal=25)
	nbins=len(data)/4
	noT=False
	noE=False
	noTE=False
	allder.append(der)
	allbasic_spec.append(basic_spec)




fm_nsr_lrs = []
fm_nsr_stage4 = []
fm_nsr_lrs_stage4 = []
fm_nsr_hrs = []
fm_nsr_hrsplus = []

fm_nsr_margnt_stage4 = []
fm_nsr_margnt_lrs = []
fm_nsr_margnt_lrs_stage4 = []
fm_nsr_margnt_hrs = []
fm_nsr_margnt_hrsplus = []

for i in np.arange(len(allder)):
	der = allder[i]
	basic_spec = allbasic_spec[i]
	pars['tensor_ratio']=rvalues[i]
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

	fm_nsr_lrs.append(np.linalg.inv(cov_nsr_lrs))
	fm_nsr_stage4.append(np.linalg.inv(cov_nsr_stage4))
	fm_nsr_lrs_stage4.append(np.linalg.inv(cov_nsr_lrs_stage4))
	fm_nsr_hrs.append(np.linalg.inv(cov_nsr_hrs))
	fm_nsr_hrsplus.append(np.linalg.inv(cov_nsr_hrsplus))

	fm_nsr_margnt_stage4.append(np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_stage4), [0,3])))
	fm_nsr_margnt_lrs.append(np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_lrs), [0,3])))
	fm_nsr_margnt_lrs_stage4.append(np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_lrs+fm_stage4), [0,3])))
	fm_nsr_margnt_hrs.append(np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_hrs), [0,3])))
	fm_nsr_margnt_hrsplus.append(np.linalg.inv(FisherNew.submatrix(np.linalg.inv(fm_hrsplus), [0,3])))




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
for i in np.arange(len(rvalues)):
	nsval = pars['scalar_index']
	rval = rvalues[i]
	a1=FisherNew.cont_from_fisher2d(fm_nsr_margnt_stage4[i], [nsval, rval] ,color='b', onesigma=True)
	a2=FisherNew.cont_from_fisher2d(fm_nsr_margnt_lrs[i], [nsval, rval] ,color='g', onesigma=True)
	a4=FisherNew.cont_from_fisher2d(fm_nsr_margnt_lrs_stage4[i], [nsval, rval] ,color='m', onesigma=True)
	a3=FisherNew.cont_from_fisher2d(fm_nsr_margnt_hrs[i], [nsval, rval] ,color='r', onesigma=True)

plot([nsval-0.0054,nsval-0.0054],[0,1],'k:')
plot([nsval+0.0054,nsval+0.0054],[0,1],'k:')
legend([a1, a2, a4, a3],['Stage 4', 'LRS', 'Stage 4 + LRS', 'HRS'],loc='upper right',title='TT+EE+TE+BB \n(marginalising over $n_T, a_L$)', fontsize=15, frameon=0)
savefig('inflationplot.png')

clf()
xlim(0.956,0.97)
ylim(0.,0.3)
xlabel('$n_s$',fontsize=20)
ylabel('$r$',fontsize=20)
al = 0.3
aa = fill_between([0.8, 1], [0,0], [0.45, 0],color='b', alpha=al)
text(0.957,0.07,'Small Field',fontsize=20)
bb = fill_between([0.8, 1], [0.45, 0], [1.4,0],color='r', alpha=al)
text(0.957,0.12,'Large Field',fontsize=20)
cc = fill_between([0.8, 1.1], [1.4,-0.7], [1e8,0],color='y', alpha=al)
text(0.961,0.28,'Hybrid',fontsize=20)
for i in np.arange(len(rvalues)):
	nsval = pars['scalar_index']
	rval = rvalues[i]
	a1=FisherNew.cont_from_fisher2d(fm_nsr_margnt_stage4[i], [nsval, rval] ,color='b', onesigma=True)
	a2=FisherNew.cont_from_fisher2d(fm_nsr_margnt_lrs[i], [nsval, rval] ,color='g', onesigma=True)
	a4=FisherNew.cont_from_fisher2d(fm_nsr_margnt_lrs_stage4[i], [nsval, rval] ,color='m', onesigma=True)
	a3=FisherNew.cont_from_fisher2d(fm_nsr_margnt_hrs[i], [nsval, rval] ,color='r', onesigma=True)

plot([nsval-0.0054,nsval-0.0054],[0,1],'k:')
plot([nsval+0.0054,nsval+0.0054],[0,1],'k:')
legend([a1, a2, a4, a3],['Stage 4', 'LRS', 'Stage 4 + LRS', 'HRS'],loc='upper right',title='TT+EE+TE+BB \n(marginalising over $n_T, a_L$)', fontsize=15, frameon=0)
savefig('inflationplot_zoom.png')




clf()
xlim(0.956,0.97)
ylim(0.,0.3)
xlabel('$n_s$',fontsize=20)
ylabel('$r$',fontsize=20)
al = 0.3
aa = fill_between([0.8, 1], [0,0], [0.45, 0],color='b', alpha=al)
text(0.957,0.07,'Small Field',fontsize=20)
bb = fill_between([0.8, 1], [0.45, 0], [1.4,0],color='r', alpha=al)
text(0.957,0.12,'Large Field',fontsize=20)
cc = fill_between([0.8, 1.1], [1.4,-0.7], [1e8,0],color='y', alpha=al)
text(0.961,0.28,'Hybrid',fontsize=20)
for i in np.arange(len(rvalues)):
	nsval = params['scalar_index']
	rval = rvalues[i]
	a1=FisherNew.cont_from_fisher2d(fm_nsr_stage4[i], [nsval, rval] ,color='b', onesigma=True)
	a2=FisherNew.cont_from_fisher2d(fm_nsr_lrs[i], [nsval, rval] ,color='g', onesigma=True)
	a4=FisherNew.cont_from_fisher2d(fm_nsr_lrs_stage4[i], [nsval, rval] ,color='m', onesigma=True)
	a3=FisherNew.cont_from_fisher2d(fm_nsr_hrs[i], [nsval, rval] ,color='r', onesigma=True)
	#a0=FisherNew.cont_from_fisher2d(fm_nsr_hrsplus[i], [nsval, rval] ,color='k', onesigma=True)

plot([nsval-0.0054,nsval-0.0054],[0,1],'k:')
plot([nsval+0.0054,nsval+0.0054],[0,1],'k:')
legend([a1, a2, a4, a3],['Stage 4', 'LRS', 'Stage 4 + LRS', 'HRS'],loc='upper right',title='TT+EE+TE+BB \n(marginalising over $a_L$, fixing $n_T$)', fontsize=15, frameon=0)
savefig('inflationplot_zoom_fixednt.png')









