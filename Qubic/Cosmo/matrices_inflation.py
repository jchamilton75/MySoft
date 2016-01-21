from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import pycamb
#from qubic.utils import progress_bar
from Homogeneity import fitting
from Cosmo import Fisher

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
rvalue = 0.2
print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',omegav/h2
params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':rvalue,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False,
         'tensor_index':-rvalue/8, 'lensing_amplitude':1.}


##### Fisher Analysis
## First run to get the derivatives and spectra
deltal = 25
s_nt, ssvl_nt, der_nt, spectra_nt, fmnt, fmntsvl = Fisher.get_tratio_accuracy(params, 1., 1., 0.5, 1., 
	der=None, spectra = None, consistency = False, deltal=deltal)


# QUBIC
fsky = 0.01
mukarcmin = 4.
lens_res = 1
fwhmdeg = 0.52
squbic, s_svl, dum, dum, fmnt_qubic, fmntsvl_qubic = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, deltal=deltal, consistency=False, plotmat=True, fixed=1)




# BICEP prediction should find s[0] ~ 0.05
fsky = 384/41000
mukarcmin = 87/1000*60
lens_res = 1
fwhmdeg = 0.52
s, s_svl, dum, dum, fmnt_bicep, fmntsvl_bicep = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, deltal=deltal, consistency=False, plotcl=True,fixed=[1,2])


# QUBIC
fsky = 0.01
mukarcmin = 4.
lens_res = 1
fwhmdeg = 0.52
squbic, s_svl, dum, dum, fmnt_qubic, fmntsvl_qubic = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, deltal=deltal, consistency=False, plotcl=True, fixed=[1,2])
ylim(0,0.04)
xlim(0,300)
savefig('qubicnow.png')

##### COmpare BICEP2 and QUBIC

vals = [rvalue, params['scalar_index'],-0.025, 1.]
parnames = ['$r$','$n_s$', '$n_t$','$a_l$']
lim = [[rvalue-0.1,rvalue+0.1],[0.,2]]
clf()
a1,l1 = Fisher.plot_fisher(fmnt_bicep, vals, parnames,'b',limits=lim,fixed=[1,2])
a0,l0 = Fisher.plot_fisher(fmnt_qubic, vals, parnames,'r',limits=lim,fixed=[1,2])
subplot(3,3,3)
axis('off')
legend([a1,a0],['BICEP2 : '+l1, 'QUBIC 1 year : '+l0])


###### Optimizing limit on r with QUBIC by changing fsky ?
fskyvals = linspace(0.005, 0.1, 101)
s_r = np.zeros(len(fskyvals))
svl_r = np.zeros(len(fskyvals))
sn_r = np.zeros(len(fskyvals))
for i in np.arange(len(fskyvals)):
	fsky = fskyvals[i]
	mukarcmin = 4.*(fsky/0.01)**0.5
	print(mukarcmin)
	lens_res = 1
	fwhmdeg = 0.52
	s, s_svl, dum, dum, blo, bla = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
			der=der_nt, spectra=spectra_nt, deltal=deltal, consistency=False, fixed=[1,2])
	sn, sn_svl, dum, dum, blo, bla = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
			der=der_nt, spectra=spectra_nt, deltal=deltal, consistency=False, fixed=[1,2],noiseonly=True)
	s_r[i] = s[0]
	svl_r[i] = s_svl[0]
	sn_r[i] = sn[0]

mm = np.min(s_r)
ff = fskyvals[s_r == mm]

clf()
xlim(0,np.max(fskyvals))
ylim(0,0.05)
asv,=plot(fskyvals,svl_r,'b',lw=3,label='Sample Variance')
anv,=plot(fskyvals,sn_r,'g',lw=3,label='Noise Variance')
atv,=plot(fskyvals,s_r,'r',lw=3,label='Total')
ylabel('$\sigma_r$',fontsize=20)
xlabel('$f_{sky}$',fontsize=20)
ac,=plot([0.01,0.01],[0,10],'k:',lw=2)
plot([0.0, np.max(fskyvals)],[squbic[0],squbic[0]],'k:',lw=2)
anew,=plot([ff,ff],[0,10],'k--',lw=2)
plot([0.0, np.max(fskyvals)],[mm,mm],'k--',lw=2)
legend([asv,anv,atv,ac,anew],['Sample Variance','Noise Variance', 'Total','Current Strategy (upper-limit)', 'Optimal Strategy for r=0.2'])
title('Error on tensor to scalar ratio - QUBIC')



# QUBIC reoptimized
fsky_reopt = 0.04
mukarcmin = 4.*(fsky_reopt/0.01)**0.5
lens_res = 1
fwhmdeg = 0.52
squbicnew, s_svl, dum, dum, fmnt_qubicnew, fmntsvl_qubicnew = Fisher.get_tratio_accuracy(params, fsky_reopt, mukarcmin, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, deltal=deltal, consistency=False, plotmat=True, fixed=[1,2])
ylim(0,0.04)
xlim(0,300)



# QUBIC reoptimized
fsky_reopt = 0.04
mukarcmin = 4.*(fsky_reopt/0.01)**0.5/sqrt(6)
lens_res = 1
fwhmdeg = 0.52
squbicnew6, s_svl, dum, dum, fmnt_qubicnew6, fmntsvl_qubicnew6 = Fisher.get_tratio_accuracy(params, fsky_reopt, mukarcmin, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, deltal=deltal, consistency=False, plotmat=True, fixed=[1,2])
ylim(0,0.04)
xlim(0,300)



##### COmpare BICEP2 and QUBIC

vals = [rvalue, params['scalar_index'], -0.025, 1.]
parnames = ['$r$','$n_s$', '$n_t$','$a_l$']
lim = [[rvalue-0.1,rvalue+0.1],[0.,2]]
clf()
a1,l1 = Fisher.plot_fisher(fmnt_bicep, vals, parnames,'k',limits=lim,fixed=[1,2],onesigma=True)
a0,l0 = Fisher.plot_fisher(fmnt_qubic, vals, parnames,'b',limits=lim,fixed=[1,2],onesigma=True)
a2,l2 = Fisher.plot_fisher(fmnt_qubicnew, vals, parnames,'g',limits=lim,fixed=[1,2],onesigma=True)
a3,l3 = Fisher.plot_fisher(fmnt_qubicnew6, vals, parnames,'r',limits=lim,fixed=[1,2],onesigma=True)
subplot(3,3,3)
axis('off')
legend([a1,a0,a2,a3],['BICEP2 : '+l1, 'QUBIC 1 year, fsky=0.01 : '+l0, 
	'QUBIC 1 year, fsky={0:.2g} : '.format(fsky_reopt)+l2, 
	'QUBIC 6 modules, fsky={0:.2g} : '.format(fsky_reopt)+l3], fontsize=10)
savefig('newqubic.png')

vals = [rvalue, params['scalar_index'], -0.025, 1.]
parnames = ['$r$','$n_s$', '$n_t$','$a_l$']
lim = [[-0.5,1],[-2,2],[0.,2]]
clf()
a1,l1 = Fisher.plot_fisher(fmnt_bicep, vals, parnames,'k',limits=lim,onesigma=True, fixed=1)
a0,l0 = Fisher.plot_fisher(fmnt_qubic, vals, parnames,'b',limits=lim,onesigma=True, fixed=1)
a2,l2 = Fisher.plot_fisher(fmnt_qubicnew, vals, parnames,'g',limits=lim,onesigma=True, fixed=1)
a3,l3 = Fisher.plot_fisher(fmnt_qubicnew6, vals, parnames,'r',limits=lim,onesigma=True, fixed=1)
subplot(3,3,3)
axis('off')
legend([a1,a0,a2,a3],['BICEP2 : '+l1, 'QUBIC 1 year, fsky=0.01 : '+l0, 
	'QUBIC 1 year, fsky={0:.2g} : '.format(fsky_reopt)+l2, 
	'QUBIC 6 modules, fsky={0:.2g} : '.format(fsky_reopt)+l3], fontsize=10)
savefig('newqubic_withnt.png')


################## Now Plot with ns
#### error on ns from Planck TT alone 0.9624 +/- 0.0094
errns = 0.0094
fmnt_bicep[1,1] = 1./errns**2
fmnt_qubic[1,1] = 1./errns**2
fmnt_qubicnew[1,1] = 1./errns**2
fmnt_qubicnew6[1,1] = 1./errns**2


vals = [rvalue, params['scalar_index'], -0.025, 1.]
parnames = ['$r$','$n_s$', '$n_t$','$a_l$']
lim = [[0,0.5],[0.9,1],[0.,2]]
clf()
a1,l1 = Fisher.plot_fisher(fmnt_bicep, vals, parnames,'k',limits=lim,onesigma=True, fixed=2)
a0,l0 = Fisher.plot_fisher(fmnt_qubic, vals, parnames,'b',limits=lim,onesigma=True, fixed=2)
a2,l2 = Fisher.plot_fisher(fmnt_qubicnew, vals, parnames,'g',limits=lim,onesigma=True, fixed=2)
a3,l3 = Fisher.plot_fisher(fmnt_qubicnew6, vals, parnames,'r',limits=lim,onesigma=True, fixed=2)
subplot(3,3,3)
axis('off')
legend([a1,a0,a2,a3],['BICEP2 : '+l1, 'QUBIC 1 year, fsky=0.01 : '+l0, 
	'QUBIC 1 year, fsky={0:.2g} : '.format(fsky_reopt)+l2, 
	'QUBIC 6 modules, fsky={0:.2g} : '.format(fsky_reopt)+l3], fontsize=10)

#### get the correct matrix for r and ns
## fix nt
sub_fmnt_bicep = Fisher.submatrix(fmnt_bicep,[0,1,3])
sub_fmnt_qubic = Fisher.submatrix(fmnt_qubic,[0,1,3])
sub_fmnt_qubicnew = Fisher.submatrix(fmnt_qubicnew,[0,1,3])
sub_fmnt_qubicnew6 = Fisher.submatrix(fmnt_qubicnew6,[0,1,3])

## Marginalize over al and reverse order for having ns, r
cov_nsr_bicep = Fisher.submatrix(np.linalg.inv(sub_fmnt_bicep), [1,0])
cov_nsr_qubic = Fisher.submatrix(np.linalg.inv(sub_fmnt_qubic), [1,0])
cov_nsr_qubicnew = Fisher.submatrix(np.linalg.inv(sub_fmnt_qubicnew), [1,0])
cov_nsr_qubicnew6 = Fisher.submatrix(np.linalg.inv(sub_fmnt_qubicnew6), [1,0])

fm_nsr_bicep = np.linalg.inv(cov_nsr_bicep)
fm_nsr_qubic = np.linalg.inv(cov_nsr_qubic)
fm_nsr_qubicnew = np.linalg.inv(cov_nsr_qubicnew)
fm_nsr_qubicnew6 = np.linalg.inv(cov_nsr_qubicnew6)


####### Inflation : from Dodelson, Kinney, Kolb, astro-ph/9702166v1 
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
a0=Fisher.cont_from_fisher2d(fm_nsr_bicep, [nsval, rval] ,color='k', onesigma=True)
a1=Fisher.cont_from_fisher2d(fm_nsr_qubic, [nsval, rval] ,color='b', onesigma=True)
a2=Fisher.cont_from_fisher2d(fm_nsr_qubicnew, [nsval, rval] ,color='g', onesigma=True)
a3=Fisher.cont_from_fisher2d(fm_nsr_qubicnew6, [nsval, rval] ,color='r', onesigma=True)
legend([a1,a0,a2,a3],['Planck $n_s$ + BICEP2', 'Planck $n_s$ + QUBIC 1 year, $f_{sky}=0.1$', 
	'Planck $n_s$ + QUBIC 1 year, $f_{sky}=0.4$', 
	'Planck $n_s$ + QUBIC 6 modules, $f_{sky}=0.4$'], fontsize=15, frameon=0)
savefig('inflationtype.png')




####### Cross-Check with NewFisher.py
from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import pycamb
#from qubic.utils import progress_bar
from Homogeneity import fitting
from Cosmo import Fisher

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
rvalue = 0.2
print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',omegav/h2
params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':rvalue,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False,
         'tensor_index':-rvalue/8, 'lensing_amplitude':1.}


##### Fisher Analysis
## First run to get the derivatives and spectra
deltal = 25
s_nt, ssvl_nt, der_nt, spectra_nt, fmnt, fmntsvl = Fisher.get_tratio_accuracy(params, 1., 1., 0.5, 1., 
	der=None, spectra = None, consistency = False, deltal=deltal)

##### CMB Stage IV
#### Assume a CMB stage 4 experiment
fsky = 0.5*0.8
mukarcminT = 1.
lens_res = 0.1
fwhmdeg = 3./60
s_stage4, s_svl, dum, dum, fmnt_stage4, fmntsvl_stage4 = Fisher.get_tratio_accuracy(params, fsky, mukarcminT, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, deltal=deltal, consistency=False, plotcl=True, fixed=[1])


### explore fsky
allfsky = linspace(0.01, 1,100)
mukarcminT = 1.
lens_res = 0.1
fwhmdeg = 3./60
pars = [0.2, -0.025, lens_res]
allsvals = np.zeros((3, len(allfsky)))
for i in np.arange(len(allfsky)):
	print(i)
	s_stage4, s_svl, dum, dum, fmnt_stage4, fmntsvl_stage4 = Fisher.get_tratio_accuracy(params, allfsky[i], mukarcminT, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, deltal=deltal, consistency=False, fixed=[1])
	allsvals[:,i] = s_stage4

clf()
for i in np.arange(3):
	subplot(3,1,i+1)
	plot(allfsky, np.abs(pars[i]/allsvals[i,:]))
	xlabel('fsky')




