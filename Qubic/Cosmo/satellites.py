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
deltal = 5
s_nt, ssvl_nt, der_nt, spectra_nt, fmnt, fmntsvl = Fisher.get_tratio_accuracy(params, 1., 1., 0.5, 1., der=None, spectra = None, consistency = False, deltal=deltal)






#### Assume a CMB stage 4 experiment
fsky = 0.5*0.8
mukarcmin = 1.
lens_res = 0.1
fwhmdeg = 3./60
stage4 = [fsky, mukarcmin, lens_res, fwhmdeg]
s_s4, s_s4_svl, dum, dum, fmnt_s4, fmntsvl_s4 = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, der=der_nt, spectra=spectra_nt, consistency=False, plotmat=True, fixed=1, title='CMBS4')



#### Assume a full sky low resolution satellite
fsky = 1.*0.8
mukarcmin = 1
lens_res = 1
fwhmdeg = 20./60
lrs = [fsky, mukarcmin, lens_res, fwhmdeg]
s_lo, s_lo_svl, dum, dum, fmnt_lo, fmntsvl_lo = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, der=der_nt, spectra=spectra_nt, consistency=False, plotmat=True, fixed=1, title='LRS')


#### Assume CMBS4 + LRS
fsky = 0.5*0.8
mukarcmin = 1.
lens_res = 0.1
fwhmdeg = 3./60
s4lrs = [fsky, mukarcmin, lens_res, fwhmdeg]
s_los4, s_los4_svl, dum, dum, fmnt_los44, fmntsvl_los4 = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, der=der_nt, spectra=spectra_nt, consistency=False, plotmat=True, fixed=1, prior = fmnt_lo, title='CMBS4 + LRS')



#### Assume a high resolution satellite

## CORE ##
#fsky = 1.*0.8
#mukarcmin = 2.1
#lens_res = 0.1
#fwhmdeg = 10.

## CORE+ / PRISM
fsky = 1.*0.8
mukarcmin = 1.5
lens_res = 0.1
fwhmdeg = 7./60
sat = [fsky, mukarcmin, lens_res, fwhmdeg]
s_sat, s_sat_svl, dum, dum, fmnt_sat, fmntsvl_sat = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, der=der_nt, spectra=spectra_nt, consistency=False, plotmat=True, fixed=1, title='HRS')

## CORE++ ?
fsky = 1.*0.8
mukarcmin = 0.5
lens_res = 0.1
fwhmdeg = 5./60
satplus = [fsky, mukarcmin, lens_res, fwhmdeg]
s_satplus, s_satplus_svl, dum, dum, fmnt_satplus, fmntsvl_satplus = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, der=der_nt, spectra=spectra_nt, consistency=False, plotmat=True, fixed=1, title='HRS++')





vals = [rvalue, 1., -rvalue/8, 1.]
parnames = ['$r$','$n_s$', '$n_t$','$a_l$']
lim = [[vals[0]-0.03,vals[0]+0.03],[vals[2]-0.1,vals[2]+0.1],[0.997,1.003]]
clf()
a1,l1 = Fisher.plot_fisher(fmnt_lo, vals, parnames,'g',limits=lim, fixed=1, onesigma=True)
a0,l0 = Fisher.plot_fisher(fmnt_s4, vals, parnames,'b',limits=lim, fixed=1, onesigma=True)
a3,l3 = Fisher.plot_fisher(fmnt_s4+fmnt_lo, vals, parnames,'r',limits=lim, fixed=1, onesigma=True)
a4,l4 = Fisher.plot_fisher(fmnt_sat, vals, parnames,'k',limits=lim, fixed=1, onesigma=True)
a5,l5 = Fisher.plot_fisher(fmnt_satplus, vals, parnames,'m',limits=lim, fixed=1, onesigma=True)
subplot(3,3,3)
axis('off')
legend([a1,a0,a3,a4,a5],['Low Res Sat : '+l1, 'CMB S4 : '+l0, 'Low Res Sat + CMB S4 : '+l3, 'High Res Sat : '+l4, 'High Res Sat ++: '+l5], fontsize=10,title='$r = ${0:.2g}'.format(rvalue))

savefig('compare_satellites_r{0:.2g}'.format(rvalue)+'.png')


######################### Various plots #####################
#### error on ns from Planck TT alone 0.9624 +/- 0.0094
errns = 0.0094
fmnt_los4 = fmnt_lo + fmnt_s4
fmnt_los4[1,1] = 1./errns**2
fmnt_lo[1,1] = 1./errns**2
fmnt_s4[1,1] = 1./errns**2
fmnt_sat[1,1] = 1./errns**2
fmnt_satplus[1,1] = 1./errns**2

#### get the correct matrix for r and ns
## fix nt
sub_fmnt_lo = Fisher.submatrix(fmnt_lo,[0,1,3])
sub_fmnt_s4 = Fisher.submatrix(fmnt_s4,[0,1,3])
sub_fmnt_los4 = Fisher.submatrix(fmnt_los4,[0,1,3])
sub_fmnt_sat = Fisher.submatrix(fmnt_sat,[0,1,3])
sub_fmnt_satplus = Fisher.submatrix(fmnt_satplus,[0,1,3])

## Marginalize over al and reverse order for having ns, r
cov_nsr_lo = Fisher.submatrix(np.linalg.inv(sub_fmnt_lo), [1,0])
cov_nsr_s4 = Fisher.submatrix(np.linalg.inv(sub_fmnt_s4), [1,0])
cov_nsr_los4 = Fisher.submatrix(np.linalg.inv(sub_fmnt_los4), [1,0])
cov_nsr_sat = Fisher.submatrix(np.linalg.inv(sub_fmnt_sat), [1,0])
cov_nsr_satplus = Fisher.submatrix(np.linalg.inv(sub_fmnt_satplus), [1,0])

fm_nsr_lo = np.linalg.inv(cov_nsr_lo)
fm_nsr_s4 = np.linalg.inv(cov_nsr_s4)
fm_nsr_los4 = np.linalg.inv(cov_nsr_los4)
fm_nsr_sat = np.linalg.inv(cov_nsr_sat)
fm_nsr_satplus = np.linalg.inv(cov_nsr_satplus)


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
a0=Fisher.cont_from_fisher2d(fm_nsr_lo, [nsval, rval] ,color='g', onesigma=True)
a1=Fisher.cont_from_fisher2d(fm_nsr_s4, [nsval, rval] ,color='b', onesigma=True)
a2=Fisher.cont_from_fisher2d(fm_nsr_los4, [nsval, rval] ,color='r', onesigma=True)
a3=Fisher.cont_from_fisher2d(fm_nsr_sat, [nsval, rval] ,color='k', onesigma=True)
a4=Fisher.cont_from_fisher2d(fm_nsr_satplus, [nsval, rval] ,color='m', onesigma=True)
legend([a1,a0,a2,a4],['CMBS4', 'LRS', 
	'CMBS4+LRS', 
	'HRS','HRS++'], fontsize=15, frameon=0)
savefig('inflationtype_satellites._r{0:.2g}'.format(rvalue)+'.png')


clf()
xlim(0.8,1.1)
ylim(rvalue-0.01,rvalue+0.01)
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
a0=Fisher.cont_from_fisher2d(fm_nsr_lo, [nsval, rval] ,color='g', onesigma=True)
a1=Fisher.cont_from_fisher2d(fm_nsr_s4, [nsval, rval] ,color='b', onesigma=True)
a2=Fisher.cont_from_fisher2d(fm_nsr_los4, [nsval, rval] ,color='r', onesigma=True)
a3=Fisher.cont_from_fisher2d(fm_nsr_sat, [nsval, rval] ,color='k', onesigma=True)
a4=Fisher.cont_from_fisher2d(fm_nsr_satplus, [nsval, rval] ,color='m', onesigma=True)
legend([a1,a0,a2,a4],['CMBS4', 'LRS', 
	'CMBS4+LRS', 
	'HRS','HRS++'], fontsize=15, frameon=0)
savefig('inflationtype_satellites_zoom_r{0:.2g}'.format(rvalue)+'.png')







############# Explore improvements on each of them
#### [fsky, mukarcmin, lens_res, fwhmdeg]
def explore_sigma(theparams, num, mini, maxi, prior=None):
	theprior=None
	if prior is not None:
		dum, dum, dum, dum, theprior, dum = Fisher.get_tratio_accuracy(params, prior[0], prior[1], prior[3], prior[2], der=der_nt, spectra=spectra_nt, consistency=False, fixed=1)
	nn=100
	parvals = np.zeros((4,100))
	parvals[0,:] += theparams[0]
	parvals[1,:] += theparams[1]
	parvals[2,:] += theparams[2]
	parvals[3,:] += theparams[3]
	x = linspace(mini,maxi,nn)
	parvals[num,:] = x
	s = np.zeros((3,nn))
	s_svl = np.zeros((3,nn))
	for i in np.arange(nn):
		a, b, dum, dum, fmnt, fmntsvl = Fisher.get_tratio_accuracy(params, parvals[0,i], parvals[1,i], parvals[3,i], parvals[2,i], der=der_nt, spectra=spectra_nt, consistency=False, fixed=1, prior=theprior)
		s[:,i]=a
		s_svl[:,i]=b
	return s, s_svl, x



def do_all(vals, num, col, mini, maxi, label):
	s,ssvl,x = explore_sigma(vals, num, mini, maxi)
	subplot(2,1,1)
	ylim(0,100)
	bla, = plot(x,rvalue/s[0],col,lw=2,label=label)
	plot(x,rvalue/ssvl[0],col,lw=2,ls='--')
	plot(vals[num], np.interp(vals[num],x,rvalue/s[0]),'o',color=col,ms=8)
	#plot([vals[num],vals[num]],[0,100],col,ls=':',lw=2)
	subplot(2,1,2)
	ylim(0,3)
	bla, = plot(x,ntvalue/s[1],col,lw=2,label=label)
	plot(x,ntvalue/ssvl[1],col,lw=2,ls='--')
	plot(vals[num], np.interp(vals[num],x,ntvalue/s[1]),'o',color=col,ms=8)
	#plot([vals[num],vals[num]],[0,3],col,ls=':',lw=2)
	return bla


###### Evolution with fsky
clf()
as4 = do_all(stage4, 0, 'b', 0.1, 1, 'CMBS4')
alrs = do_all(lrs, 0, 'g', 0.1, 1, 'LRS')
asat = do_all(sat, 0, 'k', 0.1, 1, 'HRS')
asatplus = do_all(satplus, 0, 'm', 0.1, 1, 'HRS++')
subplot(2,1,1)
grid()
xlabel('fsky')
ylabel('$r/\Delta r$')
legend(loc='upper left',title='$r = ${0:.2g}'.format(rvalue))
subplot(2,1,2)
grid()
xlabel('fsky')
ylabel('$n_T/\Delta n_T$')
legend(loc='upper left',title='$r = ${0:.2g}'.format(rvalue))
savefig('compare_satellites_fsky_r{0:.2g}'.format(rvalue)+'.png')


###### Evolution with mukarcmin
clf()
as4 = do_all(stage4, 1, 'b', 0.1, 5, 'CMBS4')
alrs = do_all(lrs, 1, 'g', 0.1, 5, 'LRS')
asat = do_all(sat, 1, 'k', 0.1, 5, 'HRS')
asatplus = do_all(satplus, 1, 'm', 0.1, 5, 'HRS++')
subplot(2,1,1)
grid()
xlabel('mukarcmin')
ylabel('$r/\Delta r$')
legend(loc='upper right',title='$r = ${0:.2g}'.format(rvalue))
subplot(2,1,2)
grid()
xlabel('mukarcmin')
ylabel('$n_T/\Delta n_T$')
legend(loc='upper right',title='$r = ${0:.2g}'.format(rvalue))
savefig('compare_satellites_mukarcmin_r{0:.2g}'.format(rvalue)+'.png')



###### Evolution with lens_res
clf()
as4 = do_all(stage4, 2, 'b', 0.01, 1, 'CMBS4')
alrs = do_all(lrs, 2, 'g', 0.01, 1, 'LRS')
asat = do_all(sat, 2, 'k', 0.01, 1, 'HRS')
asatplus = do_all(satplus, 2, 'm', 0.01, 1, 'HRS++')
subplot(2,1,1)
grid()
xlabel('Lensing Residuals')
ylabel('$r/\Delta r$')
legend(loc='upper right',title='$r = ${0:.2g}'.format(rvalue))
subplot(2,1,2)
grid()
xlabel('Lensing Residuals')
ylabel('$n_T/\Delta n_T$')
legend(loc='upper right',title='$r = ${0:.2g}'.format(rvalue))
savefig('compare_satellites_lensingres_r{0:.2g}'.format(rvalue)+'.png')



###### Evolution with lens_res
clf()
as4 = do_all(stage4, 3, 'b', 1/60, 60/60, 'CMBS4')
alrs = do_all(lrs, 3, 'g', 1/60, 60/60, 'LRS')
asat = do_all(sat, 3, 'k', 1/60, 60/60, 'HRS')
asatplus = do_all(satplus, 3, 'm', 1/60, 60/60, 'HRS++')
subplot(2,1,1)
grid()
xlabel('FWHM (deg)')
ylabel('$r/\Delta r$')
legend(loc='upper right',title='$r = ${0:.2g}'.format(rvalue))
subplot(2,1,2)
grid()
xlabel('FWHM (deg)')
ylabel('$n_T/\Delta n_T$')
legend(loc='upper right',title='$r = ${0:.2g}'.format(rvalue))
savefig('compare_satellites_fwhm_r{0:.2g}'.format(rvalue)+'.png')



####################################################### New Code ###########################
from Cosmo import FisherNew as Fisher



