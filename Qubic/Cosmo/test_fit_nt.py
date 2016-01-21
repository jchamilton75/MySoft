
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
         'tensor_index':-rvalue/8}
params_nl = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':rvalue,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True,
         'tensor_index':-rvalue/8}

lmax=1000
ctt,cee,cbbprim,cte = pycamb.camb(lmax,**params)
ctt,cee,cbbtot,cte = pycamb.camb(lmax,**params_nl)
cbblensing = cbbtot-cbbprim
ell = np.arange(lmax-1)+2


clf()
xlim(0,500)
a0,=plot(ell, cbbprim, 'b--', label='Primordial B-modes (r={0:.1g})'.format(rvalue),lw=3)
a1,=plot(ell, cbblensing, 'g--', label='Lensing B-modes',lw=3)
a2,=plot(ell, cbbtot, 'r', label='Total B-modes',lw=3)
a3,=plot(ell, cee, color='orange', label='E-modes',lw=3)
a4,=plot(ell, -cte, ':', color='purple',label='Total',lw=3)
a4,=plot(ell, cte, color='purple',label='TE correlation',lw=3)
a5,=plot(ell, ctt, 'k', label='Temperature',lw=3)
yscale('log')
ylim(7e-4,1e4)
xlabel('$\ell$',fontsize=20)
ylabel('$\ell(\ell +1)C_\ell /2\pi$  $[\mu K^2]$',fontsize=20)
legend([a5,a3,a4,a0,a1,a2],['Temperature', 'E-modes', 'TE correlation', 'Primordial B-modes (r={0:.1g})'.format(rvalue),
	'Lensing B-modes', 'Total B-modes'], loc='upper right',shadow=True,framealpha=0.1)
savefig('cmbspectra.png')


fsky = 0.01
deltal=35
svfact = sqrt(2./((2*ell+1)*fsky*deltal))
al=0.3
clf()
xlim(0,500)
fill_between(ell,(1+2*svfact)*cbbprim, ((1-2*svfact)*cbbprim).clip(min=1e-6), color='b', alpha=al/2)
fill_between(ell,(1+svfact)*cbbprim, ((1-svfact)*cbbprim).clip(min=1e-6), color='b', alpha=al)
fill_between(ell,(1+2*svfact)*cbblensing, ((1-2*svfact)*cbblensing).clip(min=1e-6), color='g', alpha=al/2)
fill_between(ell,(1+svfact)*cbblensing, ((1-svfact)*cbblensing).clip(min=1e-6), color='g', alpha=al)
fill_between(ell,(1+2*svfact)*cbbtot, ((1-2*svfact)*cbbtot).clip(min=1e-6), color='r', alpha=al/2)
fill_between(ell,(1+svfact)*cbbtot, ((1-svfact)*cbbtot).clip(min=1e-6), color='r', alpha=al)
a0,=plot(ell, cbbprim, 'b--', label='Primordial r=0.2',lw=3)
a1,=plot(ell, cbblensing, 'g--', label='Lensing',lw=3)
a2,=plot(ell, cbbtot, 'r', label='Total',lw=3)
yscale('log')
ylim(7e-4,0.1)
legend([a0,a1,a2],['Primordial r={0:.1g}'.format(rvalue), 'Lensing', 'Total'], loc='upper left')
xlabel('\ell')
ylabel('$\ell(\ell +1)C_\ell /2\pi$  $[\mu K^2]$')
title('B-modes with sample variance ($\Delta\ell=${0:.0f}, $f_s=${1:.2f})'.format(deltal,fsky))
savefig('bmodes.png')


##### Fisher Analysis
## First run to get the derivatives and spectra
s, ssvl, der, spectra, fmrt, fmrtsvl = Fisher.get_tratio_accuracy(params, 1., 1., 0.5, 1., der=None, spectra =None, consistency=True)

s_nt, ssvl_nt, der_nt, spectra_nt, fmnt, fmntsvl = Fisher.get_tratio_accuracy(params, 1., 1., 0.5, 1., der=None, spectra = None, consistency = False)


# BICEP prediction should find s[0] ~ 0.05
fsky = 384/41000
mukarcmin = 87/1000*60
lens_res = 1
fwhmdeg = 0.52
s, s_svl, dum, dum, fmnt_bicep, fmntsvl_bicep = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, consistency=False, plotmat=True,fixed=1)


# QUBIC
fsky = 0.01
mukarcmin = 4.
lens_res = 1
fwhmdeg = 0.52
squbic, s_svl, dum, dum, fmnt_qubic, fmntsvl_qubic = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, consistency=False, plotmat=True, fixed=1)
ylim(0,0.04)
xlim(0,300)
savefig('qubicnow.png')

##### COmpare BICEP2 and QUBIC

vals = [rvalue, -0.025, 1.]
parnames = ['$r$','$n_t$','$a_l$']
lim = [[rvalue-0.1,rvalue+0.1],[0.,2]]
clf()
a1,l1 = Fisher.plot_fisher(fmnt_bicep, vals, parnames,'b',limits=lim,fixed=1)
a0,l0 = Fisher.plot_fisher(fmnt_qubic, vals, parnames,'r',limits=lim,fixed=1)
subplot(3,3,3)
axis('off')
legend([a1,a0],['BICEP2 : '+l1, 'QUBIC 1 year : '+l0])
savefig('compare_bicep2-qubic.png')


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
			der=der_nt, spectra=spectra_nt, consistency=False, fixed=1)
	sn, sn_svl, dum, dum, blo, bla = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
			der=der_nt, spectra=spectra_nt, consistency=False, fixed=1,noiseonly=True)
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
savefig('optimization.png')






# QUBIC reoptimized
fsky_reopt = 0.04
mukarcmin = 4.*(fsky_reopt/0.01)**0.5
lens_res = 1
fwhmdeg = 0.52
squbicnew, s_svl, dum, dum, fmnt_qubicnew, fmntsvl_qubicnew = Fisher.get_tratio_accuracy(params, fsky_reopt, mukarcmin, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, consistency=False, plotmat=True, fixed=1)
ylim(0,0.04)
xlim(0,300)
savefig('qubicreopt.png')



# QUBIC reoptimized
fsky_reopt = 0.04
mukarcmin = 4.*(fsky_reopt/0.01)**0.5/sqrt(6)
lens_res = 1
fwhmdeg = 0.52
squbicnew6, s_svl, dum, dum, fmnt_qubicnew6, fmntsvl_qubicnew6 = Fisher.get_tratio_accuracy(params, fsky_reopt, mukarcmin, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, consistency=False, plotmat=True, fixed=1)
ylim(0,0.04)
xlim(0,300)
savefig('qubicreopt_6modules.png')




# QUBIC reoptimized with nT
#fsky_reopt = 0.04
#mukarcmin = 4.*(fsky/0.01)**0.5/sqrt(6)
#lens_res = 1
#fwhmdeg = 0.52
#squbicnew6, s_svl, dum, dum, fmnt_qubicnew6, fmntsvl_qubicnew6 = Fisher.get_tratio_accuracy(params, fsky_reopt, mukarcmin, fwhmdeg, lens_res, 
#		der=der_nt, spectra=spectra_nt, consistency=False, plotmat=True)
#ylim(0,0.04)
#xlim(0,300)
#savefig('qubicreopt_6modules_withnt.png')


##### COmpare BICEP2 and QUBIC

vals = [rvalue, -0.025, 1.]
parnames = ['$r$','$n_t$','$a_l$']
lim = [[rvalue-0.1,rvalue+0.1],[0.,2]]
clf()
a1,l1 = Fisher.plot_fisher(fmnt_bicep, vals, parnames,'k',limits=lim,fixed=1,onesigma=True)
a0,l0 = Fisher.plot_fisher(fmnt_qubic, vals, parnames,'b',limits=lim,fixed=1,onesigma=True)
a2,l2 = Fisher.plot_fisher(fmnt_qubicnew, vals, parnames,'g',limits=lim,fixed=1,onesigma=True)
a3,l3 = Fisher.plot_fisher(fmnt_qubicnew6, vals, parnames,'r',limits=lim,fixed=1,onesigma=True)
subplot(3,3,3)
axis('off')
legend([a1,a0,a2,a3],['BICEP2 : '+l1, 'QUBIC 1 year, fsky=0.01 : '+l0, 'QUBIC 1 year, fsky={0:.2g} : '.format(fsky_reopt)+l2, 'QUBIC 6 modules, fsky={0:.2g} : '.format(fsky_reopt)+l3])
savefig('compare_bicep2-qubic_reoptimized.png')


vals = [0.2, -0.025, 1.]
parnames = ['$r$','$n_t$','$a_l$']
lim = [[-0.5,1],[-2,2],[0.,2]]
clf()
a1,l1 = Fisher.plot_fisher(fmnt_bicep, vals, parnames,'k',limits=lim,onesigma=True)
a0,l0 = Fisher.plot_fisher(fmnt_qubic, vals, parnames,'b',limits=lim,onesigma=True)
a2,l2 = Fisher.plot_fisher(fmnt_qubicnew, vals, parnames,'g',limits=lim,onesigma=True)
a3,l3 = Fisher.plot_fisher(fmnt_qubicnew6, vals, parnames,'r',limits=lim,onesigma=True)
subplot(3,3,3)
axis('off')
legend([a1,a0,a2,a3],['BICEP2 : '+l1, 'QUBIC 1 year, fsky=0.01 : '+l0, 'QUBIC 1 year, fsky={0:.2g} : '.format(fsky_reopt)+l2, 'QUBIC 6 modules, fsky={0:.2g} : '.format(fsky_reopt)+l3])







#### Assume a CMB stage 4 experiment
fsky = 0.1
mukarcmin = 1.
lens_res = 0.1
fwhmdeg = 3./60
s, s_svl, dum, dum, fmnt_s4, fmntsvl_s4 = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, consistency=False, plotmat=True)
print(s)
print(s_svl)

s, s_svl, dum, dum, fm_s4, fmsvl_s4 = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
		der=der, spectra=spectra, consistency=True, plotmat=True)



#### Assume a full sky low resolution satellite
fsky = 0.6
mukarcmin = 1.
lens_res = 1
fwhmdeg = 20./60
s, s_svl, dum, dum, fmnt_sat, fmntsvl_sat = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, consistency=False, plotmat=True)
print(s)
print(s_svl)

s, s_svl, dum, dum, fm_sat, fmsvl_sat = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
		der=der, spectra=spectra, consistency=True, plotmat=True)


#### Assume a high resolution satellite
fsky = 0.6
mukarcmin = 0.5
lens_res = 0.1
fwhmdeg = 2./60
s, s_svl, dum, dum, fmnt_prism, fmntsvl_prism = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
		der=der_nt, spectra=spectra_nt, consistency=False, plotmat=True)
print(s)
print(s_svl)

s, s_svl, dum, dum, fm_prism, fmsvl_prism = Fisher.get_tratio_accuracy(params, fsky, mukarcmin, fwhmdeg, lens_res, 
		der=der, spectra=spectra, consistency=True, plotmat=True)






vals = [0.2, -0.025, 1.]
parnames = ['$r$','$n_t$','$a_l$']
lim = [[0.17,0.23],[-0.1,0.05],[0.997,1.003]]
clf()
a1,l1 = Fisher.plot_fisher(fmnt_sat, vals, parnames,'g',limits=lim)
a0,l0 = Fisher.plot_fisher(fmnt_s4, vals, parnames,'b',limits=lim)
a3,l3 = Fisher.plot_fisher(fmnt_s4+fmnt_sat, vals, parnames,'r',limits=lim)
a4,l4 = Fisher.plot_fisher(fmnt_prism, vals, parnames,'k',limits=lim)
subplot(3,3,3)
axis('off')
legend([a1,a0,a3,a4],['Low Res Sat : '+l1, 'CMB S4 : '+l0, 'Low Res Sat + CMB S4 : '+l3, 'High Res Sat : '+l4])

vals = [0.2, -8, 1.]
parnames = ['$r$','$r/n_t$','$a_l$']
lim = [[0.17,0.23],[-30,12],[0.99,1.01]]
clf()
a1,l1 = Fisher.plot_fisher(fm_sat, vals, parnames,'g',limits=lim)
a0,l0 = Fisher.plot_fisher(fm_s4, vals, parnames,'b',limits=lim)
a3,l3 = Fisher.plot_fisher(fm_s4+fm_sat, vals, parnames,'r',limits=lim)
a4,l4 = Fisher.plot_fisher(fm_prism, vals, parnames,'k',limits=lim)
subplot(3,3,3)
axis('off')
legend([a1,a0,a3,a4],['Low Res Sat : '+l1, 'CMB S4 : '+l0, 'Low Res Sat + CMB S4 : '+l3, 'High Res Sat : '+l4])




#############################################
nb = 30
nb2 = 30
mukarcmin = 0.5
fwhmdeg = linspace(1.,30,nb)/60
fsky = np.linspace(0.01,0.99,nb2)+0.01
lens_res = 0.05
sigs_r = np.zeros((nb,nb2))
sigs_r_svl = np.zeros((nb,nb2))
sigs_nt = np.zeros((nb,nb2))
sigs_nt_svl = np.zeros((nb,nb2))

for k in np.arange(nb2):
	print('Num:',k)	
	for i in np.arange(nb):
		s, s_svl, dum, dum, dum, dum = Fisher.get_tratio_accuracy(params, fsky[k], mukarcmin, fwhmdeg[i], lens_res, 
		der=der_nt, spectra=spectra_nt, consistency=False, fixed=2)
		sigs_r[i,k] = s[0]
		sigs_r_svl[i,k] = s_svl[0]
		sigs_nt[i,k] = s[1]
		sigs_nt_svl[i,k] = s_svl[1]

asp=1./30
clf()
subplot(2,2,1)
imshow(0.2/sigs_r, interpolation='nearest', origin='lower', extent=[fsky[0], fsky[nb2-1], fwhmdeg[0]*60, fwhmdeg[nb-1]*60], aspect=asp)
colorbar()
contour(fsky,fwhmdeg*60,0.2/sigs_r,levels=[1,2,3,10,20,30])
ylabel('FWHM')
xlabel('fsky')
title('$r/\Delta r$')
subplot(2,2,2)
imshow(0.025/sigs_nt, interpolation='nearest', origin='lower', extent=[fsky[0], fsky[nb2-1], fwhmdeg[0]*60, fwhmdeg[nb-1]*60], aspect=asp)
colorbar()
contour(fsky,fwhmdeg*60,0.2/sigs_nt,levels=[1,2,3,10,20,30])
ylabel('FWHM')
xlabel('fsky')
title('$n_T/\Delta n_T$')
subplot(2,2,3)
imshow(0.2/sigs_r_svl, interpolation='nearest', origin='lower', extent=[fsky[0], fsky[nb2-1], fwhmdeg[0]*60, fwhmdeg[nb-1]*60], aspect=asp)
colorbar()
contour(fsky,fwhmdeg*60,0.2/sigs_r_svl,levels=[1,2,3,10,20,30])
ylabel('FWHM')
xlabel('fsky')
title('$r/\Delta r$ S.V.L.')
subplot(2,2,4)
imshow(0.025/sigs_nt_svl, interpolation='nearest', origin='lower', extent=[fsky[0], fsky[nb2-1], fwhmdeg[0]*60, fwhmdeg[nb-1]*60], aspect=asp)
colorbar()
contour(fsky,fwhmdeg*60,0.2/sigs_nt_svl,levels=[1,2,3,10,20,30])
ylabel('FWHM')
xlabel('fsky')
title('$n_T/\Delta n_T$ S.V.L.')



#### Explorer l'idée d'une mission satellite a bas ell et d'instruments au sol à grand ell






