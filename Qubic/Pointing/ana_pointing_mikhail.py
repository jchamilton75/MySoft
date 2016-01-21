from __future__ import division
from pylab import *
from matplotlib.pyplot import *
import numpy as np
import os
import sys
import glob

from qubic import read_spectra

spectra = read_spectra(0)

ell = linspace(1,len(spectra[0])+1, len(spectra[0]))
fact = ell*(ell+1)/2/pi
tt = spectra[0]* fact
ee = spectra[1]* fact
bb = spectra[2]* fact
te = spectra[3]* fact

plot(ell,tt)

####################################################################
sigmas = np.array([0, 40, 60, 120, 300, 600, 1200])
clbb = []
errclbb = []
for i in xrange(len(sigmas)):
	truc = loadtxt('FilesMikhail_WithTT/clbb_{}.0arcsec.txt'.format(sigmas[i]))
	clbb.append(truc[:,1])
	errclbb.append(truc[:,2])

ellbins = truc[:,0]

clf()
for i in xrange(len(sigmas)):
	errorbar(ellbins, clbb[i], yerr=errclbb[i],label=sigmas[i])
legend()



newtt = np.interp(ellbins, ell, tt)
newee = np.interp(ellbins, ell, ee)
newbb = np.interp(ellbins, ell, bb)
newte = np.interp(ellbins, ell, te)

clf()
errorbar(ellbins, clbb[6], yerr=errclbb[6])
plot(ellbins, newtt/1000000*exp(ellbins/120))
plot(ellbins, newte/100000*exp(ellbins/120))
plot(ellbins, newee/10000*exp(ellbins/120))

from Homogeneity import fitting


### first model
def toto(x, pars):
	return newtt*pars[0]*np.exp((x/pars[1]))
guess = np.array([1./100000, 100])

def toto(x, pars):
	return pars[0]*np.exp((x/pars[1]))
guess = np.array([5.5e-4, 97])

def toto(x, pars):
	return newtt*pars[0]*np.exp((x/pars[1])) + 5.5e-4*exp(x/pars[1])
guess = np.array([1./10000000, 100])

def toto(x, pars):
	return (newtt*pars[0]+pars[2])*np.exp((x/pars[1]))
guess = np.array([1./10000000, 100, 5.5e-4])



clf()
ii=0
errorbar(ellbins, clbb[ii], yerr=errclbb[ii])
thesig = sigmas[ii]
plot(ellbins, toto(ellbins, guess))


bla = fitting.dothefit(ellbins, clbb[ii], errclbb[ii], guess, functname=toto)

clf()
errorbar(ellbins, clbb[ii], yerr=errclbb[ii])
plot(ellbins, toto(ellbins, bla[1]))


clf()
ellscale = np.zeros(len(sigmas))
normalisation = np.zeros(len(sigmas))
err_ellscale = np.zeros(len(sigmas))
err_normalisation = np.zeros(len(sigmas))
for i in xrange(len(sigmas)):
	subplot(3,3,i+1)
	thesig=sigmas[i]
	bla = fitting.dothefit(ellbins, clbb[i], errclbb[i], guess, functname=toto)
	errorbar(ellbins, clbb[i], yerr=errclbb[i])
	plot(ellbins, toto(ellbins, bla[1]))
	ellscale[i] = bla[1][1]
	normalisation[i]=bla[1][0]
	err_ellscale[i] = bla[2][1]
	err_normalisation[i]=bla[2][0]

clf()
errorbar(sigmas, ellscale, yerr=err_ellscale)
#yscale('log')
#xscale('log')

clf()
errorbar(sigmas, normalisation, yerr=err_normalisation)
yscale('log')
xscale('log')



### Better model
def toto(x, pars):
	return (newtt*pars[0]+pars[2])*np.exp((x/pars[1]))
guess = np.array([1./10000000, 100, 5.5e-4])

sigmas = np.array([0.1, 40, 60, 120, 300, 600, 1200])
def toto(x, pars):
	return (newtt*thesig*pars[0]+pars[2])*np.exp((x/pars[1]))
guess = np.array([1./10000000, 100, 5.5e-4])



clf()
ii=0
errorbar(ellbins, clbb[ii], yerr=errclbb[ii])
thesig = sigmas[ii]
plot(ellbins, toto(ellbins, guess))


ii=0
thesig = sigmas[ii]
bla = fitting.dothefit(ellbins, clbb[ii], errclbb[ii], guess, functname=toto)

clf()
errorbar(ellbins, clbb[ii], yerr=errclbb[ii])
plot(ellbins, toto(ellbins, bla[1]))


clf()
ellscale = np.zeros(len(sigmas))
normalisation = np.zeros(len(sigmas))
cst = np.zeros(len(sigmas))
err_ellscale = np.zeros(len(sigmas))
err_normalisation = np.zeros(len(sigmas))
err_cst = np.zeros(len(sigmas))
for i in xrange(len(sigmas)):
	subplot(3,3,i+1)
	thesig=sigmas[i]
	bla = fitting.dothefit(ellbins, clbb[i], errclbb[i], guess, functname=toto)
	errorbar(ellbins, clbb[i], yerr=errclbb[i])
	plot(ellbins, toto(ellbins, bla[1]))
	ellscale[i] = bla[1][1]
	normalisation[i]=bla[1][0]
	cst[i]=bla[1][2]
	err_ellscale[i] = bla[2][1]
	err_cst[i]=bla[2][2]

clf()
subplot(2,2,1)
errorbar(sigmas, ellscale, yerr=err_ellscale)
xlabe('sigma')
title('ell scale')
subplot(2,2,2)
errorbar(sigmas, normalisation, yerr=err_normalisation)
ylim(-0.2e-8, 0.4e-8)
xlabe('sigma')
title('sigma term normalisation')
subplot(2,2,3)
errorbar(sigmas, cst, yerr=err_cst)
xlabe('sigma')
title('constant term')



############## Mikhail's new code
sigmas = np.array([0, 40, 60, 120, 300, 600, 1200])

def toto(x, pars):
	return((newtt + pars[2]*x**2)*pars[0]*np.exp(x/pars[1]))
guess = np.array([1./1000000, 100, 0])

figure()
ellscale = np.zeros(len(sigmas))
normalisation = np.zeros(len(sigmas))
cst = np.zeros(len(sigmas))
err_ellscale = np.zeros(len(sigmas))
err_normalisation = np.zeros(len(sigmas))
err_cst = np.zeros(len(sigmas))
chi2_over_ndf = np.zeros(len(sigmas))
for i in xrange(len(sigmas)):
    subplot(3,3,i+1)
    thesig=sigmas[i]
    bla = fitting.dothefit(ellbins, clbb[i], errclbb[i], guess, functname=toto)
    errorbar(ellbins, clbb[i], yerr=errclbb[i])
    plot(ellbins, toto(ellbins, bla[1]))
    ellscale[i] = bla[1][1]
    normalisation[i]=bla[1][0]
    cst[i]=bla[1][2]
    err_ellscale[i] = bla[2][1]
    err_cst[i]=bla[2][2]
    #chi2_over_ndf[i] = bla[4] / bla[5]

figure()
subplot(2,2,1)
errorbar(sigmas, ellscale, yerr=err_ellscale)
xlabel('sigma')
title('ell scale')
subplot(2,2,2)
errorbar(sigmas, normalisation, yerr=err_normalisation)
xlabel('sigma')
title('sigma term normalisation')
subplot(2,2,3)
errorbar(sigmas, cst, yerr=err_cst)
xlabel('sigma')
title('noise term')
subplot(2,2,4)
#plot(sigmas, chi2_over_ndf)
xlabel('sigma')
title('chi2/ndf')

show()



