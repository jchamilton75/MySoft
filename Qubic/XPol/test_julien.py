from __future__ import division
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import pycamb
from XPol import wig3j


def wigner3j_array(l2, l3, m2, m3):
	l1min = np.max( [np.abs(l2 - l3), np.abs(m2 + m3)])
	l1max = l2 + l3
	ndim = int(l1max+1)
	wigarray = np.zeros(ndim,dtype=np.float128)
	thrcof = np.zeros(l1max-l1min+1)
	ier = wig3j.wig3j(l2, l3, m2, m3, l1min, l1max, thrcof)
	wigarray[l1min:] = thrcof
	return wigarray

def Mllmatrix(lmax, well):
	n =  lmax + 1

	### decalre the matrix

	mll = np.zeros((n,n))
	### well
	theell = np.arange(len(well))
	well_TT = well * (2 * theell + 1)

	for l1 in np.arange(lmax+1):
		print(l1,lmax+1)
		for l2 in np.arange(lmax+1):
			wigner0 = wigner3j_array(l1, l2, 0, 0)
			l3 = np.arange(np.min([l1 + l2 + 1, len(well_TT)]))
			if len(wigner0)>len(l3): wigner0 = wigner0[0:len(l3)]
			wig00 = wigner0 * wigner0
			sum_TT = np.sum(well_TT[l3] * wig00)
			mll[l1, l2] = (2 * l2 + 1) / (4 * np.pi) * sum_TT

	return mll


def MllBin(Mll, p, q):
	return np.dot(np.dot(p,Mll),q)

def BinCls(p,cls):
	lmax = p.shape[1]-1
	out = np.dot(p,cls[0:lmax+1])
	return out

def make_pq(ell, ell_low_in, ell_hi_in):
	ell_low = np.array(ell_low_in)
	ell_hi = np.array(ell_hi_in)
	nb = len(ell_low)
	nl = len(ell)
	pp = np.zeros((nb, nl))
	qq = np.zeros((nl, nb))
	ell_new = (ell_hi + ell_low) / 2

	for b in np.arange(nb):
		if b != nb - 1: 
			ellmaxbin = ell_low[b + 1]
		else:
			ellmaxbin = ell_hi[b] - 1
		for l in np.arange(nl):
			if (ell_low[b] >= 2) & (ell[l] >= ell_low[b]) & (ell[l] < ellmaxbin):
				pp[b,l] = ell[l] * (ell[l] + 1) / (ellmaxbin - ell_low[b]) / 2 / np.pi
				qq[l,b] = 2. * np.pi / (ell[l] * (ell[l] +1))

	return ell_new, pp, qq 



#### Get input Power spectra
################# Input Power spectrum ###################################
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
import pycamb
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
         'tensor_ratio':0.,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False}

lmaxcamb = 3*512
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
lll = np.arange(1,lmaxcamb+1)
fact = (lll*(lll+1))/(2*np.pi)
spectra = [lll, T/fact, E/fact, B/fact, X/fact]
clf()
plot(lll,spectra[1]*(lll*(lll+1))/(2*np.pi),label='$C_\ell^{TT}$')
#yscale('log')
xlim(0,lmaxcamb+1)
#ylim(0.0001,100)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$'+'    '+'$[\mu K^2]$ ')
legend(loc='lower right',frameon=False)








#############################################################################
nside = 256
lmax = 3*nside-1

#### Mask
maxang = 5.
center = [0.,0.]
veccenter = hp.ang2vec(pi/2-np.radians(center[1]), np.radians(center[0]))
vecpix = hp.pix2vec(nside,np.arange(12*nside**2))
cosang = np.dot(veccenter,vecpix)
maskok = np.degrees(np.arccos(cosang)) < maxang
maskmap=np.zeros(12*nside**2)
maskmap[maskok]=1

wl = hp.anafast(maskmap,regression=False)
wl = wl[0:lmax+1]


#### Do a Monte-Carlo
nbmc = 1000
allcls = np.zeros((nbmc, lmax+1))
for i in np.arange(nbmc):
	print(i)
	map = hp.synfast(spectra[1],nside,fwhm=0,pixwin=True,new=True)
	allcls[i,:]  = hp.anafast(map*maskmap)


#### Mll Matrix
Mll = Mllmatrix(lmax, wl)

#### ell binning
min_ell = 20
deltal = 30
all_ell = min_ell-1 + np.arange(1000)*deltal
max_ell = np.max(all_ell[all_ell < lmax])
nbins = (max_ell + 1 - min_ell) / deltal
ell_low = min_ell + np.arange(nbins)*deltal
ell_hi = ell_low + deltal - 1
the_ell = np.arange(lmax+1)
newl, p, q = make_pq(the_ell, ell_low, ell_hi)



#### Cls bins
MllBinned = MllBin(Mll, p, q)

#### Invert MllBinned
MllBinnedInv = np.linalg.inv(MllBinned)

##### Apply to MC
allclsout = np.zeros((nbmc, nbins))
for i in np.arange(nbmc):
	print(i)
	ClsBinned = BinCls(p,allcls[i,:])
	allclsout[i,:] = np.dot(MllBinnedInv, np.ravel(np.array(ClsBinned)))

#### Get MC results
mclsout = np.zeros(nbins)
sclsout = np.zeros(nbins)
for j in np.arange(nbins):
	mclsout[j] = np.mean(allclsout[:,j])
	sclsout[j] = np.std(allclsout[:,j])/sqrt(nbmc)

mcls = np.zeros(lmax+1)
scls = np.zeros(lmax+1)
for j in np.arange(lmax+1):
	mcls[j] = np.mean(allcls[:,j])
	scls[j] = np.std(allcls[:,j])/sqrt(nbmc)


ell = np.arange(lmax+1)
pw = hp.pixwin(nside)[0:lmax+1]
pwb = np.interp(newl,ell,pw)
fact = ell * (ell + 1) / (2 * np.pi) * len(maskmap) / maskok.sum()
factth = lll * (lll + 1) / (2 * np.pi)

clf()
subplot(2,1,1)
title('TT')
xlim(0,lmax)
errorbar(newl,mclsout/pwb**2,yerr=sclsout,fmt='bo',label='Corrected')
plot(ell,fact*mcls/pw**2,'g',label='Anafast rescaled')
plot(spectra[0],spectra[1]*factth,'r',label='Input')
legend(loc='lower right',frameon=False,fontsize=10)

subplot(2,1,2)
title('Residuals')
xlim(0,lmax)
plot(ell, fact*mcls/pw**2 - spectra[1][0:lmax+1]*factth[0:lmax+1],'g',label='Anafast rescaled')
errorbar(newl,mclsout/pwb**2 - np.dot(p,spectra[1][0:lmax+1]),yerr=sclsout,fmt='bo',label='Corrected')
plot(ell,ell*0,'k--')



