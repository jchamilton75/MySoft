from __future__ import division
from scipy import floor, sqrt
import scipy.special
from scipy.misc import factorial
import healpy as hp
import numpy as np
import wig3j

#### the wig3j library is obtained from wig3j_f.f through: f2py -c wig3j_f.f -m wig3j
def wigner3j_array(l2, l3, m2, m3):
	l1min = np.max( [np.abs(l2 - l3), np.abs(m2 + m3)])
	l1max = l2 + l3
	ndim = int(l1max+1)
	wigarray = np.zeros(ndim,dtype=np.float128)
	thrcof = np.zeros(l1max-l1min+1)
	ier = wig3j.wig3j(l2, l3, m2, m3, l1min, l1max, thrcof)
	wigarray[l1min:] = thrcof
	return wigarray

def MllPol_matrices(lmax, well):
	n =  lmax + 1

	### decalre the matrices
	mll_TT_TT = np.zeros((n,n))
	mll_EE_EE = np.zeros((n,n))
	mll_EE_BB = np.zeros((n,n))
	mll_TE_TE = np.zeros((n,n))
	mll_EB_EB = np.zeros((n,n))

	### well
	theell = np.arange(len(well))
	thewell = well * (2 * theell + 1)
	well_TT = np.array(thewell,dtype=np.float128)
	well_TP = np.array(thewell,dtype=np.float128)
	well_PP = np.array(thewell,dtype=np.float128)

	for l1 in np.arange(lmax+1):
		print(l1,lmax+1)
		for l2 in np.arange(lmax+1):
			wigner0 = wigner3j_array(l1, l2, 0, 0)
			if l1<2 or l2<2:
				wigner2 = np.array([0])
			else:
				wigner2 = wigner3j_array(l1, l2, -2, 2)
			l3 = np.arange(np.min([l1 + l2 + 1, len(well_TT)]))
			l1l2l3 = np.array(1-((l1 + l2 + l3) % 2),dtype=np.float128)

			if len(wigner0)>len(l3): wigner0 = wigner0[0:len(l3)]
			if len(wigner2)>len(l3): wigner2 = wigner2[0:len(l3)]

			wig00 = wigner0 * wigner0
			wig02 = wigner0 * wigner2
			wig22 = wigner2 * wigner2

			sum_TT = np.sum(well_TT[l3] * wig00)
			sum_TE = np.sum(well_TP[l3] * wig02 * l1l2l3)
			sum_EE_EE = np.sum(well_PP[l3] * wig22 * l1l2l3)
			sum_EE_BB = np.sum(well_PP[l3] * wig22 * (1-l1l2l3))
			sum_EB = sum_EE_EE + sum_EE_BB #np.sum(well_PP[l3] * wig22)

			mll_TT_TT[l1, l2] = (2 * l2 + 1) / (4 * np.pi) * sum_TT
			mll_EE_EE[l1, l2] = (2 * l2 + 1) / (4 * np.pi) * sum_EE_EE
			mll_EE_BB[l1, l2] = (2 * l2 + 1) / (4 * np.pi) * sum_EE_BB
			mll_TE_TE[l1, l2] = (2 * l2 + 1) / (4 * np.pi) * sum_TE
			mll_EB_EB[l1, l2] = (2 * l2 + 1) / (4 * np.pi) * sum_EB

	return [mll_TT_TT, mll_EE_EE, mll_EE_BB, mll_TE_TE, mll_EB_EB]


def Mll(lmax, well):
	mll_TT_TT, mll_EE_EE, mll_EE_BB, mll_TE_TE, mll_EB_EB = MllPol_matrices(lmax, well)
	n = lmax + 1
	theMll = np.zeros((6*n, 6*n))
	theMll[0:n,0:n] = mll_TT_TT
	theMll[n:2*n,n:2*n] = mll_EE_EE
	theMll[2*n:3*n,2*n:3*n] = mll_EE_EE
	theMll[n:2*n,2*n:3*n] = mll_EE_BB
	theMll[2*n:3*n,n:2*n] = mll_EE_BB
	theMll[3*n:4*n,3*n:4*n] = mll_TE_TE
	theMll[4*n:5*n,4*n:5*n] = mll_TE_TE
	theMll[5*n:6*n,5*n:6*n] = mll_EB_EB
	return(theMll)

def MllBin(Mll, p, q):
	n = int(Mll.shape[0]/6)
	nb = p.shape[0]
	theMll = np.zeros((nb*6, nb*6))
	theMll[0:nb,0:nb] = np.dot(np.dot(p,Mll[0:n,0:n]),q)
	theMll[nb:2*nb,nb:2*nb] = np.dot(np.dot(p,Mll[n:2*n,n:2*n]),q)
	theMll[2*nb:3*nb,2*nb:3*nb] = np.dot(np.dot(p,Mll[2*n:3*n,2*n:3*n]),q)
	theMll[nb:2*nb,2*nb:3*nb] = np.dot(np.dot(p,Mll[n:2*n,2*n:3*n]),q)
	theMll[2*nb:3*nb,nb:2*nb] = np.dot(np.dot(p,Mll[2*n:3*n,n:2*n]),q)
	theMll[3*nb:4*nb,3*nb:4*nb] = np.dot(np.dot(p,Mll[3*n:4*n,3*n:4*n]),q)
	theMll[4*nb:5*nb,4*nb:5*nb] = np.dot(np.dot(p,Mll[4*n:5*n,4*n:5*n]),q)
	theMll[5*nb:6*nb,5*nb:6*nb] = np.dot(np.dot(p,Mll[5*n:6*n,5*n:6*n]),q)

	return theMll

def BinCls(p,cls):
	lmax = p.shape[1]-1
	out=[]
	for i in np.arange(len(cls)):
		out.append(np.dot(p,cls[i][0:lmax+1]))
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

def map_ang_from_edges(maskok):
	print('Calculating Angle Mask Map')
	nsmask = hp.npix2nside(len(maskok))
	### Get the list of pixels on the external border of the mask
	ip = np.arange(12*nsmask**2)
	neigh = hp.get_all_neighbours(nsmask, ip[maskok])
	nn = np.unique(np.sort(neigh.ravel()))
	nn = nn[maskok[nn] == False]
	### Get unit vectors for border and inner pixels
	vecpixarray_inner = np.array(hp.pix2vec(nsmask,ip[maskok]))
	vecpixarray_outer = np.array(hp.pix2vec(nsmask,nn))
	### get angles between the border pixels and inner pixels
	cosang = np.dot(np.transpose(vecpixarray_inner),vecpixarray_outer)
	mapang = np.zeros(12*nsmask**2)
	mapang[maskok] = np.degrees(np.arccos(np.max(cosang,axis=1)))
	return mapang

def apodize_mask(maskok,fwhm,mapang=None):
	if mapang == None:
		mapang = map_ang_from_edges(maskok)

	mask = 1 - np.exp(-mapang**2/(2*fwhm**2))
	return mask


def get_spectra(maps, maskmap, lmax, min_ell, delta_ell, wl=None, Mllmat=None, MllBinned=None, ellpq=None, MllBinnedInv=None):
	nside = hp.npix2nside(len(maps[0]))
	if wl == None:
		wl = hp.anafast(maskmap,regression=False)
		wl = wl[0:lmax+1]
	if Mllmat == None:
		Mllmat=Mll(lmax,wl)

	cutmaps = np.array([maps[0] * maskmap, maps[1] * maskmap, maps[2] * maskmap])
	cls = hp.anafast(cutmaps,pol=True)
	allcls= np.array([cls[0][0:lmax+1], cls[1][0:lmax+1], cls[2][0:lmax+1], cls[3][0:lmax+1], cls[4][0:lmax+1], cls[5][0:lmax+1]])

	if ellpq == None:
		all_ell = min_ell-1 + np.arange(1000)*delta_ell
		max_ell = np.max(all_ell[all_ell < lmax])
		nbins = np.int((max_ell + 1 - min_ell) / delta_ell)
		ell_low = min_ell + np.arange(nbins)*delta_ell
		ell_hi = ell_low + delta_ell - 1
		the_ell = np.arange(lmax+1)
		newl, p, q = make_pq(the_ell, ell_low, ell_hi)
	else:
		newl = ellpq[0]
		p = ellpq[1]
		q = ellpq[2]


	if MllBinned == None: MllBinned = MllBin(Mllmat, p, q)

	if MllBinnedInv == None: MllBinnedInv = np.linalg.inv(MllBinned)
	
	ClsBinned = BinCls(p,allcls)
	newcls = np.dot(MllBinnedInv, np.ravel(np.array(ClsBinned)))
	newcls = np.reshape(newcls, (6, p.shape[0]))
	return newcls, newl, Mllmat, MllBinned, MllBinnedInv, p, q, allcls




