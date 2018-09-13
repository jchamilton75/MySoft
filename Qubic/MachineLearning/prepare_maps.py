import healpy as hp 
import numpy as np
from pylab import *
rcParams['image.cmap'] = 'jet'

from pysimulators import FitsArray
CL = FitsArray('cltest.fits')


ls = np.arange(CL.shape[0])
clf()
plot(ls,CL*ls*(ls+1), color='k')

ns = 64
themap =hp.synfast(CL[0:5*ns], ns, pixwin=False)
hp.mollview(themap)

lmax = 2*ns-1
nl = 2*ns
nalm = (2*ns)*(2*ns+1)/2
outcl, outalm = hp.anafast(themap,alm=True, lmax=lmax)
print(outcl.shape, nl)
print(outalm.shape, nalm)
print(outalm.dtype)
print(outalm[0:50])
ll = np.arange(outcl.shape[0])
clf()
plot(ll,ll*(ll+1)*outcl)
plot(ls, ls*(ls+1)*CL, 'r')
xlim(0,max(ll))

### Define a mask
center_gal = np.array([316.45, -58.76])
pixcenter = hp.ang2pix(ns, np.radians(90-center_gal[1]), np.radians(center_gal[0]))
veccenter = np.array(hp.pix2vec(ns, pixcenter))
vecall = np.array(hp.pix2vec(ns, arange(12*ns**2)))

scalprod = np.zeros(12*ns**2)
for i in xrange(12*ns**2): scalprod[i] = np.dot(veccenter, vecall[:,i])
ang = np.arccos(scalprod)

#thescalprod = np.dot(veccenter, vecall)
#ang = np.arccos(np.dot(veccenter, vecall))

okpix = np.degrees(ang) < 20.
npixok = okpix.sum()

mask = np.zeros(12*ns**2)
mask[okpix] = 1
hp.mollview(mask)

FitsArray(mask).save('mymask_ns{}.fits'.format(ns))

mappatch = themap * mask
clpatch = hp.anafast(mappatch, lmax = 2*ns-1)
hp.mollview(mappatch)

clf()
plot(ll,ll*(ll+1)*clpatch/np.sum(mask)*len(mask))
plot(ls, ls*(ls+1)*CL, 'r')
xlim(0,max(ll))
print(np.sum(mask),len(mask))


### Target power spectra
clt = CL[0:nl]
lt = ll[0:nl]

nbmodels = 100000
nnn = int(nbmodels/1000)
limit_shape = 3*ns
mymaps = np.zeros((nbmodels, npixok))
myalms = np.zeros((nbmodels, nalm), dtype=complex128)
expcls = np.zeros((nbmodels, nl))
mycls = np.zeros((nbmodels, nl))
allshapes = np.zeros((nbmodels, len(ls)))
for i in xrange(nbmodels):
  ylo = np.random.rand()*2
  yhi = np.random.rand()*2
  #print(i)
  if (i/nnn)*nnn == i: 
    print(i,nbmodels,ylo,yhi)
  theshape = ylo+(yhi-ylo)/(limit_shape)*ls
  theshape[theshape < 0] = 0
  theshape[limit_shape:] = 0
  allshapes[i,:] = theshape
  theCL = CL*theshape
  themap = hp.synfast(theCL, ns, pixwin=False, verbose=False) * mask
  mymaps[i,:] = themap[okpix]
  expcls[i,:], myalms[i,:] = hp.anafast(themap, lmax=lmax, alm=True)
  mycls[i,:] = theCL[0:nl]

from pysimulators import FitsArray

FitsArray(mymaps).save('mymaps_ns{}.fits'.format(ns))
FitsArray(expcls).save('expcls_ns{}.fits'.format(ns))
FitsArray(mycls).save('mycls_ns{}.fits'.format(ns))



clf()
for i in xrange(nbmodels):
  if (i/nnn)*nnn == i:
    plot(lt, lt*(lt+1)*mycls[i,:])
    
clf()
xlim(0,3*ns+10)
for i in xrange(nbmodels):
  if (i/nnn)*nnn == i:
    plot(ls, allshapes[i,:])

