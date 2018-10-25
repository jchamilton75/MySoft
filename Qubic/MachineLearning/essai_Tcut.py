# -*- coding: utf-8 -*-
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

nbmodels = 10000
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

clf()
for i in xrange(nbmodels):
  if (i/nnn)*nnn == i:
    plot(lt, lt*(lt+1)*mycls[i,:])
    
clf()
xlim(0,3*ns+10)
for i in xrange(nbmodels):
  if (i/nnn)*nnn == i:
    plot(ls, allshapes[i,:])

#### Rebinning
delta_l = 16
nbins = nl/delta_l
mybinnedcls = np.zeros((nbmodels, nbins))
mybins = np.zeros(nbins)
for i in xrange(nbmodels):
  if (i/nnn)*nnn == i: 
    print(i,nbmodels)
  dl = lt*(lt+1)*mycls[i,:]
  for k in xrange(nbins):
    mybinnedcls[i,k] = np.mean(dl[k*delta_l:(k+1)*delta_l])
    if i==0: mybins[k] = np.mean(lt[k*delta_l:(k+1)*delta_l])


num = np.random.randint(0,nbmodels)
clf()
fsky = npixok * 1. / (12*ns**2)
plot(lt, lt*(lt+1)*mycls[num,:])
plot(lt, lt*(lt+1)*expcls[num,:] / fsky)
plot(mybins, mybinnedcls[num,:], 'ro-')



### Deep Networks Configuration for the case of T -> spectra
#from tensorflow import keras
import keras
#from tensorflow.keras.models import Sequential
from keras.models import Sequential
model_T = Sequential()

#from tensorflow.keras.layers import Dense
from keras.layers import Dense
model_T.add(Dense(units=npixok*3, activation='relu', input_dim=npixok, kernel_initializer='uniform'))
model_T.add(Dense(units=npixok/6, activation='relu'))
model_T.add(Dense(units=nbins*3, activation='relu'))
model_T.add(Dense(units=nbins, activation='linear'))

model_T.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

# Training
fraction = 0.9
ilim = int(nbmodels*fraction)
print(ilim)

mxT = np.max(np.abs(mymaps))
myT = np.max(mybinnedcls)

from __future__ import print_function
class PrintNum(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 10 == 0: 
      print('')
      print(epoch, end='')
    sys.stdout.write('.')
    sys.stdout.flush()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)


historyT=model_T.fit(mymaps[0:ilim,:]/mxT, mybinnedcls[0:ilim,:]/myT, epochs=500, batch_size=500, 
  validation_split=0.1, verbose=1, callbacks=[early_stop, PrintNum()])

# summarize history for loss
nepoch = len(historyT.history['val_loss'])
vmin = min(historyT.history['val_loss'])
tmin = min(historyT.history['loss'])
clf()
subplot(2,1,1)
plot(historyT.history['loss'])
plot(historyT.history['val_loss'])
title('model loss')
ylabel('loss')
xlabel('epoch')
legend(['train: min = {0:5.2g}'.format(tmin), 'test: min = {0:5.2g}'.format(vmin)], loc='lower left')
yscale('log')
#title(session_id)
show()
print(min(historyT.history['loss']), min(historyT.history['val_loss']), len(historyT.history['val_loss']))


mymaps_test = mymaps[ilim:,:]
mycls_test = mycls[ilim:,:]
expcls_test = expcls[ilim:,:]
mybinnedcls_test = mybinnedcls[ilim:,:]

resultT = myT * model_T.predict(mymaps_test / mxT, batch_size=500)

clf()
num=np.random.randint(resultT.shape[0])
ylim(-500,np.max(mybinnedcls_test[num,:])*1.2)
plot(lt, lt*(lt+1)*mycls_test[num,:],label ='Input spectra')
plot(lt, lt*(lt+1)*expcls_test[num,:]/fsky,label ='Anafast')
plot(mybins, mybinnedcls_test[num,:],'o',label ='Input Binned')
plot(mybins, resultT[num,:],'o-',label ='ML T')
title(num)
legend()

clf()
resid = expcls_test[:,:]/fsky*lt*(lt+1)-lt*(lt+1)*mycls_test[:,:]
a=hist(np.ravel(resid[:,2:]), bins=100, range=[-10000,10000], alpha=0.5, label='Anafast: {0:5.2f}+/-{1:5.2f} RMS={2:5.2f}'.format(np.mean(resid[:,2:]), np.std(resid[:,2:])/np.sqrt(len(mybins-2)), np.std(resid[:,2:])), normed=True)
rrr = resultT-mybinnedcls_test
a=hist(np.ravel(rrr), bins=100, range=[-10000,10000], alpha=0.5, label = 'ML T: {0:5.2f}+/-{1:5.2f} RMS={2:5.2f}'.format(np.mean(rrr), np.std(rrr)/np.sqrt(len(mybins-2)), np.std(rrr)), normed=True)
legend()


#### Notes: https://www.evernote.com/l/AIl7Djc8kHVJ9Y-x4KN6_sxwaPzjK66g16w


