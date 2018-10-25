from __future__ import print_function
import os
import healpy as hp 
import numpy as np
from pylab import *
rcParams['image.cmap'] = 'jet'
import sys
from pysimulators import FitsArray
##### Deep Networks
### Deep Networks Configuration for the case of T -> spectra
#from tensorflow import keras
import keras
#from tensorflow.keras.models import Sequential
from keras.models import Sequential
#from tensorflow.keras.layers import Dense
from keras.layers import Dense
import string
import glob
from qubic import Xpol
from qubic.utils import progress_bar

#arguments = sys.argv
arguments = string.split('~/Python/MySoft/Qubic/MachineLearning/train_and_test.py 64 100000 16 0.1 relu npixok_x_3 relu npixok_d_4 relu nbins_x_3 linear')



## Session ID:
session_id = 'mysim'
for s in arguments[1:]: session_id += '_' + str(s)
print(session_id)
if not os.path.exists(session_id):
	os.mkdir(session_id)

##### read training sample
ns = int(arguments[1])
nmax = int(arguments[2])
mask = FitsArray('mymask_ns{}.fits'.format(ns))
mymaps = FitsArray('mymaps_ns{}.fits'.format(ns))[0:nmax,:]
expcls = FitsArray('expcls_ns{}.fits'.format(ns))[0:nmax,:]
mycls = FitsArray('mycls_ns{}.fits'.format(ns))[0:nmax,:]
myXpolCls = FitsArray('myXpolCl_ns{}.fits'.format(ns))[0:nmax,:]

nbmodels = np.shape(mymaps)[0]
npixok = np.shape(mymaps)[1]
lt = np.arange(np.shape(mycls)[1])
lmin = 1
lmax = 2*ns-1
delta_l = int(arguments[3])
fsky = npixok*1./(12*ns**2)

xpol = Xpol(mask, lmin, lmax, delta_l)
mybins = xpol.ell_binned
nbins = len(mybins)

##### Rebinning
mybinnedcls = np.zeros((nbmodels, nbins))
bar = progress_bar(nbmodels,info='Rebinning Cl')
for i in xrange(nbmodels):
	bar.update()
	mybinnedcls[i,:] = mybins*(mybins+1)*xpol.bin_spectra(mycls[i,:])


myXpolCls = myXpolCls*mybins*(mybins+1)


clf()
num = np.random.randint(nmax)
plot(lt,lt*(lt+1)*expcls[num,:]/fsky,label='Anafast')
plot(lt,lt*(lt+1)*mycls[num,:],label='Input')
plot(mybins, mybinnedcls[num,:], '-o',label='Input Binned')
plot(mybins, myXpolCls[num,:], '-o',label='Xpol')
title(num)
legend()



## Fraction for training and post-validation
fraction = float(arguments[4])

## Number of layers
arg_layers = arguments[5:]
nlayers = len(arg_layers)/2+2


layer_info = [[arg_layers[0], npixok]]
for i in xrange(nlayers-2):
	theactivation = arg_layers[i*2]
	nodes_info = str.split(arg_layers[i*2+1], '_')
	if nodes_info[1]=='d':
		thennodes = eval(nodes_info[0]) / int(nodes_info[2])
	elif nodes_info[1]=='x':
		thennodes = eval(nodes_info[0]) * int(nodes_info[2])
	if thennodes != 0:
		layer_info.append([theactivation, thennodes])

layer_info.append([arg_layers[-1], nbins])

nlayers = len(layer_info)

print(layer_info)

model_T = Sequential()
print(layer_info)
print(nlayers)
for i in xrange(nlayers-1):
	if i==0:
		print('layer {}: input={} - Output = {} - Activation = {}'.format(i,layer_info[i][1], layer_info[i+1][1], layer_info[i+1][0]))
		model_T.add(Dense(units=layer_info[i][1], activation=layer_info[i][0]))
	else:
		print('layer {}: input={} - Output = {} - Activation = {}'.format(i,layer_info[i][1], layer_info[i+1][1], layer_info[i+1][0]))
		model_T.add(Dense(units=layer_info[i][1], activation=layer_info[i][0]))
model_T.add(Dense(units=layer_info[nlayers-1][1], activation=layer_info[nlayers-1][0]))


# Compile model
model_T.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

# Training
ilim = int(nbmodels*fraction)
print(ilim)

mxT = np.max(np.abs(mymaps))
myT = np.max(mybinnedcls)

class PrintNum(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 10 == 0: 
      print('')
      print(epoch, end='')
    sys.stdout.write('.')
    sys.stdout.flush()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)


historyT=model_T.fit(mymaps[0:ilim,:]/mxT, mybinnedcls[0:ilim,:]/myT, epochs=500, batch_size=500, 
  validation_split=0.1, verbose=0, callbacks=[early_stop, PrintNum()])
# historyT=model_T.fit(mymaps[0:ilim,:]/mxT, myXpolCls[0:ilim,:]/myT, epochs=500, batch_size=500, 
#   validation_split=0.1, verbose=0, callbacks=[early_stop, PrintNum()])

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
legend(['train: min = {0:5.2g}'.format(tmin), 'test: min = {0:5.2g}'.format(vmin)], loc='lower left', fontsize=8)
yscale('log')
title(session_id, fontsize=8)

print('Training done in {0:} epochs: training_loss_min = {1:5.2g} - test_loss_min = {2:5.2g}'.format(nepoch,tmin,vmin))


#np.savetxt(session_id+'/training.txt', np.array([arange(nepoch),historyT.history['loss'], historyT.history['val_loss']]).T, fmt='%i %10.5G %10.5G')


########## Validation sample

mymaps_test = mymaps[ilim:,:]
mycls_test = mycls[ilim:,:]
expcls_test = expcls[ilim:,:]
mybinnedcls_test = mybinnedcls[ilim:,:]
myXpolCls_test = myXpolCls[ilim:,:]
resultT = myT * model_T.predict(mymaps_test / mxT, batch_size=500)


subplot(2,1,2)
resid = expcls_test[:,:]/fsky*lt*(lt+1)-lt*(lt+1)*mycls_test[:,:]
#anafast_mean = np.mean(resid[:,2:])
#anafast_error = np.std(resid[:,2:])/np.sqrt(len(mybins-2))
#anafast_rms = np.std(resid[:,2:])
#la = 'Anafast: {0:5.1f}+/-{1:5.1f} RMS={2:5.1f}'.format(anafast_mean, anafast_error, anafast_rms)
#a=hist(np.ravel(resid[:,2:]), bins=100, range=[-2500,2500], alpha=0.5, label=la, normed=True)
rrr = resultT-mybinnedcls_test
ml_mean =np.mean(rrr)
ml_error = np.std(rrr)/np.sqrt(len(mybins-2))
ml_rms = np.std(rrr)
lml = 'ML T: {0:5.1f}+/-{1:5.1f} RMS={2:5.1f}'.format(ml_mean, ml_error, ml_rms)
a=hist(np.ravel(rrr), bins=100, range=[-2500,2500], alpha=0.5, label = lml, normed=True)
xxx = myXpolCls_test-mybinnedcls_test
xp_mean =np.mean(xxx)
xp_error = np.std(xxx)/np.sqrt(len(mybins-2))
xp_rms = np.std(xxx)
lxp = 'Xpol T: {0:5.1f}+/-{1:5.1f} RMS={2:5.1f}'.format(xp_mean, xp_error, xp_rms)
a=hist(np.ravel(xxx), bins=100, range=[-2500,2500], alpha=0.5, label = lxp, normed=True)
legend(fontsize=6)
savefig(session_id+'/result.png')


clf()
for k in xrange(9):
	subplot(3,3,k+1)
	num=np.random.randint(resultT.shape[0])
	ylim(-500,np.max(mybinnedcls_test[num,:])*1.2)
	plot(lt, lt*(lt+1)*mycls_test[num,:],label ='Input spectra')
	plot(lt, lt*(lt+1)*expcls_test[num,:]/fsky,label ='Anafast')
	plot(mybins, mybinnedcls_test[num,:],'o',label ='Input Binned')
	plot(mybins, resultT[num,:],'o-',label ='ML T')
	plot(mybins, myXpolCls_test[num,:],'o-',label ='Xpol T')
	title(num)
	if k==0: legend()
savefig(session_id+'/samples.png')


print(la)
print(lml)

f =open(session_id+'/result.txt','w')
f.write('{} {} {} {} {} {} {} {} {}'.format(nepoch, tmin, vmin, xp_mean, xp_error, xp_rms, ml_mean, ml_error, ml_rms))
f.close()












