from __future__ import division
import wave
import struct

## data = wave.open('FoliesdEspagne.wav')
## nb = data.getnframes()
## truc = data.readframes(100000)



def everyOther (v, offset=0):
   return [v[i] for i in range(offset, len(v), 2)]

def wavLoad (fname):
   wav = wave.open (fname, "r")
   (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams ()
   frames = wav.readframes (nframes * nchannels)
   out = struct.unpack_from ("%dh" % nframes * nchannels, frames)

   # Convert 2 channles to numpy arrays
   if nchannels == 2:
       left = array (list (everyOther (out, 0)))
       right = array (list  (everyOther (out, 1)))
   else:
       left = array (out)
       right = left

   return left, right, framerate

left, right, framerate = wavLoad('FoliesdEspagne.wav')
signal = np.abs(left+right)**2

time = arange(len(signal))/framerate

clf()
plot(time,signal)

from scipy import ndimage
theft = np.fft.fft(signal)
theps = np.abs(theft)**2
theps = theps/np.mean(theps)
freq = np.fft.fftfreq(len(signal))*framerate


########## find global tempo
## nombre de sous division dans un temps
nsub = 9
smoothedps = ndimage.gaussian_filter1d(theps, 1)

clf()
plot(freq, theps)
yscale('log')
xscale('log')
mask = (freq > 1) & (freq < 10)
xm = freq[mask][smoothedps[mask] == np.max(smoothedps[mask])]
for i in np.arange(nsub)+1: plot([xm*i,xm*i],[1e-10, 1e20],'r:')
plot(freq, smoothedps)
xlim(0.5*xm,nsub*2*xm)

### fit gaussians
from Homogeneity import fitting
def thegaussian(x,pars):
	return(pars[0]*np.exp(-(x-pars[1])**2/(2*pars[2]**2)))

clf()
uu=0
allx0 = np.zeros(nsub)
for i in np.arange(nsub)+1:
	uu += 1
	subplot(3,3,uu)
	mm = np.abs(freq-i*xm) < 0.3
	plot(freq[mm]/i, smoothedps[mm])
	ampguess = np.max(smoothedps[mm])
	x0guess = np.sum(freq[mm]/i*smoothedps[mm])/np.sum(smoothedps[mm])
	x02guess = np.sum((freq[mm]/i)**2*smoothedps[mm])/np.sum(smoothedps[mm])
	sigguess = np.sqrt(x02guess-x0guess**2)
	res = fitting.dothefit(freq[mm]/i, smoothedps[mm],freq[mm]*0+1,[ampguess,x0guess,sigguess], functname=thegaussian)
	plot([res[1][1],res[1][1]],[0, np.max(smoothedps[mm])],'r:')
	plot(freq[mm]/i, thegaussian(freq[mm]/i, res[1]))
	allx0[i-1]=res[1][1]

xm = np.mean(allx0)



noire = 1/xm
ixm = np.int(xm/framerate*len(signal))

inoire = np.int(noire * framerate)

########## trace tempo stability
size = inoire*16
newsignal = ndimage.gaussian_filter1d(signal,200)
newsignal=signal
signal_chunks = np.reshape(newsignal[0:np.int(len(newsignal)/size)*size], (np.int(len(newsignal)/size), size))
tt = np.arange(size)/framerate

allft = np.fft.fft(signal_chunks, axis = 1)
allps = np.abs(allft)**2

npl = 1
ff = np.fft.fftfreq(size)*framerate
clf()
for i in np.arange(npl):
	plot(ff, allps[i,:])
xscale('log')
yscale('log')
for i in np.arange(nsub)+1: plot([xm*i,xm*i],[1e-10, 1e30],'r:')
ylim(1e18,1e23)
xlim(1,100)


clf()
nn=20
plot(time, signal)
xlim(0,nn*noire)
ylim(-10000,10000)
decal = 0.
for i in np.arange(nn):
	plot([decal+i*noire, decal+i*noire], [-10000,10000],'r:')






