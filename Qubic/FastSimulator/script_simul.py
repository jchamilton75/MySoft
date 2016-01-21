from __future__ import division

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp

from qubic_v1 import get_random_pointings
from qubic_v1 import simulation
from qubic import QubicInstrument
from pyoperators import pcg, DiagonalOperator, UnpackOperator
from pysimulators import ProjectionInMemoryOperator

kmax = 2
#pointings = get_random_pointings(1000, 10)

racenter=0.0
deccenter=-57.0
angspeed=2 # deg/sec
delta_az=15.
angspeed_psi=0.1
maxpsi=45.
nsweeps_el=300
duration=24   # hours
ts=1         # seconds
pointings,azel = simulation.get_sweeping_pointings([racenter,deccenter],duration,ts, angspeed, delta_az, nsweeps_el,angspeed_psi,maxpsi)

fig=figure()
fig.add_axes([0.1, 0.1, 0.8, 0.8],projection='polar')
plot(azel[0]*np.pi/180,90-azel[1],'.')
fig.suptitle('Azimuth Elevation')

fig2=figure()
plot(pointings[:,2])
xlabel('Pitch angle')

fig=figure()
plot(pointings[:,1],90-pointings[:,0],',')
xlabel('ra')
ylabel('dec')
mp.show()

map_orig = hp.read_map('/Volumes/Data/Qubic/qubic_v1/syn256.fits')

q = QubicInstrument()
C = q.get_convolution_peak_operator()
P = q.get_projection_peak_operator(pointings, kmax=kmax)
H = P * C

### TOD
tod = H(map_orig)
ndet,npix=tod.shape
#ikeep=511
    #for i in arange(ndet):
    #if i != ikeep:
#    tod[i,:]=0


### noise
white_noise_sig=30
alpha=1
fknee=0.1
fech=1./ts
thenoise=np.zeros((ndet,npix))
for i in arange(ndet):
    thenoise[i,:]=simulation.noise1f(fech,len(pointings),white_noise_sig,fknee,alpha)

tod=tod+thenoise
tod.imshow()


coverage = P.T(np.ones_like(tod))
coverage = P.T(tod)
hp.gnomview(coverage, rot=[racenter,deccenter], reso=7, xsize=600, min=0, max=30, title='Coverage map')
mp.show()

mask = coverage < 10
P.matrix.pack(mask)
P_packed = ProjectionInMemoryOperator(P.matrix)
unpack = UnpackOperator(mask)

solution = pcg(P_packed.T * P_packed, P_packed.T(tod), M=DiagonalOperator(1/coverage[~mask]), disp=True)
map = unpack(solution['x'])

orig = map_orig.copy()
orig[mask] = np.nan
hp.gnomview(orig, rot=[racenter,deccenter], reso=7, xsize=600, min=-200, max=200, title='Original map')
cmap = C(map_orig)
cmap[mask] = np.nan
hp.gnomview(cmap, rot=[racenter,deccenter], reso=7, xsize=600, min=-200, max=200, title='Convolved original map')
map[mask] = np.nan
hp.gnomview(map, rot=[racenter,deccenter], reso=7, xsize=600, min=-200, max=200, title='Reconstructed map (simulpeak)')
hp.gnomview(cmap-map, rot=[racenter,deccenter], reso=7, xsize=600, min=-200, max=200, title='Difference map')

mp.show()

orig = map_orig.copy()
orig[mask] = np.nan
hp.mollview(orig, xsize=600, min=-200, max=200, title='Original map')
cmap = C(map_orig)
cmap[mask] = np.nan
hp.mollview(cmap, xsize=600, min=-200, max=200, title='Convolved original map')
map[mask] = np.nan
hp.mollview(map, xsize=600, min=-200, max=200, title='Reconstructed map (simulpeak)')
hp.mollview(map-cmap, xsize=600, min=-200, max=200, title='Difference map')
mp.show()




#1/f noise
white=30.
fknee=1
alpha=1

nb=100000
fech=100
time=np.arange(nb)/fech
gauss=np.random.randn(nb)
timestep=1./fech
freq=np.fft.fftfreq(nb,d=timestep)
ftgauss=np.fft.fft(gauss)
ftgauss[0]=0

powspec=white**2*(1+abs(fknee/freq)**(alpha))
powspec[0]=0

spec=ftgauss*sqrt(abs(powspec))
noise=np.fft.ifft(spec)

clf()
plot(time,noise)

clf()
yscale('log')
xscale('log')
ylim(min(powspec[1:])/10,max(powspec)*10)
plot(freq,abs(np.fft.fft(noise))**2/nb,',')
plot(freq,powspec,lw=4,color='red')



