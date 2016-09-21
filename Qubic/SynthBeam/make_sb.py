from qubic import QubicScene
from SynthBeam import myinstrument
import healpy as hp
from pyoperators.utils import split
from pyoperators import (Cartesian2SphericalOperator)
import numexpr as ne
from scipy.constants import c, h, k
from pyoperators.utils.ufuncs import abs2


def plotcirc(center, radius, color=None):
    ang = np.linspace(0,pi,100)
    plot(center[0]+radius*cos(ang), center[1]+radius*sin(ang), color=color)

scene = QubicScene(256)
inst = myinstrument.QubicInstrument(filter_nu=150e9)

### horns
#clf()
#inst.horn.plot()

###
centers = inst.horn.center[:,0:2]

sb = inst.get_synthbeam(scene, 0)
hp.gnomview(sb, rot=[0,90], reso=10)


### Now we try to reconstruct it from scratch
    
#position = inst.detector.center           # Detector Location
xx = linspace(-0.06,0.06,100)
xy = []
for xi in xx:
    for yj in xx:
        xy.append([xi,yj,-0.3])
position=np.array(xy)

area = inst.detector.area                       # Detector Area
nu = inst.filter.nu
bandwidth = inst.filter.bandwidth
spectral_irradiance = bandwidth
horn = inst.horn
primary_beam = inst.primary_beam
secondary_beam = inst.secondary_beam 
synthbeam_dtype = inst.synthbeam.dtype
theta_max = 45.

### inside _get_synthbeam
MAX_MEMORY_B = 1e9
theta, phi = hp.pix2ang(scene.nside, scene.index)
index = np.where(theta <= np.radians(theta_max))[0]

theta = theta[index]
phi = phi[index]

nhorn = int(np.sum(horn.open))
npix = len(index)
nbytes_B = npix * nhorn * 24
ngroup = np.ceil(nbytes_B / MAX_MEMORY_B)
outfinal = np.zeros(position.shape[:-1] + (len(scene),),
               dtype=synthbeam_dtype)

### Response A
# out : complex array of shape (#positions, #horns)
#     The phase and transmission from the horns to the focal plane.
#A = inst._get_response_A(position, area, nu, horn, secondary_beam)
uvec = position / np.sqrt(np.sum(position**2, axis=-1))[..., None]
thetaphi = Cartesian2SphericalOperator('zenith,azimuth')(uvec)
sr = -area / position[..., 2]**2 * np.cos(thetaphi[..., 0])**3
tr = np.sqrt(secondary_beam(thetaphi[..., 0], thetaphi[..., 1]) *
             sr / secondary_beam.solid_angle)[..., None]
const = 2j * np.pi * nu / c
product = np.dot(uvec, horn[horn.open].center.T)
A = ne.evaluate('tr * exp(const * product)')

ih=122
clf()
subplot(1,3,1)
amp = np.reshape(abs2(A[:,ih])/np.max(abs2(A)),(100,100))
imshow(amp, extent= [np.min(xx), np.max(xx), np.min(xx), np.max(xx)])
colorbar()
subplot(1,3,2)
phase = np.reshape(angle(A[:,ih]),(100,100))
imshow(phase, extent= [np.min(xx), np.max(xx), np.min(xx), np.max(xx)])
colorbar()
draw()
subplot(1,3,3)
inst.horn.plot(facecolor_open='grey')
plot(inst.horn.center[ih,0],inst.horn.center[ih,1],'ro')
draw()





### Response B : la gaussienne de chaque cornet sur le ciel...
### Donc à remplacer par le beam de chaque cornet B.shape = (400, npix<45deg)
# out : complex array of shape (#horns, #sources)
#     The phase and amplitudes from the sources to the horns.
#B = inst._get_response_B(
#    theta, phi, spectral_irradiance, nu, horn, primary_beam)
shape = np.broadcast(theta, phi, spectral_irradiance).shape
theta, phi, spectral_irradiance = [np.ravel(_) for _ in theta, phi,
                                   spectral_irradiance]
uvec = hp.ang2vec(theta, phi)
source_E = np.sqrt(spectral_irradiance *
                   primary_beam(theta, phi) * np.pi * horn.radius**2)
const = 2j * np.pi * nu / c
product = np.dot(horn[horn.open].center, uvec.T)
out = ne.evaluate('source_E * exp(const * product)')
B = out.reshape((-1,) + shape)


E = np.dot(A, B.reshape((B.shape[0], -1))).reshape(
    A.shape[:-1] + B.shape[1:])

outfinal[index]= abs2(E, dtype=synthbeam_dtype)
hp.gnomview(outfinal, rot=[0,90], reso=10)

#### Here is the synthesized beam
hp.mollview(np.log10(outfinal-sb), rot=[0,90])

