from __future__ import division
import numpy as np
from matplotlib import pyplot as mp

#### pi using mahdava formula
kmax = 29
k = np.arange(kmax)
pi_est = (np.sqrt(12)*((-1/3)**k)/(2*k+1)).sum()

#### Strides
bla = np.zeros((5000,3))
blaT = bla.T
%timeit [ [bla[i,j]+1 for i in np.arange(bla.shape[0])] for j in np.arange(bla.shape[1])]
%timeit [ [blaT[i,j]+1 for i in np.arange(blaT.shape[0])] for j in np.arange(blaT.shape[1])]

#### array initialization
%timeit np.zeros((1000,1000))
%timeit np.zeros((1000,1000))+1
%timeit np.ones((1000,1000))
%timeit np.empty((1000,1000))


#### Boolean
tf = np.array([True, False])
tf & tf[::-1]
tf | tf[::-1]

#### Exercise: Pi calculation
NTOTAL = 1000000
d = (np.random.random((2, NTOTAL))**2).sum(axis=0)
mask = d < 1
%timeit 4*mask.sum()/NTOTAL
%timeit 4*d[mask].size/NTOTAL

#### Exercise: Histogram
def velocity2speed(velocity, ndims):
    return np.sqrt((velocity[:,0:ndims]**2).sum(axis=1))


def speed_distribution(speed, ndims):
    return (np.pi/2)**(-np.abs(ndims-2)/2)*speed**(ndims-1)*exp(-0.5*speed**2)


NPARTICULES = 1000000

velocity = np.random.standard_normal((NPARTICULES, 3))

clf()
for ndims in (1, 2, 3):
    speed = velocity2speed(velocity, ndims)
    ax = mp.subplot(1, 3, ndims)
    n, bins, patches = ax.hist(speed,normed=True,bins=30,range=[0,5])
    ax.set_title('{}-d speed distribution'.format(ndims))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel('speed')
    ax.plot(bins,speed_distribution(bins,ndims), 'r', linewidth=2)

mp.show()

###############
n=15
vect = np.random.random(100*n)

%timeit mm0 = np.array([vect[i*100:(i+1)*100].mean() for i in np.arange(n)])
%timeit mm1 = np.reshape(vect,(n,100)).mean(axis=1)
%timeit mm2 = fromiter((v.mean() for v in vect.reshape(-1, 100)), float)

#### Views and copies
def isview(a, b):
        """
        Return True if b is a view of a.
        (It is assumed that a's memory buffer is contiguous)
        """
        return a.ctypes.data <= b.ctypes.data < a.ctypes.data + a.nbytes

a = np.arange(24, dtype=float)
a.shape = (3, 2, 4)

isview(a, a[:2, 1, 1:3])

a[:, ::-1, :]
a.view(complex)
a.view([('position', float, 3), ('mass', float)])
a.reshape((6, -1))
a[..., None]
a.ravel()
a.flatten()
a.T
a.T.ravel()
a.swapaxes(0, 1)
np.rollaxis(a, 2)
a.astype(int)
a.astype(float)
np.asarray(a)
np.asarray(a, dtype=float)
np.asarray(a, dtype=int)
np.array(a, dtype=float, copy=False)

#### Broadcasting
np.zeros((5, 9)) + np.zeros(9)
np.zeros((5, 9)) + np.zeros(5)
np.zeros((5, 9)) + np.zeros(5)[:, None]


from __future__ import division
import numpy as np

NDETECTORS = 8
NSAMPLES = 1000
SAMPLING_PERIOD = 0.1
GLITCH_TAU = 0.3
GLITCH_AMPL = 20
GAIN_SIGMA = 0.03
SOURCE_AMPL = 7
SOURCE_PERIOD = 10
NOISE_AMPL = 0.7

time = np.arange(NSAMPLES) * SAMPLING_PERIOD
glitch = np.zeros(NSAMPLES)
glitch[100:] = GLITCH_AMPL * np.exp(-time[:-100] / GLITCH_TAU)
gain = 1 + GAIN_SIGMA * np.random.standard_normal(NDETECTORS)
offset = np.arange(NDETECTORS)
source = SOURCE_AMPL * np.sin(2 * np.pi * time / SOURCE_PERIOD)
noise = NOISE_AMPL * np.random.standard_normal((NDETECTORS, NSAMPLES))

def toto():
    signal = np.empty((NDETECTORS, NSAMPLES))
    for idet in xrange(NDETECTORS):
        for isample in xrange(NSAMPLES):
            signal[idet, isample] = gain[idet] * source[isample] + \
                glitch[isample] + offset[idet] + \
                noise[idet, isample]
    return signal

%timeit signal = toto()
%timeit signal2 = np.outer(gain,source) + glitch + offset[:, None] + noise
%timeit signal2 = gain[:, None] + source + glitch + offset[:, None] + noise

signal=toto()
mp.figure()
mp.subplot('211')
mp.imshow(signal, aspect='auto', interpolation='none')
mp.xlabel('sample')
mp.ylabel('detector')
mp.subplot('212')
for s in signal:
    mp.plot(time, s)
    mp.xlabel('time [s]')
    mp.ylabel('signal')
mp.show()

#### Normalisation d'un tableau
n = 10000
m = 5
vec=np.random.random((n,m))

def normalize(vec):
    return vec / np.sqrt(np.sum(vec**2,axis=1))[:, None]

def normalize_ml(vec):
    vec2 = vec**2
    vec2s = np.sum(vec2,axis=1)
    norm = np.sqrt(vec2s)
    return vec / norm[:, None]

def normalize_slow(vec):
    sh = vec.shape
    vecnorm =  np.empty_like(vec)
    norm = np.sqrt(np.sum(vec**2,axis=1))
    for i in np.arange(sh[0]):
        vecnorm[i,:] = vec[i,:] / norm[i]
    return(vecnorm)
        
%timeit nvec = normalize(vec)
%timeit nvec_ml = normalize_ml(vec)
%timeit nvec2 = normalize_slow(vec)

##############################################################
## Universal functions
##############################################################

#### 1: Buffer on output
N = 1000000
x = np.random.random_sample(N)
%timeit 2 * np.sin(x) + x

out = np.empty_like(x)
%timeit global out; np.sin(x, out); out *= 2; out += x

##### 2: Methods
N = 10000
x = np.random.random_sample(N)
uf = np.add

## reduce
def ufred(x,uf):
    r = x[0]
    for i in range(1, len(x) - 1):
        r = uf(r, x[i])
    return r

%timeit ufred(x, uf)
%timeit uf.reduce(x)

## accumulate
def ufacc(x,uf):
    a = np.empty(len(x))
    a[0] = x[0]
    for i in range(1, len(x) - 1):
        a[i] = uf(a[i - 1], x[i])
    return a

%timeit ufacc(x, uf)
%timeit uf.accumulate(x)

## outer
def ufout(x, y, uf):
    z = np.empty((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            z[i,j] =  x[i] * y[j]
    return z

x = np.random.random_sample(1000)
y = np.random.random_sample(500)
%timeit ufout(x, y, uf)
%timeit uf.outer(x,y)

### Exercise
M = 500
N = 10000

def calc_a(M, N):
    A = np.empty((M, N), dtype='int64')
    for j in range(N):
        for i in range(M):
            A[i,j] = i + j
    return A

A = calc_a(M, N)
A2 = np.add.outer(arange(M), arange(N))
A3 = arange(M)[:, None] + arange(N)

%timeit calc_a(M, N)
%timeit np.add.outer(arange(M), arange(N))
%timeit arange(M)[:, None] + arange(N)

M = 500
N = 10000

def calc_b(M, N):
    B = np.empty((M, N), dtype='int64')
    for j in xrange(N):
        for i in xrange(M):
            B[i,j] = i * j
    return B

B = calc_b(M, N)
B2 = np.multiply.outer(arange(M), arange(N))
B3 = arange(M)[:, None] * arange(N)

%timeit calc_b(M, N)
%timeit np.multiply.outer(arange(M), arange(N))
%timeit arange(M)[:, None] * arange(N)

I = arange(M)
J = arange(N)
%timeit calc_b(M, N)
%timeit np.multiply.outer(I, J)
%timeit I[:, None] * J



#################################################
#### 11: special values



#################################################
#### 12: special values

# ex/
point_dtype = [('x', float), ('y', float), ('z', float)]
n = 100
points = np.empty(n, dtype=point_dtype)
points['x'] = np.random.random_sample(n)
points['y'] = np.random.random_sample(n)
points['z'] = np.random.random_sample(n)
points[0]
points[10] = (1, 1 , 0)

# ex/
spectra_dtype = [('fluxdensity', float, 100),
                 ('wavelength', float, 100)]
spectrum = np.zeros((), dtype=spectra_dtype)
spectrum['wavelength'].size

# ex/
galaxy_dtype = [('name', 'S256'),
                ('pos', point_dtype)]
galaxy = np.zeros(10, dtype=galaxy_dtype)
galaxy[0] = ('M81', (1, -1, 0))
galaxy[0]['name']
galaxy[0]['pos']['x'], galaxy[0]['pos']['y'], galaxy[0]['pos']['z']

#### Exercise: Indirect Sort
person_dtype = [('name', 'S10'), ('age', float)]
nb  = 10000
guys = np.empty(nb, dtype=person_dtype)
guys['name'] = np.char.add('id',arange(nb).astype('str'))
guys['age'] = np.random.random_sample(nb)*100

# method 1
order = np.argsort(guys['age'])
ordered_guys = guys[order]

# method 1
ordered_guys2 = np.sort(guys, order='age')



############ 13: Record Arrays
source_dtype = [('name', 'S256'),
                ('ra', float),
                ('dec', float)]
source = np.recarray(10, dtype=source_dtype)
source[0] = ('M81', 148.8882208, 69.0652947)
print(source[0].name, source[0].ra, source[0].dec)
print(source[0]['name'], source[0]['ra'], source[0]['dec'])

#### Attention
source[[0,1,2]].name = 'toto'
source.name[[0,1,2]] = 'toto'


source = np.empty(10, dtype=source_dtype).view(np.recarray)
source[0] = ('M81', 148.8882208, 69.0652947)
source[0].name
source.name[0]

############## 14: Dense linear algebra

############## More exercises
#### Fitting
"""
non-linear fit example with 3 parameters model 1/f noise :
   variance*[ 1 + (f_knee/f)^alpha ]

Jm. Colley
"""
from __future__ import division, print_function
import numpy as np
import scipy.optimize as spo
from matplotlib import pyplot as mp
from astropy.io import fits

FREQ_SAMPLING = 180


def spectrum_estimate(signal):
    """
    Return power spectrum mode 0 to Nyquist
    """
    size_signal = signal.size
    fft_signal = np.fft.fft(signal)
    ps = abs(fft_signal)**2 / size_signal
    return ps[:1 + size_signal // 2]

# inspect the FITS Table
fits.info('data_oof.fits')
hdulist = fits.open('data_oof.fits')
print(hdulist[1].header)

# read the FITS table as a record array
table = hdulist[1].data
signal = table.timeline
nsamples = signal.size

# create array freq mode 1 to Nyquist Mode
delta_freq = FREQ_SAMPLING / nsamples
freq = delta_freq * np.arange(1, nsamples // 2 + 1)

# remove mode 0
spectrum = spectrum_estimate(signal)[1:]

def oof_model(freq, param):
    """
    1/f noise model
        param[0] : noise standard deviation
        param[1] : knee frequency
        param[2] : alpha
    freq : array of frequency
    """
    sigma, fknee, alpha = param
    return sigma**2*( 1 + (fknee/freq)**alpha )


def compute_residuals(param, observation, freq):
    """
    Return array: observation - model
    """
    model = oof_model(freq, param)
    residual = np.log(observation / model)
    print("residual: ", np.sum(residual**2))
    return residual

# fit with scipy optimize, leastsq() function
param_guess = [np.std(signal), 1, 1.2] * np.random.uniform(0.5, 2, 3)
ret_lsq = spo.leastsq(compute_residuals, param_guess, args=(spectrum, freq),
                      full_output=True)

param_est, cov_x, infodict, mesg_result, ret_value = ret_lsq
print("Return value:", ret_value)
print("Return message:", mesg_result)

if ret_value not in (1, 2, 3, 4):
    raise RuntimeError(mesg_result)

print("guess    :", param_guess)
print("solution :", param_est)

# plot
mp.figure()
mesg_title = ('Fit of power spectrum: '
              r'${\rm PSD}(f)=\sigma^2'
              r'\left(1+(\frac{f_{\rm knee}}{f})^\alpha\right)$'
              '\n')
mp.title(mesg_title)
mp.xlabel('Hertz')
mp.loglog(freq, spectrum)
mp.loglog(freq, oof_model(freq, param_guess), '.-')
mp.loglog(freq, oof_model(freq, param_est))
mp.ylim(ymax=1.e-4)
mp.grid()
sigma_param_est = np.sqrt(np.diagonal(cov_x))
mesg_fit = (
    r'     $\sigma={:5.3g}\pm{:3.2g}$'.format(
        param_est[0], sigma_param_est[0]) + '\n'
    r'fit: $f_{{\rm knee}}={:5.3f}\pm{:3.2f}$'.format(
        param_est[1], sigma_param_est[1]) + '\n'
    r'     $\alpha={:5.3f}\pm{:3.2f}$'.format(
        param_est[2], sigma_param_est[2]))
mp.legend(['raw spectrum', 'guess', mesg_fit], loc='best')
mp.show()


######### Condition number and error propagation
A = np.array([[3, -sqrt(3), 1, -sqrt(3)],
              [sqrt(3), 3, -sqrt(3), -1],
              [sqrt(3), 1, sqrt(3), 3],
              [1, -sqrt(3), -3, sqrt(3)]])
B = np.array([[10, 10, 7, 8],
              [9, 2, 7, 7],
              [1, 5, 11, 1],
              [10, 11, 4, 8]])
x = np.array([1.,1.,1.,1.])
bA = np.dot(A,x)
bB = np.dot(B,x)

### Orthogonality of A
np.dot(A,A.T)

### Condition number of A and B: use the SVD values, not the eigenvalues !!!!
eA = np.linalg.eigvals(A)
eB = np.linalg.eigvals(B)
cnA = np.max(abs(eA)) / np.min(abs(eA))
cnB = np.max(abs(eB)) / np.min(abs(eB))

uA, sA, vA = np.linalg.svd(A)
uB, sB, vB = np.linalg.svd(B)
cnA = np.max(abs(sA)) / np.min(abs(sA))
cnB = np.max(abs(sB)) / np.min(abs(sB))

np.linalg.cond(A)
np.linalg.cond(B)

### Inverse
Ainv = np.linalg.inv(A)
Binv = np.linalg.inv(B)

np.dot(A, Ainv)
np.dot(B, Binv)

#### Monte-Carlo
nb = 10000
sig = 1.

xnorm = np.sqrt(np.sum(x**2))
bAnorm = np.sqrt(np.sum(bA**2))
bBnorm = np.sqrt(np.sum(bB**2))
ratioA = np.empty(nb)
ratioB = np.empty(nb)

for i in np.arange(nb):
    deltabA = np.random.randn(len(bA))*sig
    deltabB = np.random.randn(len(bB))*sig
    deltaxA = np.dot(Ainv, deltabA)
    deltaxB = np.dot(Binv, deltabB)
    dx_x_A = np.sqrt(np.sum(deltaxA**2)) / xnorm
    dx_x_B = np.sqrt(np.sum(deltaxB**2)) / xnorm
    db_b_A = np.sqrt(np.sum(deltabA**2)) / bAnorm
    db_b_B = np.sqrt(np.sum(deltabB**2)) / bBnorm
    ratioA[i] = dx_x_A / db_b_A
    ratioB[i] = dx_x_B / db_b_B
    
    
np.mean(ratioA)/cnA
np.mean(ratioB)/cnB


##### Verify Stefan-Boltzman Law
nb = 10000
nu = np.linspace(nb)*1000*1e9
bnu = 2 * h * nu**3/c**2 * 1/.(np.exp()






