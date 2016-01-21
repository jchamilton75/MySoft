from Cosmology import pyxi_old
from Cosmology import pyxi
import cosmolopy
from matplotlib import cm
from scipy import interpolate
import scipy.integrate
import pycamb
import sys
import numpy as np
from pysimulators import FitsArray
import scipy.ndimage.filters as spf


################# Input Power spectrum ###################################
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
r = 0.05
h=0.7
H0 = h*100
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
         'tensor_ratio':r,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}


########## returns P9K0 with k in Mpc-1 and P(k) in Mpc**3
def get_pk(cosmo,z, kmax=10, nk=32768):
	h = cosmo['H0']/100
	kvals = np.linspace(0,kmax/h,nk)
	pk=pycamb.matter_power(redshifts=[z], k=kvals, **cosmo)[1]
	pk[0]=0
	return kvals*h, pk / h**3


##### Useful piece of code
def statstr(vec,divide=False):
	m=np.mean(vec)
	s=np.std(vec)
	if divide: s/=sqrt(len(vec))
	return '{0:.4f} +/- {1:.4f}'.format(m,s)


##### Log Normal Slice with Java code from Nico
def LogNormalSlice(k,pk,nboxsize=256l, boxsize=1600):
	from subprocess import Popen, PIPE
	execution_dir = '/Users/hamilton/idl/pro/SDSS-APC/idl/Simulations'
	### Write P(k)
	filepk = execution_dir+'/'+'Pktmp.txt'
	outfile = file(filepk,'wb')
	for x in zip(k,pk):
		outfile.write(('%10f\t%10f\n')%x)
	sys.stdout.flush()
	cmd = './Run.sh '+str(nboxsize)+' '+str(boxsize)+' '+filepk+' 1 0 1 toto.txt'
	print(cmd)
	process = Popen(cmd.split(), cwd=execution_dir)#, stdout=PIPE, stderr=PIPE)
	stdout, stderr = process.communicate()
	x,y,rho,a,b,c,d = np.loadtxt(execution_dir+'/Slice.txt').T
	newx=x.reshape((nboxsize,nboxsize)).T
	newy=y.reshape((nboxsize,nboxsize)).T
	newrho=rho.reshape((nboxsize,nboxsize)).T
	### Go back to inial directory
	return newx, newy, newrho



##### Power spectrum and correlation function realted formulae
def pk3d2xi(kin,pkin, padding=None):
	if padding is None:
		k=kin.copy()
		pk=pkin.copy()
	else:
		dkin = kin[1]-kin[0]
		k = linspace(0, padding*np.max(kin)+dkin, padding*len(kin))
		pk = np.zeros(len(k))
		pk[0:len(kin)]=pkin
	kmax = np.max(k)
	nk=len(k)
	r=2*np.pi/kmax*np.arange(nk)
	pkk=k*pk
	cric=-np.imag(np.fft.fft(pkk)/nk)/r/2/np.pi**2*kmax
	cric[0]=scipy.integrate.trapz(k*k*pk,x=k)/(2*np.pi**2)
	kinmax = np.max(kin)
	nkin = len(kin)
	rin = 2*np.pi/kinmax*np.arange(nkin)
	return r, cric

### Calcul du P1D directement a partir de P(k)
## On sait que l'on n'a le P1D qu'a une constante addititve pres qui correspond aux modes a grand k qui manquent dans l'integrale
def integ_p1d(k,pk, kmax=None):
	if kmax is None: kmax=np.max(k)
	ok = k <= kmax
	bla = np.append(0,scipy.integrate.cumtrapz(k[ok]*pk[ok]/(2*np.pi),x=k[ok]))
	return k[ok], bla[-1]-bla

### Une autre maniere de calculer P1D est avec la FFT de xi
def xi2pk1d(rin,xiin, padding = None, kout=None):
	## with padding and replication
	if padding is None:
		r=rin.copy()
		xi=np.append(xiin, xiin[:0:-1])
	else:
		drin = rin[1]-rin[0]
		r = linspace(0, padding*np.max(rin)+drin, padding*len(rin))
		xi = np.zeros(len(r))
		xi[0:len(rin)]=xiin
		xi = np.append(xi, xi[:0:-1])
	dr = r[1]-r[0]
	newpk = np.real(np.fft.fft(xi))*dr
	kvals = np.fft.fftfreq(len(xi),dr)*2*np.pi
	if kout is None:
		return kvals, newpk
	else:
		pkout = np.zeros(len(kout))
		pkout[kout < 0] = np.interp(kout[kout < 0], kvals[kvals < 0], newpk[kvals < 0])
		pkout[kout >= 0] = np.interp(kout[kout >= 0], kvals[kvals >= 0], newpk[kvals >= 0])
		return kout, pkout


def lnsim_1d(k, pk, xmax, nn,padding=None, check=False, seed=None, datasave=None):
	if datasave is None:
		### 0/ the final sampling and corresponding k
		xvals = np.linspace(0,xmax, nn)
		kvals = np.fft.fftfreq(nn,xvals[1]-xvals[0])*2*np.pi
		kmax = np.max(kvals)
		### 1/ input is camb P(k)
		### 2/ get xi(r) with massive zero-padding
		r, xi = pk3d2xi(k,pk*np.exp(-k**2/2/(kmax/2)**2), padding=padding)
		dr = r[1]-r[0]
		### 3/ C(r) = log(1+xi(r)/thenorm)
		thenorm = 0.5+np.sqrt(0.25+xi[0])
		thenorm=1
		cr = np.log(1+xi/thenorm)
		### 4/ P1D from C(r)
		kout, thepk1d = xi2pk1d(r, cr, padding=None, kout=kvals)
		thepk1d[thepk1d <0] =0
		datasave = [r, xi, cr, kvals, thepk1d]
	else:
		xvals = linspace(0,xmax, nn)
		cr= datasave[2]
		thepk1d = datasave[4]
	### 5/ exp(gaussian field)
	if seed: np.random.seed(seed)
	gaussnoise=np.random.randn(nn)
	ftgaussnoise = np.fft.fft(gaussnoise)
	dk = 2 * np.pi / xmax
	ftgaussnoisenew = ftgaussnoise * np.sqrt(thepk1d*dk/2/np.pi*nn)
	signal_noexp = np.real(np.fft.ifft(ftgaussnoisenew)) - cr[0]/2
	signal = np.exp(signal_noexp)
	#### if everything is fine, one should have var(signal)=xi[0] and var(signal_noexp)=cr[0]
	if check:
		print('Moyenne Noexp:  {0:8.4f}   (attendu: {1:8.4f})'.format(np.mean(signal_noexp), -cr[0]/2))
		print('Variance Noexp: {0:8.4f}   (attendu: {1:8.4f})'.format(np.var(signal_noexp), cr[0]))
		print('Moyenne LN:     {0:8.4f}   (attendu: {1:8.4f})'.format(np.mean(signal), 1))
		print('Variance LN:    {0:8.4f}   (attendu: {1:8.4f})'.format(np.var(signal), xi[0]))
	return xvals, signal, datasave

def build_pklib(params, zmin, zmax, nz, kmax=100, nk=2**17, log=False, extra_name=''):
	if log:
		z=np.append([0.], np.logspace(np.log10(0.1), np.log10(zmax), nz-1))
	else:
		z = np.linspace(zmin, zmax, nz)

	allpk = np.zeros((nz, nk))
	for i in xrange(nz):
		print('Running Camb at z={}   #{} out of {}'.format(z[i],i,nz))
		k, thepk = get_pk(params, z[i], kmax, nk)
		allpk[i,:] = thepk
	FitsArray(z, copy=False).save(extra_name+'pklib{}_z_{}_{}_zvalues.fits'.format(nz, zmin, zmax))
	FitsArray(k, copy=False).save(extra_name+'pklib{}_z_{}_{}_kvalues.fits'.format(nz, zmin, zmax))
	FitsArray(allpk, copy=False).save(extra_name+'pklib{}_z_{}_{}_pkvalues.fits'.format(nz, zmin, zmax))
	return [z, k, allpk]

def read_pklib(zmin, zmax, nz, extra_name=''):
	z = FitsArray(extra_name+'pklib{}_z_{}_{}_zvalues.fits'.format(nz, zmin, zmax))
	k = FitsArray(extra_name+'pklib{}_z_{}_{}_kvalues.fits'.format(nz, zmin, zmax))
	allpk = FitsArray(extra_name+'pklib{}_z_{}_{}_pkvalues.fits'.format(nz, zmin, zmax))
	return [z, k, allpk]

def bilinear_interpolate(imin, xim, yim, xout, yout):
	im=imin.T

	x = np.asarray((xout - np.min(xim)) / (np.max(xim) - np.min(xim)) * (len(xim)-1))
	y = np.asarray((yout - np.min(yim)) / (np.max(yim) - np.min(yim)) * (len(yim)-1))

	x0 = np.floor(x).astype(int)
	x1 = x0 + 1
	y0 = np.floor(y).astype(int)
	y1 = y0 + 1

	x0 = np.clip(x0, 0, im.shape[1]-1);
	x1 = np.clip(x1, 0, im.shape[1]-1);
	y0 = np.clip(y0, 0, im.shape[0]-1);
	y1 = np.clip(y1, 0, im.shape[0]-1);

	Ia = im[ y0, x0 ]
	Ib = im[ y1, x0 ]
	Ic = im[ y0, x1 ]
	Id = im[ y1, x1 ]

	wa = (x1-x) * (y1-y)
	wb = (x1-x) * (y-y0)
	wc = (x-x0) * (y1-y)
	wd = (x-x0) * (y-y0)

	return wa*Ia + wb*Ib + wc*Ic + wd*Id



class lightcone_1d:
	def __init__(self, maxd, nn, pklib=None, seed=None, omegam=0.3, smoothing=None):
		if pklib is None:
			pklib = read_pklib(0, 1100, 100)
		self.z = pklib[0]
		self.k = pklib[1]
		self.allpk = pklib[2]
		self.nz = len(self.z)
		self.nn = nn
		self.omegam = omegam

		self.alldelta = np.zeros((self.nz, nn))
		self.allvar = np.zeros(self.nz)
		for i in xrange(self.nz):
			print(i)
			xx, delta, bla = lnsim_1d(self.k,self.allpk[i,:],maxd,nn, seed=seed)
			if smoothing:
				delta = spf.gaussian_filter1d(delta, smoothing)
			self.alldelta[i,:] = delta
			self.allvar[i] = np.var(delta)
		self.xx = xx

	def __call__(self, zval, xval):
		return bilinear_interpolate(self.alldelta, self.z, self.xx, zval, xval)










