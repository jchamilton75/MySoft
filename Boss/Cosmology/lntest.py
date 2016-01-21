from Cosmology import pyxi_old
from Cosmology import pyxi
import cosmolopy
from matplotlib import cm
from scipy import interpolate
import scipy.integrate
import pycamb


#### Get input Power spectra
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


def statstr(vec,divide=False):
	m=np.mean(vec)
	s=np.std(vec)
	if divide: s/=sqrt(len(vec))
	return '{0:.4f} +/- {1:.4f}'.format(m,s)



def get_pk(cosmo,z, kmax=10, nk=32768):
	kvals = np.linspace(0,kmax,nk)
	pk=pycamb.matter_power(redshifts=[z], k=kvals, **cosmo)[1]
	pk[0]=0
	return kvals, pk


################# LN Slice with Nico Code

import sys
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

nboxsize=128l
boxsize=1000

z=0
k1, pk1 = get_pk(params, z)
xx1, yy1, rhonorm1 = LogNormalSlice(k1,pk1,nboxsize=nboxsize, boxsize=boxsize)

z=1
k2, pk2 = get_pk(params, z)
xx2, yy2, rhonorm2 = LogNormalSlice(k2,pk2,nboxsize=nboxsize, boxsize=boxsize)

clf()
plot(k1,pk1)
plot(k2,pk2)
xscale('log')
yscale('log')


clf()
subplot(2,2,1)
imshow(rhonorm1, interpolation='nearest')
colorbar()
subplot(2,2,2)
imshow(rhonorm2, interpolation='nearest')
colorbar()
subplot(2,2,3)
imshow(rhonorm1-rhonorm2, interpolation='nearest')
colorbar()
subplot(2,2,4)
imshow(rhonorm2/rhonorm1, interpolation='nearest')
colorbar()

#####################################




########## We want to do it in 1D directly


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
	cric=-np.imag(fft(pkk)/nk)/r/2/np.pi**2*kmax
	cric[0]=scipy.integrate.trapz(k*k*pk,x=k)/(2*pi**2)
	kinmax = np.max(kin)
	nkin = len(kin)
	rin = 2*np.pi/kinmax*np.arange(nkin)
	return r, cric

def pk3d2xi_old(k,pk):
	kmax = np.max(k)
	nk=len(k)
	r=2*np.pi/kmax*np.arange(nk)
	pkk=k*pk
	cric=-np.imag(fft(pkk)/nk)/r/2/np.pi**2*kmax
	cric[0]=scipy.integrate.trapz(k*k*pk,x=k)/(2*pi**2)
	return r, cric


z=1
k, pk = get_pk(params, z, kmax=1000, nk=2**17)
r, xi = pk3d2xi(k,pk)

clf()
subplot(2,1,1)
plot(k,pk, 'red')
xscale('log')
yscale('log')
legend()
subplot(2,1,2)
plot(r, xi*r**2,'red')
ylim(-10,100)
xlim(0,300)

### Calcul du P1D directement à partir de P(k)
## On sait que l'on n'a le P1D qu'a une constante addititve près qui correspond aux modes à grand k qui manquent dans l'integrale
def integ_p1d(k,pk, kmax=None):
	if kmax is None: kmax=np.max(k)
	ok = k <= kmax
	bla = np.append(0,scipy.integrate.cumtrapz(k[ok]*pk[ok]/(2*np.pi),x=k[ok]))
	return k[ok], bla[-1]-bla

bla, pk1d = integ_p1d(k,pk)

clf()
plot(k,pk)
plot(k,pk1d)
xscale('log')
yscale('log')


### Une autre manière de calculer P1D est avec la FFT de xi
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
	newpk = real(np.fft.fft(xi))*dr
	kvals = np.fft.fftfreq(len(xi),dr)*2*pi
	if kout is None:
		return kvals, newpk
	else:
		pkout = np.zeros(len(kout))
		pkout[kout < 0] = np.interp(kout[kout < 0], kvals[kvals < 0], newpk[kvals < 0])
		pkout[kout >= 0] = np.interp(kout[kout >= 0], kvals[kvals >= 0], newpk[kvals >= 0])
		return kout, pkout

pad=100
rpad, xipad = pk3d2xi(k,pk, padding=pad)
kvals,newpk1d = xi2pk1d(rpad, xipad,padding=None, kout=k)

clf()
xscale('log')
yscale('log')
plot(k,pk,'k',label='$P_{3d}(k)$')
plot(k,pk1d,'g',label='Direct $P_{1d}(k)$')
plot(kvals, newpk1d,'r',label='$P_{1d}(k)$ from xi(r)')
legend(loc='lower left', fontsize=10)



################### C'est la que ça se passe mec ####################################################
#### On veut faire une simu LN 1D => on a besoin du spectre 1D

z=0
k, pk = get_pk(params, z, kmax=100, nk=2**17)
pk *= exp(-k**2/2/10**2)
import scipy.ndimage

xmax = 1600
nn=2**20
r, xi = pk3d2xi(k,pk, padding=100)
dr = r[1]-r[0]
x = 0.5+np.sqrt(0.25+xi[0])
cr = log(1+xi/x)
clf()
plot(r,xi*r**2,label='Xi(r)')
plot(r,cr*r**2,label='C(r)')
legend()

### 3) calcul de pk1d du champ gaussien
xvals = linspace(0,xmax, nn)
kvals = np.fft.fftfreq(nn,xvals[1]-xvals[0])*2*pi
kout, thepk1d = xi2pk1d(r, cr,padding=None, kout=kvals)
thepk1d[thepk1d <0] =0

### 4) génération des variables aléatoires
#4a il faut que la gaussienne reelle soit de largeur sqrt(n) pour que les modes de Fourier soient de largeur 1
## Old JC
gaussnoise=np.random.randn(nn)
ftgaussnoise = fft(gaussnoise)
mean(abs(ftgaussnoise)**2)
dk = kvals[1]-kvals[0]
ftgaussnoisenew = ftgaussnoise * np.sqrt(thepk1d*dk/2/pi*nn)


plus = kvals > 0
clf()
plot(kvals[plus], scipy.ndimage.filters.gaussian_filter1d(np.abs(ftgaussnoise[plus])**2,100))
plot(kvals[plus], kvals[plus]*0+1,'r--')
xscale('log')
yscale('log')

#4b scaling par le spectre de puissance
#old
#ftgaussnoisenew = ftgaussnoise * np.sqrt(thepk1d)
clf()
plot(kvals[plus], scipy.ndimage.filters.gaussian_filter1d(np.abs(ftgaussnoisenew[plus])**2,100))
plot(kvals[plus], thepk1d[plus]*dk/2/pi*nn*nn,'r--')
xscale('log')
yscale('log')

#4c retour en arrière vers l'espace réel
# attention aux normalisations selon que l'on utilise une fft ou une ifft
#np.std(real(fft(ftgaussnoise)/sqrt(nn)))
#np.std(real(ifft(ftgaussnoise)*sqrt(nn)))

### ici on voit deja un probleme: la variance de signal_noexp devrait être cr(0)
signal_noexp = np.real(ifft(ftgaussnoisenew))
print(var(signal_noexp), cr[0], var(signal_noexp)/cr[0])


signal = np.exp(signal_noexp)
print(var(signal), xi[0])



def lnsim_1d(k, pk, xmax, nn,padding=None, check=False, seed=None, datasave=None):
	if datasave is None:
		### 1/ input is camb P(k)
		### 2/ get xi(r) with massive zero-padding
		r, xi = pk3d2xi(k,pk, padding=padding)
		dr = r[1]-r[0]
		### 3/ C(r) = log(1+xi(r)/thenorm)
		thenorm = 0.5+np.sqrt(0.25+xi[0])
		thenorm=1
		cr = log(1+xi/thenorm)
		### 4/ the final sampling and corresponding k
		xvals = linspace(0,xmax, nn)
		kvals = np.fft.fftfreq(nn,xvals[1]-xvals[0])*2*pi
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
	ftgaussnoise = fft(gaussnoise)
	dk = 2 * pi / xmax
	ftgaussnoisenew = ftgaussnoise * np.sqrt(thepk1d*dk/2/pi*nn)
	signal_noexp = np.real(ifft(ftgaussnoisenew)) - cr[0]/2
	signal = np.exp(signal_noexp)
	#### if everything is fine, one should have var(signal)=xi[0] and var(signal_noexp)=cr[0]
	if check:
		print('Moyenne Noexp:  {0:8.4f}   (attendu: {1:8.4f})'.format(mean(signal_noexp), -cr[0]/2))
		print('Variance Noexp: {0:8.4f}   (attendu: {1:8.4f})'.format(var(signal_noexp), cr[0]))
		print('Moyenne LN:     {0:8.4f}   (attendu: {1:8.4f})'.format(mean(signal), 1))
		print('Variance LN:    {0:8.4f}   (attendu: {1:8.4f})'.format(var(signal), xi[0]))
	return xvals, signal, datasave


#### Calculate xi
r, xi = pk3d2xi(k,pk, padding=None)

#### with 1D
xmax = 2000.
nn=2**17
xvals, signal, bla = lnsim_1d(k,pk,xmax,nn, padding=None, check=True)
mean(signal)

#### make a lot
xmax = 40000.
nn=2**20
nb=20000
xv,lny, datasave = lnsim_1d(k,pk,xmax,nn)
means = np.zeros(nb)
variances= np.zeros(nb)
corrfunct= np.zeros(nn)
for i in xrange(nb):
	print(i)
	xv,lny, bla = lnsim_1d(k,pk,xmax,nn, datasave=datasave)
	means[i]=mean(lny)
	variances[i]=var(lny)
	cfi = real(ifft(abs(fft(lny-mean(lny)))**2))
	corrfunct += cfi/nb


clf()
subplot(3,1,1)
hist(means, bins=30, label=statstr(means,divide=True)+' expected: 1')
xlabel('Mean')
legend()
subplot(3,1,2)
hist(variances, bins=30, range=[0,3*xi[0]], label=statstr(variances,divide=True)+' expected: {0:8.4f}'.format(xi[0]))
xlabel('Variance')
legend()
subplot(3,1,3)
thermax=200
w=xv<thermax
wr = r < thermax
plot(xv[w], scipy.ndimage.filters.gaussian_filter1d(corrfunct[w]/nn*xv[w]**2,0.1))
plot(r[wr],xi[wr]*r[wr]**2)
xlim(0,max(xv[w]))
ylim(-10,50)
xlabel('2pt correlation function')

xmax = 2000.
nn=256
truc=np.zeros((nn,nn))
allxi = np.zeros((nn,nn))
allxi2 = np.zeros((nn,nn))
for i in xrange(nn):
	print(i)
	bla, truc[:,i] = lnsim_1d(k,pk,xmax,nn, padding=None)

clf()
hist(np.ravel(truc), bins=100, range=[0,5], color='red', alpha=0.3, label='1D: '+statstr(np.ravel(truc)))
hist(np.ravel(rhonorm), bins=100, range=[0,5], color='blue', alpha=0.3, label='3D: '+statstr(np.ravel(rhonorm)))
legend()


#### With 3D box slice
z=0
xmax = 2000.
nn=256
xx, yy, rhonorm = LogNormalSlice(k,pk,nboxsize=nn, boxsize=int(xmax/2))

clf()
hist(np.ravel(truc), bins=100, range=[0,5], color='red', alpha=0.3, label=statstr(np.ravel(truc)))
hist(np.ravel(rhonorm), bins=100, range=[0,5], color='blue', alpha=0.3, label=statstr(np.ravel(rhonorm)))
legend()
















