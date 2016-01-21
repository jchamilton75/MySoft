from Cosmology import pyxi_old
from Cosmology import pyxi
import cosmolopy
from matplotlib import cm
from scipy import interpolate
import scipy.integrate
import pycamb
from Cosmology import LogNormal as ln
import cosmolopy.distance as cd
from pysimulators import FitsArray
from matplotlib import cm



################# Input Power spectrum ###################################
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
r = 0.05
h=0.7
H0 = h*100
omegab = 0.022032
omegac = 0.12038
h2 = (H0/100.)**2
Omegab = omegab/h2
Omegac = omegac/h2
Omegam = (omegac+omegab)/h2
#Omegav = 1. - Omegam
Omegav = 0.
Omegak = 1. - Omegam - Omegav

print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',Omegav, 'Omegak = ', Omegak

params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':Omegak,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925}


###### Get input P(k)
z=0
k, pk = ln.get_pk(params, z, kmax=100, nk=2**20)
pkinit = pk.copy()
pk *= exp(-k**2/2/1**2)

r,xi = ln.pk3d2xi(k,pk)
r,xi_init = ln.pk3d2xi(k, pkinit)
clf()
subplot(2,1,1)
plot(k,pk)
plot(k,pkinit)
ylim(1e-5, 1e5)
yscale('log')
xscale('log')
subplot(2,1,2)
plot(r,xi*r**2)
plot(r,xi_init*r**2)
xlim(0, 500)
ylim(-100,100)


#### Simple LogNormal Skewer at constant redshift
xmax = 20000.
nn=1000
xvals, signal, bla = ln.lnsim_1d(k,pk,xmax,nn, padding=None, check=True)
clf()
plot(xvals,signal)

nbtry = floor(np.logspace(3,6,20))
nbmc = 30
allmeans = np.zeros(len(nbtry))
allvars = np.zeros(len(nbtry))
allerrmeans = np.zeros(len(nbtry))
allerrvars = np.zeros(len(nbtry))
for i in xrange(len(nbtry)):
	print(i,nbtry[i])
	themeans = np.zeros(nbmc)
	thevars = np.zeros(nbmc)
	for j in xrange(nbmc):
		xvals, signal, bla = ln.lnsim_1d(k,pk,xmax,int(nbtry[i]), padding=None)
		themeans[j] = np.mean(signal)
		thevars[j] = np.var(signal)
	allmeans[i] = np.mean(themeans)
	allerrmeans[i] = np.std(themeans)
	allvars[i] = np.mean(thevars)
	allerrvars[i] = np.std(thevars)

clf()
subplot(2,1,1)
errorbar(nbtry, allmeans, yerr=allerrmeans)
xscale('log')
subplot(2,1,2)
errorbar(nbtry, allvars, yerr=allerrvars)
xscale('log')

#### P(K) library at various redshifts 
nz = 100
zmin = 0
zmax = 1100
# BUILD IT ###############################################################################
#pklib = ln.build_pklib(params, zmin, zmax, nz)
# RESTORE IT ##############################################################################
pklib = ln.read_pklib(zmin, zmax, nz)

#### Skewers library with the same seed at different reshifts
# set the x range so that it extends well beyond the distance at maximum redshift
zvals = np.linspace(0,zmax,1000)
lcdm = cd.set_omega_k_0({'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.7})
d_lcdm = cd.comoving_distance_transverse(zvals, **lcdm)
maxd = np.max(d_lcdm)*1.2

theseed = 1
nn = 2**18
lightcone = ln.lightcone_1d(maxd, nn, pklib, seed=theseed)


thezinit = linspace(0,zmax, 10000)
dinit = cd.comoving_distance_transverse(thezinit, **lcdm)

nbvals = 2**19
zz = linspace(0, zmax*0.99, nbvals)
dd = np.interp(zz, thezinit, dinit)


fg = cosmolopy.perturbation.fgrowth(lightcone.z, 0.3)
clf()
plot(lightcone.z, lightcone.allvar)
plot(lightcone.z, fg**2*lightcone.allvar[0]/(fg[0]**2))
yscale('log')
xscale('log')

clf()
plot(dd, lightcone(zz, dd))
yscale('log')

clf()
plot(zz, lightcone(zz, dd))
yscale('log')







