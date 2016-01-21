from pylab import *
import numpy as N
from scipy import integrate
from Cosmology import cosmology as cosmo
from Cosmology import inhodist
from Cosmology import LogNormal as ln


def profile(x,y,range=None,nbins=10,fmt=None,plot=True, dispersion=True, median=False):
  if range == None:
    mini = np.min(x)
    maxi = np.max(x)
  else:
    mini = range[0]
    maxi = range[1]
  dx = (maxi - mini) / nbins
  xmin = np.linspace(mini,maxi-dx,nbins)
  xmax = xmin + dx
  xc = xmin + dx / 2
  yval = np.zeros(nbins)
  dy = np.zeros(nbins)
  dx = np.zeros(nbins) + dx / 2
  for i in np.arange(nbins):
    ok = (x > xmin[i]) & (x < xmax[i])
    yval[i] = np.mean(y[ok])
    if median: yval[i]=np.median(y[ok])
    if dispersion: 
      fact = 1
    else:
      fact = np.sqrt(len(y[ok]))
    dy[i] = np.std(y[ok])/fact
  if plot: errorbar(xc, yval, xerr=dx, yerr=dy, fmt=fmt)
  return xc, yval, dx, dy


h=0.7
om_global = 0.3
dist_type = 'comoving_transverse'

nn=10000
xmax=5.
x=linspace(0,xmax,nn)


nperiods = 5
ampmod =0.3
om_sine = om_global+ampmod*np.sin(2*np.pi*x/xmax*nperiods)
om_sine[om_sine<0] = 0


om_uniform = np.zeros(nn)+om_global


clf()
plot(x,om_uniform)
plot(x,om_sine)

zrec_uniform = inhodist.x2z_inho(x, om_uniform, h,type=dist_type)
zrec_sine = inhodist.x2z_inho(x, om_sine, h,type=dist_type)

clf()
plot(zrec_uniform,x, lw=2, label='Uniform $\Omega_m={0:3.1f}$'.format(om_global))
plot(zrec_sine,x, lw=2, label='Sine modulation with average $\Omega_m={0:3.1f}$'.format(om_global))
xlabel('redshift')
ylabel(dist_type)
legend(loc='upper left',framealpha=0.8)

clf()
plot(zrec_uniform,x/x, lw=2, label='Uniform $\Omega_m={0:3.1f}$'.format(om_global))
plot(zrec_sine,x/np.interp(zrec_sine,zrec_uniform,x), lw=2, label='Sine modulation with average $\Omega_m={0:3.1f}$'.format(om_global))
xlabel('redshift')
ylabel('Distance ratio to Uniform')
legend(loc='upper left',framealpha=0.8)



import cosmolopy
import scipy.integrate
import numpy as np

nn=10000
zz = np.linspace(0,1100,nn)
### NB this formula will be wrong as it assumes Omega_lambda = 1 - omega_m
clf()
plot(zz, cosmolopy.perturbation.fgrowth(zz,0.3))
yscale('log')

fgz = cosmolopy.perturbation.fgrowth(zz,0.3)
fgz = fgz/fgz[-1]*1e-5
clf()
plot(zz, fgz)
yscale('log')

import scipy.integrate



def LogNormal(nnx,m,v, randomx=None):
	# can be related to mu and sigma of gaussian to be exponentialized by:
	# mu = ln( m^2 / sqrt(v + m^2))
	# sigma = sqrt( ln(1 + v/m^2) )
	# see : http://en.wikipedia.org/wiki/Log-normal_distribution
	# vérifié avec Mathematica
	mu = np.log(m**2 / np.sqrt(v + m**2))
	sigma = np.sqrt( np.log(1+v/m**2))
	if randomx is None:
		randomx=randn(nnx)
	return np.exp(sigma*randomx+mu)

def deriv(z, x, h, omega_av, randomxall, xall, fgz, zall, type):
	c=3e8
	H0 = 1000*1000*h*100
	xnew = x/(c/H0)
	growth = np.interp(z, zall,fgz)
	randx = np.interp(x, xall,randomxall)
	om = LogNormal(1,omega_av, growth**2, randomx=randx)
	print(z,x,om)
	ok = 1.-om
	if type=='comoving':
		fact=1
	elif type=='comoving_transverse':
		if ok==0:
			fact=1
		elif ok<0:
			fact=1./np.sqrt(1-ok*xnew**2)
		elif ok>0:
			fact = 1./np.sqrt(1+ok*xnew**2)
		else:
			stop
	else:
		return np.nan
	zprime = np.array(fact*(1 + z) * np.sqrt(1 + z * om))
	if isnan(zprime): stop
	return zprime/(c/H0)

def x2z_inho(x, omega_av, zall, fgz, h, type='comoving_transverse', randomxall=None):
	if randomxall is None:
		randomxall = randn(len(x))
	z0 = np.array([0.0])
	zrec = scipy.integrate.odeint(deriv, z0, x,args=(h, omega_av, randomxall, x, fgz, zall, type) )
	zrec = zrec[:,0]
	gr = np.interp(zrec, zall, fgz)
	om = LogNormal(len(x), omega_av, gr**2, randomx=randomxall)
	return zrec, gr, np.array(om), randomxall


scale = 1 #Mpc
xmax = 5 #Gpc
nnx= xmax*1000/scale
xx = np.linspace(0,xmax,nnx) ### in Gpc

amp_uniform = 0
zrec_uniform, gr_uniform, om_uniform, rndx = x2z_inho(xx, om_global, zz,fgz*amp_uniform, h)

# relation de Jim: Delta_rho/rho[echelle R] = DeltaPhi * (c/H0)^2/R^2
c_H0 = (3e5/h/100)   #Mpc
amp_inho = c_H0**2/scale**2

nbmean = 10
allzrec_inho = np.zeros((nnx,nbmean))
allgr_inho = np.zeros((nnx,nbmean))
allom_inho = np.zeros((nnx, nbmean))
for i in xrange(nbmean):
	print(i,nbmean)
	allzrec_inho[:,i], allgr_inho[:,i], allom_inho[:,i], rndx = x2z_inho(xx, om_global, zz,fgz*amp_inho, h)

zrec_inho = np.mean(allzrec_inho, axis=1)
dzrec_inho = np.std(allzrec_inho, axis=1)
gr_inho = np.mean(allgr_inho, axis=1)
om_inho = allom_inho[:,0]
mom_inho = np.mean(allom_inho, axis=1)
som_inho = np.std(allom_inho, axis=1)
mlogminho = np.mean(np.log(allom_inho),axis=1)
slogminho = np.std(np.log(allom_inho),axis=1)



import cosmolopy.distance as cd
zvals = np.linspace(0,np.max(zrec_inho),1000)
lcdm = cd.set_omega_k_0({'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.7})
d_lcdm = cd.comoving_distance_transverse(zvals, **lcdm)

open = cd.set_omega_k_0({'omega_M_0' : 0.3, 'omega_lambda_0' : 0., 'h' : 0.7})
d_open = cd.comoving_distance_transverse(zvals, **open)

newinhox = np.interp(zvals, zrec_inho, xx)
newinhoxplus = np.interp(zvals, zrec_inho+dzrec_inho, xx)
newinhoxmoins = np.interp(zvals, zrec_inho-dzrec_inho, xx)


clf()
subplot(2,1,1)
plot(zrec_uniform,xx,'b',lw=2,label='Uniform JC om=0.3 open')
plot(zrec_inho,xx,'r',lw=2,label='Inho JC')
fill_between(zvals, newinhoxplus, y2=newinhoxmoins, color='red',alpha=0.2)
plot(zvals, d_lcdm/1000,'g', lw=2, label='LCDM Cosmolopy')
plot(zvals, d_open/1000,'m--', lw=2, label='Open om=0.3 Cosmolopy')
legend(loc='upper left', fontsize=10)
xlim(zrec_inho[1],np.max(zrec_inho))
ylim(xx[1],xmax)
ylabel('Comoving Transverse Distance')
xlabel('z')
subplot(2,2,3)
plot(zrec_inho,xx*0,'b',lw=2)
fill_between(zvals, (newinhoxplus*1000/d_open-1)*100, y2=(newinhoxmoins*1000/d_open-1)*100,color='red',alpha=0.2)
plot(zvals, (newinhox*1000/d_open-1)*100, 'r',lw=2)
plot(zvals, 100*(d_lcdm/d_open-1),'g',lw=2)
xlim(0,np.max(zrec_inho))
xlabel('z')
ylabel('Relative difference (%)')
subplot(2,2,4)
for i in xrange(nbmean):
	plot(allzrec_inho[:,i], allom_inho[:,i],'k,')
plot(zrec_inho,mom_inho,'r',lw=2)
plot(zrec_uniform,om_uniform,'b',lw=2)
yscale('log')
xlim(0,np.max(zrec_inho))
xlabel('z')
ylabel('Omega')


# mean and variance of Lognormal distribution
clf()
subplot(2,1,1)
plot(zrec_inho, om_inho,',')
xc,yc,dx,dy = profile(zrec_inho,om_inho,nbins=10,fmt='ro') 
subplot(2,2,3)
errorbar(xc,yc,yerr=dy)
plot(zrec_inho,zrec_inho*0+om_global)
subplot(2,2,4)
plot(xc,dy,'ro')
plot(zrec_inho,gr_inho)
yscale('log')









