from pylab import *
from scipy import integrate
from Cosmology import cosmology as cosmo
from Cosmology import inhodist
from Cosmology import LogNormal as ln
import cosmolopy.distance as cd
from matplotlib import cm

################# Input Power spectrum ###################################
###### With Lambda = 0
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


nz = 100
zmin = 0
zmax = 1100
# BUILD IT ###############################################################################
#pklib = ln.build_pklib(params, zmin, zmax, nz, extra_name='nolambda_',log=True)
# RESTORE IT ##############################################################################
pklib = ln.read_pklib(zmin, zmax, nz, extra_name = 'nolambda_')








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




xmax = 4000 #Mpc
nnx= 3000

theseed=3
lightcone_cst = ln.lightcone_1d(xmax, nnx, seed=theseed, pklib=pklib, omegam=(omegac+omegab)/h2)
lightcone_cst.alldelta = lightcone_cst.alldelta * 0 + 1

clf()
plot(lightcone_cst.xx, lightcone_cst(0, lightcone_cst.xx))
ylim(0,2)

zrec_cst = inhodist.x2z_inho_lightcone(lightcone_cst, h)

theseed=None
lightcone = ln.lightcone_1d(xmax, nnx, seed=theseed, pklib=pklib, omegam=(omegac+omegab)/h2, smoothing=None)
clf()
plot(lightcone.xx, lightcone.alldelta[0,:])
np.mean(lightcone.alldelta[0,:])



zrec = inhodist.x2z_inho_lightcone(lightcone, h)


zvals = np.linspace(0,np.max(zrec_cst),1000)
open = cd.set_omega_k_0({'omega_M_0' : (omegac+omegab)/h2, 'omega_lambda_0' : 0., 'h' : 0.7})
d_open = cd.comoving_distance_transverse(zvals, **open)
lcdm = cd.set_omega_k_0({'omega_M_0' : (omegac+omegab)/h2, 'omega_lambda_0' : 0.7, 'h' : 0.7})
d_lcdm = cd.comoving_distance_transverse(zvals, **lcdm)
empty = cd.set_omega_k_0({'omega_M_0' : 0, 'omega_lambda_0' : 0., 'h' : 0.7})
d_empty = cd.comoving_distance_transverse(zvals, **empty)


clf()
plot(zvals, d_open/d_lcdm, lw=2, label='Standard: Open $\Omega_m=0.3$')
plot(zvals, d_lcdm/d_lcdm, lw=2, label='Standard: $\Lambda$CDM')
plot(zvals, d_empty/d_lcdm, lw=2, label='Standard: Empty $\Omega_m=0.$')
plot(zrec_cst, lightcone_cst.xx/np.interp(zrec_cst, zvals, d_lcdm), '--', lw=2, label='JC: Uniform Open $\Omega_m=0.3$')
plot(zrec, lightcone.xx/np.interp(zrec, zvals, d_lcdm), lw=2, label='JC: Inhomogeneous Open $\Omega_m=0.3$')
legend(loc='upper left', fontsize=10, frameon=False)





xmax = 4000 #mpc
allnnx = np.floor(np.logspace(2, 6, 10))
allzrec = []
for i in xrange(len(allnnx)):
	theseed=None
	lightcone = ln.lightcone_1d(xmax, int(allnnx[i]), seed=theseed, pklib=pklib, omegam=(omegac+omegab)/h2, smoothing=None)
	allzrec.append(inhodist.x2z_inho_lightcone(lightcone, h))

clf()
plot(zvals, d_open, lw=4, label='Standard: Open $\Omega_m=0.3$')
plot(zvals, d_lcdm, lw=4, label='Standard: $\Lambda$CDM')
plot(zvals, d_empty, lw=4, label='Standard: Empty $\Omega_m=0.$')
plot(zrec_cst, lightcone_cst.xx, '--', lw=2, label='JC: Uniform Open $\Omega_m=0.3$')
for i in xrange(len(allnnx)):
	xx = np.linspace(0,xmax,allnnx[i])
	plot(allzrec[i], xx, color=cm.jet(i*1./len(omvals)), label='pixel size = {}'.format(xmax/allnnx[i]))
legend(loc='upper left',fontsize=8, frameon=False)

clf()
plot(zvals, d_open/d_lcdm, lw=4, label='Standard: Open $\Omega_m=0.3$')
plot(zvals, d_lcdm/d_lcdm, lw=4, label='Standard: $\Lambda$CDM')
plot(zvals, d_empty/d_lcdm, lw=4, label='Standard: Empty $\Omega_m=0.$')
plot(zrec_cst, lightcone_cst.xx/np.interp(zrec_cst, zvals, d_lcdm), '--', lw=2, label='JC: Uniform Open $\Omega_m=0.3$')
for i in xrange(len(allnnx)):
	xx = np.linspace(0,xmax,allnnx[i])
	plot(allzrec[i], xx/np.interp(allzrec[i], zvals, d_lcdm), color=cm.jet(i*1./len(omvals)), label='pixel size = {}'.format(xmax/allnnx[i]))
legend(loc='lower left',fontsize=8, frameon=False)

savefig('dist_inho_pixsize.png')



######################## Varying Omega (but not changing the P(k) so rathe rincorrect...)
nnx=3000
omvals = np.linspace(0,1.,11)
allzrec = np.zeros((len(omvals), nnx))

for i in xrange(len(omvals)):
	theseed=None
	lightcone = ln.lightcone_1d(xmax, nnx, seed=theseed, pklib=pklib, omegam=omvals[i], smoothing=None)
	allzrec[i,:] = inhodist.x2z_inho_lightcone(lightcone, h)

clf()
plot(zvals, d_open, lw=4, label='Standard: Open $\Omega_m=0.3$')
plot(zvals, d_lcdm, lw=4, label='Standard: $\Lambda$CDM')
plot(zvals, d_empty, lw=4, label='Standard: Empty $\Omega_m=0.$')
plot(zrec_cst, lightcone_cst.xx, '--', lw=2, label='JC: Uniform Open $\Omega_m=0.3$')
for i in xrange(len(omvals)):
	plot(allzrec[i,:], lightcone.xx, color=cm.jet(i*1./len(omvals)), label='om={}'.format(omvals[i]))
legend(loc='upper left',fontsize=8)


clf()
plot(zvals, d_open/d_lcdm, lw=4, label='Standard: Open $\Omega_m=0.3$')
plot(zvals, d_lcdm/d_lcdm, lw=4, label='Standard: $\Lambda$CDM')
plot(zvals, d_empty/d_lcdm, lw=4, label='Standard: Empty $\Omega_m=0.$')
plot(zrec_cst, lightcone_cst.xx/np.interp(zrec_cst, zvals, d_lcdm), '--', lw=2, label='JC: Uniform Open $\Omega_m=0.3$')
for i in xrange(len(omvals)):
	plot(allzrec[i,:], lightcone.xx/np.interp(allzrec[i,:], zvals, d_lcdm), color=cm.jet(i*1./len(omvals)), label='om={}'.format(omvals[i]))
legend(loc='lower left',fontsize=8, frameon=False)
##################################################################################
















