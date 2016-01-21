from pylab import *
import numpy as N
from scipy import integrate
import cosmology as cosmo
reload(cosmo)


z=linspace(0,2,1000)
omegam=0.3
omegax=0.7
w0=-1
w1=0.

h=0.7

params=[omegam,omegax,w0,w1]

clf()
xlabel('redshift')
ylabel('distance (Gpc/h)')
plot(z,cosmo.get_dist(z,type='prop',params=params))
plot(z,cosmo.get_dist(z,type='dl',params=params))
plot(z,cosmo.get_dist(z,type='dang',params=params))
plot(z,cosmo.get_dist(z,type='hz',params=params)/1e6/100)
legend( ('Proper distance', 'Luminosity distance', 'Angular distance', 'h(z)=H(z)/100') )


clf()
xlabel('redshift')
ylabel('distance')
plot z,cosmo.get_dist(z,type='dangco',params=params)
plot z,cosmo.get_dist(z,type='vco',params=params)
plot z,cosmo.get_dist(z,type='rapp',params=params)
legend( ('Comoving angular distance', 'Comoving volume', 'Ratio for AP test') )

clf()
xlabel('redshift')
ylabel('distance')
plot z,cosmo.get_dist(z,type='wz',params=params)
plot z,cosmo.get_dist(z,type='omegaxz',params=params)
ylim(-2,2)
legend( ('equation of state of DE', 'OmegaX(z)') )





#####
clf()
plot(z,cosmo.get_dist(z,type='dl',params=[0.3,0.7,-1,0]))
plot(z,cosmo.get_dist(z,type='dl',params=[0.3,0,-1,0]))




#####
from scipy.ndimage import gaussian_filter1d


##### Not good because does not tend to omegam=0.3 at high z - but shows the expected effect
#omega_av = 0.3
#amp = 10
#nb = 100000
#z=linspace(0,1,nb)
#omegamzcst = np.zeros(nb)+omega_av
#dlcst = cosmo.get_dist(z,type='dl',params=[omegamzcst,0,-1,0])
#theamp = amp-z/np.max(z)*amp
#omegamzsin = np.exp(theamp*np.sin(z/np.max(z)*2*np.pi*100))
#omegamzsin = omegamzsin/np.mean(omegamzsin)*omega_av
#clf()
#plot(z, omegamzsin)
#xlim(0,0.1)
#dlsin = cosmo.get_dist(z,type='dl',params=[omegamzsin,0,-1,0])
##dlsin_sm = gaussian_filter1d(dlsin,nb/100, mode='nearest')

def profile(x,y,range=None,nbins=10,fmt=None,plot=True, dispersion=True):
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
    if dispersion: 
      fact = 1
    else:
      fact = np.sqrt(len(y[ok]))
    dy[i] = np.std(y[ok])/fact
  if plot: errorbar(xc, yval, xerr=dx, yerr=dy, fmt=fmt)
  return xc, yval, dx, dy



omega_av = 0.8
amp = 100
nb = 100000
nsm = 100
z=linspace(0,100,nb)

### test
omegamzcst = np.zeros(nb)+omega_av
dlcst = cosmo.get_dist(z,type='dl',params=[omegamzcst,0,-1,0])
clf()
plot(z,cosmo.get_dist(z,type='dl',params=[omega_av,0,-1,0]))
plot(z,dlcst,'r--',lw=3)

### lognormal with mean=omega_av and variable amplitude
import cosmolopy
# mean and variance of Lognormal distribution are m and v
m = np.zeros(nb) + omega_av
v = (amp * cosmolopy.perturbation.fgrowth(z, omega_av))**2
# can be related to mu and sigma of gaussian to be exponentialized by:
# mu = ln( m^2 / sqrt(v + m^2))
# sigma = sqrt( ln(1 + v/m^2) )
# see : http://en.wikipedia.org/wiki/Log-normal_distribution
# vérifié avec Mathematica
mu = np.log(m**2 / np.sqrt(v + m**2))
sigma = np.sqrt( np.log(1+v/m**2))
omegamzsin = np.exp(sigma*randn(nb)+mu)

clf()
subplot(2,1,1)
plot(z, omegamzsin,',')
xc,yc,dx,dy = profile(z,omegamzsin,nbins=100,fmt='ro') 
subplot(2,2,3)
errorbar(xc,yc,yerr=dy)
plot(z,m)
subplot(2,2,4)
plot(xc,dy,'ro')
plot(z,np.sqrt(v))


dlsin = cosmo.get_dist(z,type='dl',params=[omegamzsin,0,-1,0])
dlsin_sm = gaussian_filter1d(dlsin,nsm, mode='nearest')

dasin = cosmo.get_dist(z,type='dang',params=[omegamzsin,0,-1,0])
dasin_sm = gaussian_filter1d(dasin,nsm, mode='nearest')




clf() 

subplot(2,2,1)
plot(z,cosmo.get_dist(z,type='dl',params=[0.3,0.7,-1,0]),'m',lw=2, label = '$\Lambda CDM$')
plot(z, cosmo.get_dist(z,type='dl',params=[0.3,0.,-1,0]),'b',lw=2, label = '$\Omega_m=0.3,\, \Omega_\Lambda=0$')
plot(z, cosmo.get_dist(z,type='dl',params=[1,0.,-1,0]),'k',lw=2, label = '$\Omega_m=1,\, \Omega_\Lambda=0$')
plot(z, cosmo.get_dist(z,type='dl',params=[omega_av,0.,-1,0]),'y--',lw=2, label = '$\Omega_m={0:5.2f},\, \Omega_\Lambda=0$'.format(omega_av))
#plot(z,dlsin,'r--', lw=2, label = 'Inhomogeneous')
plot(z,gaussian_filter1d(dlsin_sm,100, mode='nearest'),'g--',lw=2, label = 'Inhomogeneous $\Omega = {0:5.2f}$'.format(omega_av))
legend(loc='upper left', fontsize='x-small')
ylabel('$D_L(z)$')
xlabel('z')

subplot(2,2,2)
plot(z, cosmo.get_dist(z,type='dl',params=[0.3,0.,-1,0]) / cosmo.get_dist(z,type='dl',params=[0.3,0.7,-1,0]),'b',lw=2)
plot(z, cosmo.get_dist(z,type='dl',params=[0.3,0.7,-1,0]) / cosmo.get_dist(z,type='dl',params=[0.3,0.7,-1,0]),'m',lw=2)
plot(z, cosmo.get_dist(z,type='dl',params=[omega_av,0.,-1,0]) / cosmo.get_dist(z,type='dl',params=[0.3,0.7,-1,0]),'y--',lw=2)
plot(z, dlsin_sm / cosmo.get_dist(z,type='dl',params=[0.3,0.7,-1,0]),'g--',lw=2)
ylabel('$D_L(z) / D^{\Lambda CDM}_L(z)$')
xlabel('z')
ylim([0,2])


subplot(2,2,3)
plot(z,cosmo.get_dist(z,type='dang',params=[0.3,0.7,-1,0]),'m',lw=2, label = '$\Lambda CDM$')
plot(z, cosmo.get_dist(z,type='dang',params=[0.3,0.,-1,0]),'b',lw=2, label = '$\Omega_m=0.3,\, \Omega_\Lambda=0$')
plot(z, cosmo.get_dist(z,type='dang',params=[1,0.,-1,0]),'k',lw=2, label = '$\Omega_m=1,\, \Omega_\Lambda=0$')
plot(z, cosmo.get_dist(z,type='dang',params=[omega_av,0.,-1,0]),'y--',lw=2, label = '$\Omega_m={0:5.2f},\, \Omega_\Lambda=0$'.format(omega_av))
#plot(z,dasin,'r--', lw=2, label = 'Inhomogeneous')
plot(z,gaussian_filter1d(dasin_sm,100, mode='nearest'),'g--',lw=2, label = 'Inhomogeneous $\Omega = {0:5.2f}$'.format(omega_av))
legend(loc='bottom right', fontsize='x-small')
ylabel('$D_A(z)$')
xlabel('z')

subplot(2,2,4)
plot(z, cosmo.get_dist(z,type='dang',params=[0.3,0.,-1,0]) / cosmo.get_dist(z,type='dang',params=[0.3,0.7,-1,0]),'b',lw=2)
plot(z, cosmo.get_dist(z,type='dang',params=[0.3,0.7,-1,0]) / cosmo.get_dist(z,type='dang',params=[0.3,0.7,-1,0]),'m',lw=2)
plot(z, cosmo.get_dist(z,type='dang',params=[omega_av,0.,-1,0]) / cosmo.get_dist(z,type='dang',params=[0.3,0.7,-1,0]),'y--',lw=2)
plot(z, dasin_sm / cosmo.get_dist(z,type='dang',params=[0.3,0.7,-1,0]),'g--',lw=2)
ylabel('$D_A(z) / D^{\Lambda CDM}_A(z)$')
xlabel('z')
ylim([0,2])
xscale('log')
xlim(10*nsm*z[1],np.max(z))




