from pylab import *
import numpy as N
from scipy import integrate
from Cosmology import cosmology as cosmo
from Cosmology import inhodist



zmax = 2
### ces distances sont en Mpc
dist_type = 'comoving_transverse'
x0 = 0
xs = 1.5
xe = 2.5
x1 = 3.5


h=0.7
om_global = 0.3
om_region_vide = 0.
om_region_dense = ((x1-x0)*om_global - (xe-xs)*om_region_vide) / ((xs-x0) + (x1-xe))
params_region_vide = [om_region_vide,0,-1,0]
params_region_dense = [om_region_dense,0,-1,0]
params_global = [om_global, 0, -1, 0]
nn = 1000
c=3e8
H0=1000*1000*h*100

zz = linspace(0,zmax, nn)

hz_region_vide = cosmo.get_dist(zz,type='hz',params=params_region_vide, h=h)/H0
hz_region_dense = cosmo.get_dist(zz,type='hz',params=params_region_dense, h=h)/H0
hz_global = cosmo.get_dist(zz,type='hz',params=params_global, h=h)/H0
dp_region_vide = cosmo.get_dist(zz,type=dist_type,params=params_region_vide, h=h)
dp_region_dense = cosmo.get_dist(zz,type=dist_type,params=params_region_dense, h=h)
dp_global = cosmo.get_dist(zz,type=dist_type,params=params_global, h=h)
clf()
subplot(1,2,1)
plot(zz,hz_global,'b',label='$\Omega_m = {0:4.2f}$'.format(om_global))
plot(zz,hz_region_vide,'r',label='$\Omega_m = {0:4.2f}$'.format(om_region_vide))
plot(zz,hz_region_dense,'g',label='$\Omega_m = {0:4.2f}$'.format(om_region_dense))
xlabel('z')
ylabel('H(z)')
legend(loc='upper left')
subplot(1,2,2)
plot(zz,dp_global,'b')
plot(zz,dp_region_vide,'r')
plot(zz,dp_region_dense,'g')
xlabel('z')
ylabel(dist_type)




########### En résolvant l'équation différentielle
import scipy.integrate
c=3e8
H0 = 1000*1000*h*100

x = linspace(0,x1,nn)

# constant om global situation A
omA=np.zeros(nn)+om_global

# constant om vide situation A
omVide=np.zeros(nn)+om_region_vide

# constant om dense situation A
omDense=np.zeros(nn)+om_region_dense

# With hole - situation B
omB = np.zeros(nn)
omB[x < xs]=om_region_dense
omB[(x>=xs) & (x<xe)] = om_region_vide
omB[x>=xe]=om_region_dense

zrecA = inhodist.x2z_inho(x, omA, h,type=dist_type)
zrecB = inhodist.x2z_inho(x, omB, h,type=dist_type)
zrecVide = inhodist.x2z_inho(x, omVide, h,type=dist_type)
zrecDense = inhodist.x2z_inho(x, omDense, h,type=dist_type)

clf()
plot(zz, dp_global,lw=2)
plot(zrecA,x,'r--',lw=2)



clf()
plot(zrecA,x, label='A')
plot(zrecB,x, label='B')
plot(zrecVide,x, label='Vide')
plot(zrecDense,x, label='Dense')
legend()

clf()
plot(zrecA,x/np.interp(zrecA, zz, dp_global), lw=2, label='A: Uniform $\Omega_m={0:3.1f}$'.format(om_global))
plot(zrecB,x/np.interp(zrecB, zz, dp_global), lw=2, label='B: Non-Uniform with same average $\Omega_m={0:3.1f}$ with hole $\Omega_m={1:3.1f}$'.format(om_region_dense,om_region_vide))
plot(zrecVide,x/np.interp(zrecVide, zz, dp_global), lw=2, label='Uniform $\Omega_m={0:3.1f}$'.format(om_region_vide))
plot(zrecDense,x/np.interp(zrecDense, zz, dp_global), lw=2, label='Uniform $\Omega_m={0:3.1f}$'.format(om_region_dense))
xlabel('redshift')
ylabel('Distance ratio to global')
ylim(0.9,1.1)
legend(loc='upper left',framealpha=0.8)
savefig('essai.png')


############# Check that it is fine
clf()
subplot(2,2,1)
xlim(0,zmax)
plot(zrecA,x,'b')
plot(zz,dp_global,'b:',lw=4)
title('Situation A: $\Omega_m = {0:3.1f}$'.format(om_global))

subplot(2,2,2)
plot(zrecB,x,'r')
title('Situation B: with zone $\Omega_m = {0:4.2f}$'.format(om_region_vide))

subplot(2,2,3)
plot(zrecVide,x,'r')
plot(zz,dp_region_vide,'r:',lw=4)
title('Constant $\Omega_m = {0:4.2f}$'.format(om_region_vide))

subplot(2,2,4)
plot(zrecDense,x,'r')
plot(zz,dp_region_dense,'r:',lw=4)
title('Constant $\Omega_m = {0:4.2f}$'.format(om_region_dense))



### 

