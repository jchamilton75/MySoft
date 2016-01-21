from pylab import *
import numpy as N
from scipy import integrate
from Cosmology import cosmology as cosmo
from Cosmology import inhodist

zmax = 2
### ces distances sont en Mpc
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

######## On travaille en COMOVING DISTANCE (line of sight) au sens de Hogg, 
######## donc pas de correction de courbure. On prend directement chi
hz_region_vide = cosmo.get_dist(zz,type='hz',params=params_region_vide, h=h)/H0
hz_region_dense = cosmo.get_dist(zz,type='hz',params=params_region_dense, h=h)/H0
hz_global = cosmo.get_dist(zz,type='hz',params=params_global, h=h)/H0
dp_region_vide = cosmo.get_dist(zz,type='comoving',params=params_region_vide, h=h)
dp_region_dense = cosmo.get_dist(zz,type='comoving',params=params_region_dense, h=h)
dp_global = cosmo.get_dist(zz,type='comoving',params=params_global, h=h)
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
ylabel('comoving distance (l.o.s.)')



#### Configuration A
hz_A = hz_global.copy()
chi_A = np.zeros(nn)
chi_A[1:nn] = integrate.cumtrapz(1./hz_A, zz)
dp_A = chi_A*c/H0
zs_A = np.interp(xs, dp_A, zz)
ze_A = np.interp(xe, dp_A, zz)
z1_A = np.interp(x1, dp_A, zz)
## Other calculation from differential equation
xx = linspace(0,x1,nn)
omA=np.zeros(nn)+om_global
zrecA = inhodist.x2z_inho(xx, omA, h,type='comoving')



clf()
plot(zz, dp_global,lw=2)
plot(zrecA,xx,'r--',lw=2)





#### Configuration B
# region dense du début
hz_B = hz_region_dense.copy()
chi = np.zeros(nn)
chi[1:nn] = integrate.cumtrapz(1./hz_B, zz)
theok = 1.-om_region_dense
dp0 = chi*c/H0
# region vide
mask2 = (dp0 >= xs)
hz_B[mask2] = hz_region_vide[mask2] 
chi[1:nn] = integrate.cumtrapz(1./hz_B, zz)
theom = 1.-om_region_vide
dp1 = chi*c/H0
# region finale
mask3 = (dp1 >= xe)
hz_B[mask3] = hz_region_dense[mask3] 
chi[1:nn] = integrate.cumtrapz(1./hz_B, zz)

dp_B = chi*c/H0

ze_B = np.interp(xe, dp_B, zz)
zs_B = np.interp(xs, dp_B, zz)
z1_B = np.interp(x1, dp_B, zz)

# With hole - situation B
omB = np.zeros(nn)
omB[xx < xs]=om_region_dense
omB[(xx>=xs) & (xx<xe)] = om_region_vide
omB[xx>=xe]=om_region_dense
zrecB = inhodist.x2z_inho(xx, omB, h,type='comoving')





clf()
subplot(1,3,1)
mx = np.max([hz_global,hz_B])
plot(zz,hz_global,'b',label='global $\Omega_m = {0:3.1f}$'.format(om_global))
plot(zz,hz_B,'r',label='with zone $\Omega_m = {0:3.1f}$'.format(om_region_vide))
plot([zs_A,zs_A],[0,mx],'b:')
plot([ze_A,ze_A],[0,mx],'b:')
plot([z1_A,z1_A],[0,mx],'b:')
plot([zs_B,zs_B],[0,mx],'r:')
plot([ze_B,ze_B],[0,mx],'r:')
plot([z1_B,z1_B],[0,mx],'r:')
ylim(0,mx)
ylabel('H(z)')
xlabel('z')
legend(loc='lower right')

subplot(1,3,2)
mx = np.max([dp_A,dp_B])
plot(zz,dp_A,'b',label='global $\Omega_m = {0:3.1f}$'.format(om_global))
plot(zz,dp_B,'r',label='with zone $\Omega_m = {0:3.1f}$'.format(om_region_vide))
plot(zrecA, xx, 'b--', lw=2, label='Equa diff A')
plot(zrecB, xx, 'r--', lw=2, label='Equa diff B')
plot([0,zmax],[xs,xs],'k:')
plot([0,zmax],[xe,xe],'k:')
plot([0,zmax],[x1,x1],'k:')
plot([zs_A,zs_A],[0,mx ],'b:')
plot([ze_A,ze_A],[0,mx ],'b:')
plot([z1_A,z1_A],[0,mx ],'b:')
plot([zs_B,zs_B],[0,mx ],'r:')
plot([ze_B,ze_B],[0,mx ],'r:')
plot([z1_B,z1_B],[0,mx ],'r:')
ylim(0,mx )
xlim(0,zmax)
ylabel('Comoving distance (l.o.s.)')
xlabel('z')
plot(zz, dp_global,'k:',lw=4, label='Global')
legend(loc='lower right')


subplot(1,3,3)
mx = np.max([dp_A,dp_B])
plot(zz,dp_A/np.interp(zz, zz, dp_global),'b',label='global $\Omega_m = {0:3.1f}$'.format(om_global))
plot(zz,dp_B/np.interp(zz, zz, dp_global),'r',label='with zone $\Omega_m = {0:3.1f}$'.format(om_region_vide))
plot(zrecA, xx/np.interp(zrecA, zz, dp_global), 'b--', lw=2, label='Equa diff A')
plot(zrecB, xx/np.interp(zrecB, zz, dp_global), 'r--', lw=2, label='Equa diff B')
plot([zs_A,zs_A],[0,mx],'b:')
plot([ze_A,ze_A],[0,mx],'b:')
plot([z1_A,z1_A],[0,mx],'b:')
plot([zs_B,zs_B],[0,mx],'r:')
plot([ze_B,ze_B],[0,mx],'r:')
plot([z1_B,z1_B],[0,mx],'r:')
ylim(0.9,1.1)
xlim(0,zmax)
ylabel('Comoving distance (l.o.s.) Ratio to Global')
xlabel('z')
legend(loc='lower right')



