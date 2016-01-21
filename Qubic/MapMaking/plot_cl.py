from __future__ import division
import healpy as hp
import numpy as np
import pycamb as pc

## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075

H0 = 67.04
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
         'tensor_ratio':0.2,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}
paramsnl = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
           'tensor_ratio':0.2,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False}
paramsnl2 = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
           'tensor_ratio':0.01,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False}
paramsnl3 = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
           'tensor_ratio':0.001,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False}
lmax = 1200
ell = np.arange(1,lmax+1)
Tnl,Enl,Bnl,Xnl = pc.camb(lmax+1,**paramsnl)
Tnl2,Enl2,Bnl2,Xnl2 = pc.camb(lmax+1,**paramsnl2)
Tnl3,Enl3,Bnl3,Xnl3 = pc.camb(lmax+1,**paramsnl3)
T,E,B,X = pc.camb(lmax+1,**params)
Blens = B-Bnl

clf()
lw=3
plot(ell,np.sqrt(T),'k-',label='$C_\ell^{TT}$',lw=lw)
plot(ell,np.sqrt(X),'g-',label='$C_\ell^{TE}$',lw=lw)
plot(ell,np.sqrt(-X),'g--',label='$-C_\ell^{TE}$',lw=lw)
plot(ell,np.sqrt(E),'b-',label='$C_\ell^{EE}$',lw=lw)
plot(ell,np.sqrt(Bnl),'r-',label='Primordial $C_\ell^{BB} (r=0.1)$',lw=lw)
plot(ell,np.sqrt(Bnl2),'-',label='Primordial $C_\ell^{BB} (r=0.01)$',lw=lw,color='firebrick')
plot(ell,np.sqrt(Bnl3),'-',label='Primordial $C_\ell^{BB} (r=0.001)$',lw=lw,color='darkred')
plot(ell,np.sqrt(Blens),'--',label='Lensing $C_\ell^{BB}$',lw=lw,color='darkviolet')
yscale('log')
xscale('log')
xlim(1,1000)
ylim(0.001,100)
xlabel('$\ell$',fontsize=20)
ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ',fontsize=20)
ax=gca()
leg = ax.legend(loc='upper left',fancybox=True,fontsize=15)
leg.get_frame().set_alpha(0.7)
savefig('cl.png')





