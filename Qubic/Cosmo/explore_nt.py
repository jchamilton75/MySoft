
from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import pycamb
#from qubic.utils import progress_bar
from Homogeneity import fitting

###### Input Power spectrum ###################################
################# Input Power spectrum ###################################
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
rvalue = 0.2
print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',omegav/h2

params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':rvalue,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False,
         'tensor_index':-rvalue/8}

params_l = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
        	 'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
        	 'tensor_ratio':0,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}

lmax = 1000
ell = np.arange(1,600)


Tprim,Eprim,Bprim,Xprim = pycamb.camb(lmax+1,**params)
Tprim = Tprim[0:599]
Eprim = Eprim[0:599]
Bprim = Bprim[0:599]
Xprim = Xprim[0:599]
Tl,El,Bl,Xl = pycamb.camb(lmax+1,**params_l)
Tl = Tl[0:599]
El = El[0:599]
Bl = Bl[0:599]
Xl = Xl[0:599]
Btot = Bl + Bprim

clf()
plot(ell,Bprim,label='Primordial $C_\ell^{BB}$'+'$ (r={0:.2f})$'.format(rvalue))
plot(ell,Bl,label='Lensing $C_\ell^{BB}$')
plot(ell,Btot,label='Total $C_\ell^{BB}$'+'$ (r={0:.2f})$'.format(rvalue))
yscale('log')
xlim(0,600)
ylim(0.0005,0.1)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$'+'    '+'$[\mu K]^2$ ')
legend(loc='upper left',frameon=False)



####### varying nt
nbnt = 11
ntmin = -0.1
ntmax = 0.1
ntvals = linspace(ntmin, ntmax, nbnt) 

allspectra_prim = np.zeros((nbnt, len(Bprim)))
#bar = progress_bar(nbnt)
for i in range(nbnt):
	theparams = params.copy()
	theparams['tensor_index'] = ntvals[i]
	t,e,b,x = pycamb.camb(lmax+1, **theparams)
	b = b[0:599]
	allspectra_prim[i,:] = b
	#bar.update()

allspectra_tot = allspectra_prim + Bl



clf()
plot(ell,Bprim,label='Primordial $C_\ell^{BB}$'+'$ (n_t={0:.3f})$'.format(params['tensor_index']),lw=2)
for i in range(nbnt):
	plot(ell,allspectra_prim[i,:],label='Primordial $C_\ell^{BB}$'+'$ (n_t={0:.3f})$'.format(ntvals[i]))

yscale('log')
xlim(0,600)
ylim(0.0005,0.1)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$'+'    '+'$[\mu K]^2$ ')
legend(loc='upper left',frameon=False)



clf()
for i in range(nbnt):
	plot(ell,allspectra_prim[i,:]/Bprim,label='Primordial $C_\ell^{BB}$'+'$ (n_t={0:.3f})$'.format(ntvals[i]))

yscale('log')
xscale('log')
xlim(0,600)
ylim(0.8,1.2)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$'+'    '+'$[\mu K]^2$ ')
legend(loc='upper left',frameon=False)


clf()
plot(ell,Bpl,label='Primordial $C_\ell^{BB}$'+'$ (n_t={0:.3f})$'.format(params['tensor_index']),lw=2)
for i in range(nbnt):
	plot(ell,allspectra_tot[i,:],label='Primordial $C_\ell^{BB}$'+'$ (n_t={0:.3f})$'.format(ntvals[i]))

yscale('log')
xlim(0,600)
ylim(0.0005,0.1)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$'+'    '+'$[\mu K]^2$ ')
legend(loc='upper left',frameon=False)


###### try to avooid calculating nt with camb ?
clf()
for i in range(nbnt):
	plot(ell,allspectra_prim[i,:]/Bprim*((ell*0.002)**(params['tensor_index']-ntvals[i])),
		label='Primordial $C_\ell^{BB}$'+'$ (n_t={0:.3f})$'.format(ntvals[i]))

#yscale('log')
#xscale('log')
xlim(0,600)
ylim(0.3,3)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$'+'    '+'$[\mu K]^2$ ')
legend(loc='upper left',frameon=False)



