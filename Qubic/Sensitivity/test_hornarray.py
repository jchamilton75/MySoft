from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import pi
import pycamb
from qubic import QubicInstrument
from scipy.constants import c
from Sensitivity import qubic_sensitivity
from Homogeneity import SplineFitting


#### Get input Power spectra
################# Input Power spectrum ###################################
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
r = 0.05
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
         'tensor_ratio':r,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}
lmaxcamb = 1000
T,E,B,X = pycamb.camb(lmaxcamb+1+150,**params)
lll = np.arange(1,lmaxcamb+1)
T=T[:lmaxcamb]
E=E[:lmaxcamb]
B=B[:lmaxcamb]
X=X[:lmaxcamb]
fact = (lll*(lll+1))/(2*np.pi)
spectra = [lll, T/fact, E/fact, B/fact, X/fact]

params_nl = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':r,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False}
lmaxcamb = 1000
Tnl,Enl,Bnl,Xnl = pycamb.camb(lmaxcamb+1+150,**params_nl)
lll = np.arange(1,lmaxcamb+1)
Tnl=Tnl[:lmaxcamb]
Enl=Enl[:lmaxcamb]
Bnl=Bnl[:lmaxcamb]
Xnl=Xnl[:lmaxcamb]
fact = (lll*(lll+1))/(2*np.pi)
spectra_nl = [lll, Tnl/fact, Enl/fact, Bnl/fact, Xnl/fact]

clf()
mp.plot(lll, np.sqrt(spectra[1]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{TT}$')
mp.plot(lll, np.sqrt(abs(spectra[4])*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{TE}$')
mp.plot(lll,np.sqrt(spectra[2]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{EE}$')
mp.plot(lll,np.sqrt(spectra[3]*(lll*(lll+1))/(2*np.pi)),label='$C_\ell^{BB}$')
mp.yscale('log')
mp.xlim(0,lmaxcamb+1)
#ylim(0.0001,100)
mp.xlabel('$\ell$')
mp.ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
mp.legend(loc='lower right',frameon=False)







########## Qubic Instrument
NET150 = (220+314)/2*sqrt(2)*sqrt(2)  ### Average between witner and summer
NET220 = (520+906)/2*sqrt(2)*sqrt(2)  ### Average between witner and summer
inst = QubicInstrument()



def shift_inst(shift):
	inst = QubicInstrument()
	matback = np.array([[np.cos(np.radians(-inst.horn.angle)), -np.sin(np.radians(-inst.horn.angle))],[np.sin(np.radians(-inst.horn.angle)), np.cos(np.radians(-inst.horn.angle))]])
	matfwd = matback = np.array([[np.cos(np.radians(inst.horn.angle)), -np.sin(np.radians(inst.horn.angle))],[np.sin(np.radians(inst.horn.angle)), np.cos(np.radians(inst.horn.angle))]])
	xyrot = np.dot(matback, inst.horn.center[:,:2].T).T
	xplus = xyrot[:,0] > 0
	xyrot[xplus,0] += shift/2
	xyrot[~xplus,0] += -shift/2
	yplus = xyrot[:,1] > 0
	xyrot[yplus,1] += shift/2
	xyrot[~yplus,1] += -shift/2
	#stop
	inst.horn.center[:,:2] = np.dot(np.linalg.inv(matback), xyrot.T).T
	return inst

inst_shift = shift_inst(0.01)

clf()
plot(inst.horn.center[:,0],inst.horn.center[:,1],'bo',alpha=0.1)
plot(inst_shift.horn.center[:,0],inst_shift.horn.center[:,1],'ro',alpha=0.1)

### Binning
deltal=100.
lmin=25
ellbins = np.array([25, 45, 65, 95, 140, 200, 300])
ellmin = ellbins[:len(ellbins)-1]
ellmax = ellbins[1:len(ellbins)]


### Get baselines and errors
epsilon = 0.7 * 0.7  #focal plane integration + optical efficiency)

clf()
subplot(2,1,1)
xlim(-300,300)
ylim(-300,300)
xlabel('$u$')
ylabel('$v$')
subplot(2,1,2)
ylim(0,1)
ylabel('$N_{eq}(\ell)/N_h$')
xlabel('$\ell$')

#### initial
ellav, deltacl, noisevar, samplevar, neq_nh, nsig, bs= qubic_sensitivity.give_qubic_errors(inst, ellbins, lll, spectra[3], nu=150e9, epsilon=epsilon,net_polar=NET150, plot_baselines=True, symplot='ro')

#### shifted
ellav, deltacl2, noisevar2, samplevar2, neq_nh2, nsig2, bs2= qubic_sensitivity.give_qubic_errors(shift_inst(0.01), ellbins, lll, spectra[3], nu=150e9, epsilon=epsilon,net_polar=NET150, plot_baselines=True, symplot='bo')


mini = 0.
nn = 4
maxi = nn*inst.horn.spacing
nb = 20*nn+1
theshift = np.linspace(mini,maxi,nb)
valnbsig = np.zeros(nb)

for i in xrange(nb):
	print(i)
	ellav, deltacl2, noisevar2, samplevar2, neq_nh2, nsig2= qubic_sensitivity.give_qubic_errors(shift_inst(theshift[i]), ellbins, lll, spectra[3], nu=150e9, epsilon=epsilon,net_polar=NET150)
	valnbsig[i] = nsig2


indices = theshift/inst.horn.spacing % 1 == 0


clf()
ylim(0,1)
xlim(-1, 60)
xlabel('central extra-spacing between quarters [mm]')
ylabel('Sensitivity loss on $\#\sigma$')
for i in xrange(nn+1): plot([inst.horn.spacing*i*1000, inst.horn.spacing*i*1000],[0,5],'k--')
plot(theshift*1000, valnbsig/valnbsig[0],'b')
plot(theshift[indices]*1000, valnbsig[indices]/valnbsig[0],'r')
savefig('central_spacing.png')







