from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import pi
import pycamb
from scipy.constants import c
from Sensitivity import qubic_sensitivity
from Homogeneity import SplineFitting
from qubic import QubicInstrument, QubicScene

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


############# Shifting horns quarters
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



############ Synthesized beam
from qubic import QubicInstrument, QubicScene
scene = QubicScene(150, nside=512)
sb = inst.get_synthbeam_healpix_from_position(scene, 0, 0)

sb_shift = inst_shift.get_synthbeam_healpix_from_position(scene, 0, 0)

hp.gnomview(np.log(sb), rot=[0,90], reso=10)
hp.gnomview(np.log(sb_shift), rot=[0,90], reso=10)

clinst = hp.anafast(sb)


############ Synthesized beam with bandwidth
def sb_bw(inst, freq, dnu_nu=0.25, nside=512, nn=10):
	valsnu = linspace(freq*(1-dnu_nu/2), freq*(1+dnu_nu/2), nn)
	sb = np.zeros(12*nside**2)
	for i in xrange(nn):
		print('subfreq '+str(i)+' over '+str(nn))
		scene = QubicScene(valsnu[i], nside=nside)
		sb += inst.get_synthbeam_healpix_from_position(scene, 0, 0)/nn
	return sb

sbbw = sb_bw(inst, 150)
hp.gnomview(np.log(sbbw), rot=[0,90], reso=10)


######################################### No constraint on window size

########### Do the loop with semi-analytical
mini = 0.
nn = 4
maxi = nn*inst.horn.spacing
nb = 20*nn+1
shifts0 = np.linspace(mini,maxi,nb)
valnbsig=np.zeros(nb)
epsilon = 0.7 * 0.7  #focal plane integration + optical efficiency)
ellbins = np.array([50.,150.])
for i in xrange(nb):
	print(i)
	### Semi-analytical errors
	ellav, deltacl2, noisevar2, samplevar2, neq_nh2, nsig2, bs2= qubic_sensitivity.give_qubic_errors(shift_inst(shifts0[i]), ellbins, lll, spectra[3], nu=150e9, epsilon=epsilon,net_polar=NET150)
	valnbsig[i] = nsig2


############ Loop with synthesized beam no bw
nbshift = 20
shifts = linspace(0,0.05,nbshift)
cl=[]
for i in xrange(nbshift):
	print(i)
	inst_shift = shift_inst(shifts[i])
	sb_shift = inst_shift.get_synthbeam_healpix_from_position(scene, 0, 0)
	cl.append(hp.anafast(sb_shift))

mm = np.zeros(nbshift)
ss = np.zeros(nbshift)
for i in xrange(nbshift):
	mm[i] = np.mean((cl[i]/cl[0])[50:150])
	ss[i] = np.std((cl[i]/cl[0])[50:150])


########### Loop with synthesized beam and BW
nbshift = 20
shifts = linspace(0,0.05,nbshift)
clbw=[]
for i in xrange(nbshift):
	print(i)
	inst_shift = shift_inst(shifts[i])
	sb_shift = sb_bw(inst_shift, 150)
	clbw.append(hp.anafast(sb_shift))

mmbw = np.zeros(nbshift)
ssbw = np.zeros(nbshift)
for i in xrange(nbshift):
	mmbw[i] = np.mean((clbw[i]/clbw[0])[50:150])
	ssbw[i] = np.std((clbw[i]/clbw[0])[50:150])


clf()
ylim(0,1)
xlim(0,50)
plot(shifts0*1000,valnbsig/valnbsig[0],'r',lw=3,label='Semi-Analytical')
plot(shifts*1000,mm,'b',lw=3,label='No Bandwidth')
plot(shifts*1000,mmbw,'g',lw=3,label='25% Bandwidth')
xlabel('central extra-spacing between quarters [mm]')
ylabel('Sensitivity loss on $\#\sigma$')
nn=10
for i in xrange(nn+1): plot([inst.horn.spacing*i*1000, inst.horn.spacing*i*1000],[0,5],'k--')
legend()
title('No constraint on window size')
savefig('horns_spacing_variable_window.png')





######################################### With constant window size
rmax = np.max(np.sqrt(np.sum(inst.horn.center[:,:2]**2,axis=1)))

########### Do the loop with semi-analytical
mini = 0.
nn = 4
maxi = nn*inst.horn.spacing
nb = 20*nn+1
shifts0 = np.linspace(mini,maxi,nb)
valnbsig=np.zeros(nb)
epsilon = 0.7 * 0.7  #focal plane integration + optical efficiency)
ellbins = np.array([50.,150.])
for i in xrange(nb):
	print(i)
	### Semi-analytical errors
	new_inst = shift_inst(shifts0[i])
	radii = np.sqrt(np.sum(new_inst.horn.center[:,:2]**2,axis=1))
	new_inst.horn.open[radii > rmax] = False
	ellav, deltacl2, noisevar2, samplevar2, neq_nh2, nsig2, bs2= qubic_sensitivity.give_qubic_errors(new_inst, ellbins, lll, spectra[3], nu=150e9, epsilon=epsilon,net_polar=NET150)
	valnbsig[i] = nsig2

############ Loop with synthesized beam no bw
nbshift = 20
shifts = linspace(0,0.05,nbshift)
cl=[]
for i in xrange(nbshift):
	print(i)
	inst_shift = shift_inst(shifts[i])
	radii = np.sqrt(np.sum(inst_shift.horn.center[:,:2]**2,axis=1))
	inst_shift.horn.open[radii > rmax] = False
	sb_shift = inst_shift.get_synthbeam_healpix_from_position(scene, 0, 0)
	cl.append(hp.anafast(sb_shift))

mm = np.zeros(nbshift)
ss = np.zeros(nbshift)
for i in xrange(nbshift):
	mm[i] = np.mean((cl[i]/cl[0])[50:150])
	ss[i] = np.std((cl[i]/cl[0])[50:150])


########### Loop with synthesized beam and BW
nbshift = 20
shifts = linspace(0,0.05,nbshift)
clbw=[]
for i in xrange(nbshift):
	print(i)
	inst_shift = shift_inst(shifts[i])
	radii = np.sqrt(np.sum(inst_shift.horn.center[:,:2]**2,axis=1))
	inst_shift.horn.open[radii > rmax] = False
	sb_shift = sb_bw(inst_shift, 150)
	clbw.append(hp.anafast(sb_shift))

mmbw = np.zeros(nbshift)
ssbw = np.zeros(nbshift)
for i in xrange(nbshift):
	mmbw[i] = np.mean((clbw[i]/clbw[0])[50:150])
	ssbw[i] = np.std((clbw[i]/clbw[0])[50:150])


clf()
ylim(0,1)
xlim(0,50)
plot(shifts0*1000,valnbsig/valnbsig[0],'r',lw=3,label='Semi-Analytical')
plot(shifts*1000,mm,'b',lw=3,label='No Bandwidth')
plot(shifts*1000,mmbw,'g',lw=3,label='25% Bandwidth')
xlabel('central extra-spacing between quarters [mm]')
ylabel('Sensitivity loss on $\#\sigma$')
nn=10
for i in xrange(nn+1): plot([inst.horn.spacing*i*1000, inst.horn.spacing*i*1000],[0,5],'k--')
legend()
title('Fixed window size')
savefig('horns_spacing_fixed_window.png')






