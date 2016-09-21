############## Nice plots to run after dualband_mcmc.py




#### Build a Cl Library with CAMB
from Cosmo import interpol_camb as ic
from pysimulators import FitsArray
 from Sensitivity import dualband_lib as db


rmin = 0.001
rmax = 1
nb =100
lmaxcamb = 600
rvalues = np.concatenate((np.zeros(1),np.logspace(np.log10(rmin),np.log10(rmax),nb)))
camblib = ic.rcamblib(rvalues, lmaxcamb)
FitsArray(camblib[0], copy=False).save('camblib600_ell.fits')
FitsArray(camblib[1], copy=False).save('camblib600_r.fits')
FitsArray(camblib[2], copy=False).save('camblib600_cl.fits')
### Restore it
ellcamblib = FitsArray('camblib600_ell.fits')
rcamblib = FitsArray('camblib600_r.fits')
clcamblib = FitsArray('camblib600_cl.fits')
camblib = [ellcamblib, rcamblib, clcamblib]


###### Nice plots
lmax = 600
lll = linspace(0,lmax,lmax+1)
cl150x150 = db.get_ClBB_cross_th(lll, 150, freqGHz2=150, dustParams = None, rvalue=0.05, camblib=camblib)
cl150x220 = db.get_ClBB_cross_th(lll, 150, freqGHz2=220, dustParams = None, rvalue=0.05, camblib=camblib)
cl150x353 = db.get_ClBB_cross_th(lll, 150, freqGHz2=353, dustParams = None, rvalue=0.05, camblib=camblib)
cl220x220 = db.get_ClBB_cross_th(lll, 220, freqGHz2=220, dustParams = None, rvalue=0.05, camblib=camblib)
cl220x353 = db.get_ClBB_cross_th(lll, 220, freqGHz2=353, dustParams = None, rvalue=0.05, camblib=camblib)
cl353x353 = db.get_ClBB_cross_th(lll, 353, freqGHz2=353, dustParams = None, rvalue=0.05, camblib=camblib)

fact = lll*(lll+1)/(2*np.pi)

clf()
yscale('log')
xscale('log')
xlim(10,lmax)
ylim(0.001,100)
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell / 2\pi \,\,[\mu K^2]$')
plot(lll, fact*cl150x150[1],'k--',lw=2, label ='Primordial B-modes r=0.05')
plot(lll, fact*cl150x150[2],'b:',lw=2, label ='Dust 150x150')
plot(lll, fact*cl150x150[0],'b',lw=2)
plot(lll, fact*cl150x220[2],'g:',lw=2, label ='Dust 150x220')
plot(lll, fact*cl150x220[0],'g',lw=2)
plot(lll, fact*cl150x353[2],'r:',lw=2, label ='Dust 150x353')
plot(lll, fact*cl150x353[0],'r',lw=2)
plot(lll, fact*cl220x220[2],'m:',lw=2, label ='Dust 220x220')
plot(lll, fact*cl220x220[0],'m',lw=2)
plot(lll, fact*cl220x353[2],'c:',lw=2, label ='Dust 220x353')
plot(lll, fact*cl220x353[0],'c',lw=2)
plot(lll, fact*cl353x353[2],'y:',lw=2, label ='Dust 353x353')
plot(lll, fact*cl353x353[0],'y',lw=2)
legend(loc='upper left',framealpha=0.5)
savefig('spectra.png', transparent=True)




from qubic import QubicInstrument
inst = QubicInstrument()

### Binning (as BICEP2)
ellbins = np.array([21, 56, 91, 126, 161, 196, 231, 266, 301, 336, 371, 406])#, 441, 476, 511, 546, 581])
ellmin = ellbins[:len(ellbins)-1]
ellmax = ellbins[1:len(ellbins)]

net150_concordia = 0.5*(291.+369.)*np.sqrt(2)
net220_concordia = 0.5*(547.+840.)*np.sqrt(2)

qubic_duration = 2*365.*24.*3600.
qubic_epsilon = 1.



### QUBIC 150 and 220 GHz

dldust_80_353 = 13.4*0.45 ## level so that the fill_between betlow shows the PlanckXXX measurement
alphadust = -2.42
betadust = 1.59
Tdust = 19.6
Planck_in_Bicep_pars = np.array([dldust_80_353, alphadust, betadust, Tdust])

cl150x150 = db.get_ClBB_cross_th(lll, 150, freqGHz2=150, dustParams = Planck_in_Bicep_pars, rvalue=0.05, camblib=camblib)
cl150x220 = db.get_ClBB_cross_th(lll, 150, freqGHz2=220, dustParams = Planck_in_Bicep_pars, rvalue=0.05, camblib=camblib)
cl220x220 = db.get_ClBB_cross_th(lll, 220, freqGHz2=220, dustParams = Planck_in_Bicep_pars, rvalue=0.05, camblib=camblib)


def truc(thervalue, inst, ellbins, freqs, type, NETs, name, col, fsky, duration, epsilon, camblib=None, dustParams=None):
	instrument = [inst, ellbins, freqs, type, NETs, fsky, duration, epsilon, name, col]
	bla = db.get_multiband_covariance(instrument, thervalue, doplot=True, dustParams=dustParams, verbose=True, camblib=camblib)
	return(bla)

bla = truc(0.05, inst, ellbins,
	[150, 220],
	['bi', 'bi'],
	[net150_concordia, net220_concordia],
	['150, 220'],
	'm',
	0.01,
	[qubic_duration, qubic_duration],
	[qubic_epsilon, qubic_epsilon],
	camblib=camblib, dustParams=Planck_in_Bicep_pars)


ellvals = (ellmin + ellmax)/2
specbin = np.reshape(bla[3], ((3,len(ellvals))))
specbinerr= np.reshape(bla[4], ((3,len(ellvals))))

spectrum = np.array(db.get_ClBB_cross_th(lll, 150, rvalue=0., camblib=camblib)[1])


clf()
subplot(3,1,1)
xlabel('$\ell$')
ylabel('$D_\ell^{BB}$ [$\mu K^2$]')
plot(lll, spectrum*lll*(lll+1)/2/pi,'k--', label ='Lensing B-modes')
plot(lll, fact*cl150x150[1],'k',lw=2, label ='Lensing + Primordial B-modes r=0.05')
plot(lll, fact*cl150x150[2],'r:',lw=2, label ='Dust 150x150')
plot(lll, fact*cl150x150[0],'r',lw=2,label='Total B-modes 150x150')
errorbar(ellvals, specbin[0,:], yerr=specbinerr[0,:], fmt='ro',label='QUBIC')
fill_between([40,120], [1.32e-2-0.29e-2, 1.32e-2-0.29e-2], y2=[1.32e-2+0.29e-2, 1.32e-2+0.29e-2], color='b',alpha=0.3,label='Planck XXX extrapolation')
plot(ellvals, specbinerr[0,:], 'r',label='QUBIC errors')
ylim(0,0.05)
xlim(0,400)
title('QUBIC 150x150 - 2 years')
legend(framealpha=0.3, fontsize=8, loc='upper left')

subplot(3,1,2)
xlabel('$\ell$')
ylabel('$D_\ell^{BB}$ [$\mu K^2$]')
plot(lll, spectrum*lll*(lll+1)/2/pi,'k--', label ='Lensing B-modes')
plot(lll, fact*cl150x150[1],'k',lw=2, label ='Lensing + Primordial B-modes r=0.05')
plot(lll, fact*cl150x220[2],'g:',lw=2, label ='Dust 150x220')
plot(lll, fact*cl150x220[0],'g',lw=2,label='Total B-modes 150x220')
errorbar(ellvals, specbin[1,:], yerr=specbinerr[1,:], fmt='go',label='QUBIC')
plot(ellvals, specbinerr[1,:], 'g',label='QUBIC errors')
ylim(0,0.12)
xlim(0,400)
title('QUBIC 150x220 - 2 years')
legend(framealpha=0.3, fontsize=8, loc='upper left')

subplot(3,1,3)
xlabel('$\ell$')
ylabel('$D_\ell^{BB}$ [$\mu K^2$]')
plot(lll, spectrum*lll*(lll+1)/2/pi,'k--', label ='Lensing B-modes')
plot(lll, fact*cl150x150[1],'k',lw=2, label ='Lensing + Primordial B-modes r=0.05')
plot(lll, fact*cl220x220[2],'b:',lw=2, label ='Dust 220x220')
plot(lll, fact*cl220x220[0],'b',lw=2,label='Total B-modes 220x220')
errorbar(ellvals, specbin[2,:], yerr=specbinerr[2,:], fmt='bo',label='QUBIC 2 years')
plot(ellvals, specbinerr[2,:], 'b',label='QUBIC errors')
ylim(0,0.25)
xlim(0,400)
title('QUBIC 220x220 - 2 years')
legend(framealpha=0.3, fontsize=8, loc='upper left')









theepsilon =0.3
bla = truc(0.05, inst, ellbins,
	[150, 220],
	['bi', 'bi'],
	[net150_concordia, net220_concordia],
	['150, 220'],
	'm',
	0.01,
	[qubic_duration, qubic_duration],
	[theepsilon, theepsilon],
	camblib=camblib, dustParams=Planck_in_Bicep_pars)


ellvals = (ellmin + ellmax)/2
specbin = np.reshape(bla[3], ((3,len(ellvals))))
specbinerr= np.reshape(bla[4], ((3,len(ellvals))))

noise_errors = np.reshape(np.sqrt(np.diag(bla[1])), (3,len(ellvals)))

ther = 0.05
thecl = ic.get_Dlbb_fromlib(lll, ther, camblib)
dllensing = spectrum*lll*(lll+1)/2/pi

lbins = np.zeros(2*len(ellvals))
errs = np.zeros((3,2*len(ellvals)))
for i in xrange(len(ellvals)):
	lbins[i*2] = ellmin[i]
	lbins[i*2+1] = ellmax[i]
	errs[:,i*2] = noise_errors[:,i] 
	errs[:,i*2+1] = noise_errors[:,i] 


clf()
yscale('log')
#xscale('log')
xlim(0,300)
ylim(2e-4, 5e-1)
title('QUBIC 2 Years - $\epsilon$ = {0:3.1f}'.format(theepsilon))
plot(lbins, errs[0,:], lw=2, color='red', label='QUBIC 150x150 GHz')
plot(lbins, errs[1,:], lw=2, color='purple', label='QUBIC 150x220 GHz')
plot(lbins, errs[2,:], lw=2, color='blue', label='QUBIC 220x220 GHz')
plot(lll, fact*cl150x150[2],'r--',lw=2, label ='Dust 150x150')
plot(lll, fact*cl150x220[2],'--',color='purple', lw=2, label ='Dust 150x220')
plot(lll, fact*cl220x220[2],'b--',lw=2, label ='Dust 220x220')
plot(lll, thecl,'k', lw=2,label ='BB r={0:4.2f} + Lensing'.format(ther))
plot(lll, thecl-dllensing,'k:',lw=2, label ='BB r={0:4.2f}'.format(ther))
legend(loc='lower right', framealpha=0)
xlabel('$\ell$')
ylabel('$D_\ell^{BB}$ [$\mu K^2$]')
savefig('sens_qubic.png')

#### from Mikhail 
ells= array([  29.5,   49.5,   69.5,   89.5,  109.5,  129.5,  149.5,  169.5,
         189.5,  209.5,  229.5])
bb_std = array([ 0.00044772,  0.0011369 ,  0.00138991,  0.00267151,  0.00268465,
         0.00349815,  0.00413161,  0.0047278 ,  0.00580218,  0.00688165,
         0.00785394])

plot(ells,bb_std, 'r',lw=5)
savefig('sens_qubic_mod.png')


for i in xrange(len(lbins)):
	print('{0:3d} {1:9.6f} {2:9.6f} {3:9.6f}'.format(long(lbins[i]), errs[0,i], errs[1,i], errs[2,i]))


dllensing = spectrum*lll*(lll+1)/2/pi
rvalues = [0.01, 0.05, 0.1]
thecl = []
for i in xrange(len(rvalues)):
	thecl.append(ic.get_Dlbb_fromlib(lll, rvalues[i], camblib) - dllensing)


clf()
xlim(0,300)
ylim(0,0.03)
cols = ['b','g','r']
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell / 2\pi \,\,\,[\mu K^2]$')
plot(lll, dllensing,'k', lw=4,label ='BB Lensing')
plot(lll, fact*cl150x150[2],'k:',lw=4, label ='Dust 150 GHz')
for i in xrange(len(rvalues)):
	plot(lll, thecl[i],'--',lw=4, label ='BB r={}'.format(rvalues[i]), color=cols[i])
	plot(lll, thecl[i]+dllensing+fact*cl150x150[2],lw=4, color=cols[i])
legend(loc='upper right')
savefig('cl_cmb_big.png')

