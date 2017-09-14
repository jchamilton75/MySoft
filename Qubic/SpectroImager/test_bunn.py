from qubic import *
import healpy as hp
from SynthBeam import myinstrument


reload(myinstrument)



nside = 256
scene = QubicScene(nside)
idet=231
detpos = np.array([[0,0,-0.3]])

##### Freqs definition
nucenter = 150
dnu_nu = 0.25
nsubnu = 25
super_res=2

filtnumin = nucenter*(1-dnu_nu/2)
filtnumax = nucenter*(1+dnu_nu/2)
nuvals = np.linspace(filtnumin,filtnumax, nsubnu+1)
numins = nuvals[0:-1]
numaxs = nuvals[1:]
nucenters = (numins+numaxs)/2

##### Sythesized beams calculation (with super resolution)
allsb = np.zeros((nsubnu, 12*nside**2))
for i in xrange(nsubnu):
	thenumin = numins[i]
	thenumax = numaxs[i]
	nus = np.linspace(thenumin, thenumax, super_res)
	for k in xrange(super_res):
		print(i,k)
		inst = myinstrument.QubicInstrument(filter_nu=nus[k]*1e9)
		allsb[i,:] += inst[idet].get_synthbeam(scene, detpos=detpos)[0]/super_res


i=0
hp.gnomview(np.log10(allsb[i,:]/np.max(allsb[i,:]))*10, rot=[0,90], reso=10,
            title=i, min=-40,max=0, unit='dB')


##### Harmonic transforms
lmax = 512
alms = np.zeros((nsubnu, (lmax+1)*(lmax+2)/2), dtype='complex')
for i in xrange(nsubnu):
	print(i)
	alms[i,:] = hp.map2alm(allsb[i,:], lmax=lmax)

##### The inverse covariance matrix
invcov = np.zeros((lmax+1, nsubnu, nsubnu))
for i in xrange(nsubnu):
	for j in xrange(nsubnu):
		print(i,j)
		invcov[:, i, j] = hp.alm2cl(alms[i,:], alms[j,:])

nn=5
vals = np.floor(np.linspace(0, 512, nn**2))
clf()
for i in xrange(nn**2):
	subplot(nn,nn,i+1)
	mat = invcov[vals[i],:,:]
	title('$\ell={0:} - cond={1:5.2g}$'.format(int(vals[i]), np.linalg.cond(mat)), fontsize=8)
	imshow(mat, interpolation='nearest', 
		extent=[filtnumin, filtnumax, filtnumin, filtnumax], origin='lower',
		vmin=0,vmax=3e8)
	colorbar()


########## Observing several ell values #########################################
all_ells = linspace(25, 511, 10)
allwidths = []
allerr_widths = []
for ellvalue in all_ells:
	#ellvalue = 500
	theinvcov = invcov[ellvalue+1,:,:]
	clf()
	imshow(theinvcov, interpolation='nearest', 
			extent=[filtnumin, filtnumax, filtnumin, filtnumax], origin='lower',vmin=0,vmax=np.max(theinvcov))
	colorbar()

	w, v = np.linalg.eigh(theinvcov)
	clf()
	yscale('log')
	plot(np.abs(w))
	plot(-w,'ro')

	### Truc de Schlegel et Bolton
	thresh = 1e-5
	neww = w.copy()
	neww[w<thresh] = 0

	Q=np.dot(np.dot(v, np.diag(np.sqrt(neww))), v.T)
	QQ = np.dot(Q,Q)
	clf()
	subplot(2,2,1)
	imshow(QQ, interpolation='nearest')
	colorbar()
	subplot(2,2,2)
	imshow(theinvcov, interpolation='nearest')
	colorbar()
	subplot(2,2,3)
	imshow(theinvcov-QQ, interpolation='nearest')
	colorbar()
	subplot(2,2,4)
	imshow((theinvcov-QQ)/theinvcov, interpolation='nearest')
	colorbar()

	s = np.sum(Q,axis=1)
	invCtilde = np.diag(s**2)
	R = np.zeros((len(s),len(s)))
	for i in xrange(len(s)):
		R[:,i] = Q[:,i]/s

	# RinvCtildeR = np.dot(np.dot(R.T, invCtilde), R)
	# clf()
	# subplot(2,2,1)
	# imshow(RinvCtildeR, interpolation='nearest')
	# colorbar()
	# subplot(2,2,2)
	# imshow(theinvcov, interpolation='nearest')
	# colorbar()
	# subplot(2,2,3)
	# imshow(theinvcov-RinvCtildeR, interpolation='nearest')
	# colorbar()
	# subplot(2,2,4)
	# imshow((theinvcov-RinvCtildeR)/theinvcov, interpolation='nearest')
	# colorbar()

	clf()
	mean_f = np.zeros(len(nucenters))
	sig_f = np.zeros(len(nucenters))
	for i in xrange(nsubnu):
		subplot(5,5,i+1)
		plot(nucenters, R[i,:],color='b')
		r = R[i,:]
		#r = exp(-0.5*(nucenters-150.3)**2/2.**2)
		rnorm = r/np.sum(r)
		mean_f[i] = np.sum(nucenters * rnorm)
		sig_f[i] = np.sqrt(np.sum((nucenters-mean_f[i])**2 * rnorm))

	allwidths.append(np.mean(sig_f))
	allerr_widths.append(np.std(sig_f))

clf()
errorbar(all_ells, np.array(allwidths)*2.35, yerr=np.array(allerr_widths), fmt='ro')
#ylim(0,10)
xlabel('Multipole')
ylabel('Average FWHM (over bandwidth) of recovered mode in GHz')

clf()
errorbar(all_ells, np.array(allwidths)/150.*2.35, yerr=np.array(allerr_widths)/150, fmt='ro')
#ylim(0,0.15)
xlabel('Multipole')
ylabel('Average FWHM (over bandwidth) of recovered mode in dnu/nu')
ylim(0, 0.25)
plot(all_ells, all_ells*0+dnu_nu/sqrt(12)*2.35,'k--', label=r'$\Delta \nu/ \nu / \sqrt{12}$')
legend()
###########################################################################
  


import scipy.special
thetavals = np.linspace(0,90,91)

lvals = arange(lmax+1)
invcovtheta = np.zeros((len(thetavals), len(nucenters), len(nucenters)))
for il in xrange(len(lvals)):
	print(lvals[il])
	pl = scipy.special.legendre(lvals[il])
	for jth in xrange(len(thetavals)):
		invcovtheta[jth,:,:] += invcov[il,:,:] * np.nan_to_num(pl(cos(np.radians(thetavals[jth]))))

########## Observing several ell values #########################################
all_th = linspace(0, 0.1, 10)
allwidths = []
allerr_widths = []
for ith in xrange(len(all_th)):
	#ith = 5
	theinvcov = invcovtheta[ith,:,:]
	clf()
	imshow(theinvcov, interpolation='nearest', 
			extent=[filtnumin, filtnumax, filtnumin, filtnumax], origin='lower',vmin=0,vmax=np.max(theinvcov))
	colorbar()
	title('Theta = '+str(all_th[ith]))

	w, v = np.linalg.eigh(theinvcov)
	clf()
	yscale('log')
	plot(np.abs(w))
	plot(-w,'ro')

	### Truc de Schlegel et Bolton
	thresh = 1e-5
	neww = w.copy()
	neww[w<thresh] = 0

	Q=np.dot(np.dot(v, np.diag(np.sqrt(neww))), v.T)
	QQ = np.dot(Q,Q)
	clf()
	subplot(2,2,1)
	imshow(QQ, interpolation='nearest')
	colorbar()
	subplot(2,2,2)
	imshow(theinvcov, interpolation='nearest')
	colorbar()
	subplot(2,2,3)
	imshow(theinvcov-QQ, interpolation='nearest')
	colorbar()
	subplot(2,2,4)
	imshow((theinvcov-QQ)/theinvcov, interpolation='nearest')
	colorbar()

	s = np.sum(Q,axis=1)
	invCtilde = np.diag(s**2)
	R = np.zeros((len(s),len(s)))
	for i in xrange(len(s)):
		R[:,i] = Q[:,i]/s

	# RinvCtildeR = np.dot(np.dot(R.T, invCtilde), R)
	# clf()
	# subplot(2,2,1)
	# imshow(RinvCtildeR, interpolation='nearest')
	# colorbar()
	# subplot(2,2,2)
	# imshow(theinvcov, interpolation='nearest')
	# colorbar()
	# subplot(2,2,3)
	# imshow(theinvcov-RinvCtildeR, interpolation='nearest')
	# colorbar()
	# subplot(2,2,4)
	# imshow((theinvcov-RinvCtildeR)/theinvcov, interpolation='nearest')
	# colorbar()

	clf()
	mean_f = np.zeros(len(nucenters))
	sig_f = np.zeros(len(nucenters))
	for i in xrange(nsubnu):
		subplot(5,5,i+1)
		plot(nucenters, R[i,:],color='b')
		r = R[i,:]
		#r = exp(-0.5*(nucenters-150.3)**2/2.**2)
		rnorm = r/np.sum(r)
		mean_f[i] = np.sum(nucenters * rnorm)
		sig_f[i] = np.sqrt(np.sum((nucenters-mean_f[i])**2 * rnorm))

	allwidths.append(np.mean(sig_f))
	allerr_widths.append(np.std(sig_f))



clf()
errorbar(all_th, np.array(allwidths)/150.*2.35, yerr=np.array(allerr_widths)/150, fmt='ro')
xlabel('theta')
ylabel('Average FWHM (over bandwidth) of recovered mode in dnu/nu')
#ylim(0, 0.25)
plot(all_th, all_th*0+dnu_nu/sqrt(12)*2.35,'k--', label=r'$\Delta \nu/ \nu / \sqrt{12}$')
legend()



