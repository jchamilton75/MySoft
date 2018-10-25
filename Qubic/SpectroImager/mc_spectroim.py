from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
from qubic import (create_random_pointings, gal2equ,
                  read_spectra,
                  compute_freq,
                  QubicScene,
                  QubicMultibandInstrument,
                  QubicMultibandAcquisition,
                  PlanckAcquisition,
                  QubicMultibandPlanckAcquisition)
import qubic
from SpectroImager import SpectroImLib as si


######## Default configuration
### Sky 
nside = 256
center_gal = 0, 90
center = gal2equ(center_gal[0], center_gal[1])
dust_coeff = 1.39e-2

### Detectors (for now using random pointing)
band = 150
relative_bandwidth = 0.25
sz_ptg = 10.
nb_ptg = 1000
effective_duration = 2.
ripples = False


### Mapmaking
tol = 1e-3

### Number of sub-bands to build the TOD
nf_sub_build = 5
nf_sub_rec = 2

parameters = {'nside':nside, 'center':center, 'dust_coeff': dust_coeff, 
				'band':band, 'relative_bandwidth':relative_bandwidth,
				'sz_ptg':sz_ptg, 'nb_ptg':nb_ptg, 'effective_duration':effective_duration, 
				'tol': tol, 'ripples':ripples,
				'nf_sub_build':nf_sub_build, 
				'nf_sub_rec': nf_sub_rec }




reload(si)

### Input maps
x0 = si.create_input_sky(parameters)

### Pointing
p = si.create_random_pointings(parameters['center'], parameters['nb_ptg'], parameters['sz_ptg'])

### TOD
TOD = si.create_TOD(parameters, p, x0)

### reconstruction
maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, parameters, p, x0=x0)


### Select good coverage pixels
cov = np.sum(cov, axis=0)
maxcov = np.max(cov)
unseen = cov < maxcov*0.1

### Residuals
diffmap = maps_convolved - maps_recon
maps_convolved[:,unseen,:] = hp.UNSEEN
maps_recon[:,unseen,:] = hp.UNSEEN
diffmap[:,unseen,:] = hp.UNSEEN
rms = np.std(diffmap[:,~unseen,:], axis = 1)

### Display
_max = [300, 5, 5]
for iband, (inp, rec, diff) in enumerate(zip(maps_convolved, maps_recon, diffmap)):
    mp.figure(iband + 1)
    for i, (inp_, rec_, diff_, iqu) in enumerate(zip(inp.T, rec.T, diff.T, 'IQU')):
        hp.gnomview(inp_, rot=center_gal, reso=5, xsize=700, fig=1,
            sub=(3, 3, i + 1), min=-_max[i], max=_max[i], title='Input convolved, {}, {:.0f} GHz'.format(iqu, nus[iband]))
        hp.gnomview(rec_, rot=center_gal, reso=5, xsize=700, fig=1,
            sub=(3, 3, i + 4), min=-_max[i], max=_max[i], title='Recon, {}, {:.0f} GHz'.format(iqu, nus[iband]))
        hp.gnomview(diff_, rot=center_gal, reso=5, xsize=700, fig=1,
            sub=(3, 3, i+7), min=-_max[i], max=_max[i], title='Diff, {}, {:.0f} GHz'.format(iqu, nus[iband]))

mp.show()






##### Monte Carlo
# from pysimulators import FitsArray
# nbmc = 1
# parameters['tol']=1e-2
# parameters['nf_sub_build']=2

# nbfreqs = 2
# freqs = linspace(150, 250,nbfreqs)

# subs = arange(1,3)
# nbsubs = len(subs)

from pysimulators import FitsArray
nbmc = 10
parameters['tol']=1e-4
parameters['nf_sub_build']=15

nbfreqs = 4
freqs = linspace(150, 225,nbfreqs)

subs = arange(1,6)
nbsubs = len(subs)


allrms = []
allnuvals = []
allnuedgevals = []
allindexes = []

for ifreq in xrange(nbfreqs):
	for jsubs in xrange(nbsubs):
		for kmc in xrange(nbmc):
			### Update parameters
			parameters['band'] = freqs[ifreq]
			parameters['nf_sub_rec'] = subs[jsubs]
			print('')
			print('')
			print('')
			print('################################################################################')
			print('Now doing Realiztion {0:3d} / {1:3d} with nu {2:3d} / {3:3d} and subs {4:3d} / {5:3d}'.format(kmc, nbmc, ifreq, nbfreqs, jsubs, nbsubs))
			print(' Central Frequency : {0:6.2f}'.format(freqs[ifreq]))
			print(' Nbsubs : {0:5.1f}'.format(subs[jsubs]))
			### Do realization
			x0 = si.create_input_sky(parameters)
			p = si.create_random_pointings(parameters['center'], parameters['nb_ptg'], parameters['sz_ptg'])
			TOD = si.create_TOD(parameters, p, x0)
			maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, parameters, p, x0=x0)
			if subs[jsubs]==1: maps_recon=np.reshape(maps_recon, np.shape(maps_convolved))
			cov = np.sum(cov, axis=0)
			maxcov = np.max(cov)
			unseen = cov < maxcov*0.1
			diffmap = maps_convolved - maps_recon
			maps_convolved[:,unseen,:] = hp.UNSEEN
			maps_recon[:,unseen,:] = hp.UNSEEN
			diffmap[:,unseen,:] = hp.UNSEEN
			therms = np.std(diffmap[:,~unseen,:], axis = 1)
			### fill results
			allrms.append(therms)
			allnuvals.append(nus)
			allnuedgevals.append(nus_edge)
			allindexes.append([ifreq, jsubs, kmc])
			FitsArray(therms).save('MC/rms_mc{0:}_sub{1:}_freq{2:}.fits'.format(kmc, jsubs, ifreq))
			FitsArray(nus).save('MC/nus_mc{0:}_sub{1:}_freq{2:}.fits'.format(kmc, jsubs, ifreq))
			FitsArray(nus_edge).save('MC/edges_mc{0:}_sub{1:}_freq{2:}.fits'.format(kmc, jsubs, ifreq))
			### Delete stuff
			X0=0
			p=0
			TOD=0
			maps_recon=0
			cov=0
			maps_convolved =0
			diffmap=0



#### Read Simulation
nbmc = 10
nbsubs = 5
nbfreqs = 4  ### MC was ended accidentally (abort trap ???)

all_rms_sum = np.zeros((nbfreqs, nbsubs, 3))
all_stdrms_sum = np.zeros((nbfreqs, nbsubs, 3))
all_nus = []
all_nus_edges = []

rep = 'MC/'
for ifreq in xrange(nbfreqs):
	nus = []
	nus_edges = []
	for jsubs in xrange(nbsubs):
		rms_sum_mc = np.zeros((nbmc, 3))
		sumstdrms_sum = np.zeros((nbmc,3))
		subnus = np.zeros((subs[jsubs]))
		subnus_edges = np.zeros((subs[jsubs]+1))
		for kmc in xrange(nbmc):
			### Update parameters
			parameters['band'] = freqs[ifreq]
			parameters['nf_sub_rec'] = subs[jsubs]
			print('')
			print('')
			print('')
			print('################################################################################')
			print('Now doing Realiztion {0:3d} / {1:3d} with nu {2:3d} / {3:3d} and subs {4:3d} / {5:3d}'.format(kmc, nbmc, ifreq, nbfreqs, jsubs, nbsubs))
			print(' Central Frequency : {0:6.2f}'.format(freqs[ifreq]))
			print(' Nbsubs : {0:5.1f}'.format(subs[jsubs]))
			name = 'mc{0:}_sub{1:}_freq{2:}.fits'.format(kmc, jsubs, ifreq)
			therms = FitsArray(rep+'rms_'+name)
			therms_sum = np.sqrt(np.sum(therms**2, axis=0))/subs[jsubs]
			thenus = FitsArray(rep+'nus_'+name)
			thenus_edges = FitsArray(rep+'edges_'+name)
			print(thenus)
			print(thenus_edges)
			print(therms_sum)
			rms_sum_mc[kmc,:] = therms_sum
			subnus = thenus
			subnus_edges = thenus_edges
		all_rms_sum[ifreq, jsubs, :] = np.mean(rms_sum_mc, axis=0)
		all_stdrms_sum[ifreq, jsubs, :] = np.std(rms_sum_mc, axis=0)
		nus.append(subnus)
		nus_edges.append(subnus_edges)
	all_nus.append(nus)
	all_nus_edges.append(nus_edges)


clf()
xlim(0,6)
xlabel('Number of Sub-Frequencies')
ylabel('Quadratic Averaged noise on maps')
for i in xrange(nbfreqs):
	errorbar(subs, all_rms_sum[i,:,1],yerr=all_stdrms_sum[i,:,1]/sqrt(nbmc), fmt='-o', label=r'$\nu_0={0:3.0f}$ GHz'.format(freqs[i]))
legend(loc='bottom right')



############### Run MC outside
command = 'python /Users/hamilton/Python/Qubic/SpectroImager/script_mc_si.py '

import random
import string
def random_string(nchars):
    lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(nchars)]
    str = "".join(lst)
    return(str)


nbmc = 40
freq = 150
subs = arange(1,5)
nbsubs = len(subs)

logmtol = 5
nfb = 15
tol = 10**(-logmtol)
repname = 'MC_Freqs_tol{0}_nfb{1}'.format(logmtol, nfb)
os.system('mkdir /Volumes/Data/Qubic/SpectroImager/AllMc/'+repname)

for kmc in xrange(nbmc):
	for jsubs in xrange(nbsubs):
		### Update parameters
		print('')
		print('')
		print('')
		print('################################################################################')
		print('Now doing: MC {0:5d} for nu {1:3d} and subs {2:3d} / {3:3d}'.format(kmc, freq, jsubs, nbsubs))
		print(' Central Frequency : {0:6.2f}'.format(freq))
		print(' Nbsubs : {0:5.1f}'.format(subs[jsubs]))
		rnd_str = random_string(10)
		cst_str = '/Volumes/Data/Qubic/SpectroImager/AllMc/MC_Freqs_tol{0}_nfb{1}/sub_{2:d}_freq_{3:d}'.format(logmtol, nfb, subs[jsubs], freq)
		all_str = cst_str + '_' + rnd_str
		allcommand = command + all_str + ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(
			'band', freq, 'nf_sub_rec', subs[jsubs], 'nf_sub_build', nfb, 'tol', tol)
		os.system(allcommand)


#### Example:
### python script_mc_si.py basefilename_output paramname0 val0 paramname1 val1 ...
###
### avec : basefilename output du genre: directory/toto2874562
### -> 4 fichiers directory/toto2874562_quelquechose
### FitsArray(therms).save(name+'_rms.fits')
### FitsArray(nus).save(name+'_nus.fits')
### FitsArray(nus_edge).save(name+'_nus_edges.fits')
### FitsArray(diffmap).save(name+'_diffmaps.fits')
### example complet:
### python script_mc_si.py dir/toto band 150 nf_sub_rec 2 nf_sub_build 7 tol 1e-5
### -> instrument centré a 150 GHz
### -> TOD construits avec 7 sous bandes pour produire la large bande du filtre
### -> reprojection des TOD sur 2 bandes de fréquences -> fichier _diffmaps va contenir 2x3 cartes (2 bandes et IQU)
### -> tolerance du mapmaking: 1e-5




#### Read New Simulation
import glob
from pysimulators import FitsArray
subs = arange(1,5)
nbsubs = len(subs)
freqs = [150, 175, 200, 225] 
nbfreqs = len(freqs) 

all_rms_sum = np.zeros((nbfreqs, nbsubs, 3))
all_stdrms_sum = np.zeros((nbfreqs, nbsubs, 3))
all_nus = []
all_nus_edges = []

cols = ['blue', 'red', 'green', 'magenta']
fig1 = figure()
clf()

rep = '/Volumes/Data/Qubic/SpectroImager/AllMc/MC_Freqs_tol4_nfb15/'
for ifreq in xrange(nbfreqs):
	nus = []
	nus_edges = []
	for jsubs in xrange(nbsubs):
		subnus = np.zeros((subs[jsubs]))
		subnus_edges = np.zeros((subs[jsubs]+1))
		cst_str = rep + '/sub_{0:d}_freq_{1:d}'.format(subs[jsubs], freqs[ifreq])
		files = glob.glob(cst_str+'*rms.fits')
		filesnus = glob.glob(cst_str+'*nus.fits')
		filesnus_edges = glob.glob(cst_str+'*nus_edges.fits')
		nbmc = len(files)
		rms_sum_mc = np.zeros((nbmc, 3))
		sumstdrms_sum = np.zeros((nbmc,3))
		print('')
		print('')
		print('')
		print('################################################################################')
		print('Now doing nu {0:3d} / {1:3d} and subs {2:3d} / {3:3d}'.format(ifreq, nbfreqs, jsubs, nbsubs))
		print(' Central Frequency : {0:6.2f}'.format(freqs[ifreq]))
		print(' Nbsubs : {0:5.1f}'.format(subs[jsubs]))
		print(' Nbmc : {0:5.1f}'.format(nbmc))
		for kmc in xrange(nbmc):
			# print('')
			# print('')
			# print('')
			# print('################################################################################')
			# print('Now doing Realiztion {0:3d} / {1:3d} with nu {2:3d} / {3:3d} and subs {4:3d} / {5:3d}'.format(kmc, nbmc, ifreq, nbfreqs, jsubs, nbsubs))
			# print(' Central Frequency : {0:6.2f}'.format(freqs[ifreq]))
			# print(' Nbsubs : {0:5.1f}'.format(subs[jsubs]))
			therms = FitsArray(files[kmc])
			therms_sum = np.sqrt(np.sum(therms**2, axis=0))/subs[jsubs]
			#thenus = FitsArray(filesnus[kmc])
			#thenus_edges = FitsArray(filesnus_edges[kmc])
			# print(thenus)
			# print(thenus_edges)
			#print(therms_sum)
			rms_sum_mc[kmc,:] = therms_sum
			subnus = thenus
			subnus_edges = thenus_edges
		subplot(4,4,ifreq+jsubs*nbfreqs+1)
		hist(rms_sum_mc[:,1],range=[0.5,1.5],bins=20,label=str(subs[jsubs]),alpha=0.2, color=cols[ifreq])
		title('nu={0:3d} subs {1:3d}'.format(freqs[ifreq], subs[jsubs]))
		fig1.canvas.draw()
		all_rms_sum[ifreq, jsubs, :] = np.mean(rms_sum_mc, axis=0)
		all_stdrms_sum[ifreq, jsubs, :] = np.std(rms_sum_mc, axis=0)/sqrt(nbmc)
		nus.append(subnus)
		nus_edges.append(subnus_edges)
	all_nus.append(nus)
	all_nus_edges.append(nus_edges)


clf()
xlim(0,6)
ylim(0.5, 1.5)
xlabel('Number of Sub-Frequencies')
ylabel('Quadratic Averaged noise on maps')
cols = ['blue', 'red', 'green', 'magenta']
for i in xrange(nbfreqs):
	errorbar(subs, all_rms_sum[i,:,1],yerr=all_stdrms_sum[i,:,1], fmt='-o', color=cols[i], label=r'$\nu_0={0:3.0f}$ GHz'.format(freqs[i]))
	plot(linspace(0,10,10), np.zeros(10)+all_rms_sum[i,0,1],'--',color=cols[i])
legend(loc='lower right')

 

clf()
xlim(0,5)
ylim(0.9, 1.5)
xlabel('Number of Sub-Frequencies')
ylabel('`relative Quadratic Averaged noise on maps')
cols = ['blue', 'red', 'green', 'magenta']
for i in xrange(nbfreqs):
	errorbar(subs, all_rms_sum[i,:,1]/all_rms_sum[i,0,1],yerr=all_stdrms_sum[i,:,1]/all_rms_sum[i,0,1], fmt='-o', color=cols[i], label=r'$\nu_0={0:3.0f}$ GHz'.format(freqs[i]))
	plot(linspace(0,10,10), np.zeros(10)+1,'--',color='k')
legend(loc='lower right')

 





#### Compare tol and input sub-freq number
def get_rms_sim(rep, name,dohist=False):
	files = glob.glob(rep + '/'+name+'*_rms.fits')
	nbmc = len(files)
	rms_sum_mc = np.zeros((nbmc, 3))
	sumstdrms_sum = np.zeros((nbmc,3))
	print('')
	print('')
	print('')
	print('################################################################################')
	print('Now doing '+rep+' files: '+name)
	print(' Nbmc : {0:5.1f}'.format(nbmc))
	for kmc in xrange(nbmc):
		therms = FitsArray(files[kmc])
		nbsubs = np.shape(therms)[0]
		#therms_sum = np.sqrt(np.sum(therms**2, axis=0))/int(nbsubs)
		therms_sum = np.mean(therms, axis=0)
		#print(therms)
		#print('')
		rms_sum_mc[kmc,:] = therms_sum
		#subnus = thenus
		#subnus_edges = thenus_edges
	mm = np.mean(rms_sum_mc, axis=0)
	ss = np.std(rms_sum_mc, axis=0)/sqrt(nbmc)
	if dohist: hist(rms_sum_mc[:,1],bins=20, range=[mm[1]-4*ss[1]*sqrt(nbmc), mm[1]+4*ss[1]*sqrt(nbmc)],alpha=0.2, label=name)
	return mm,ss, nbmc 


allrep = '/Users/hamilton/Qubic/SpectroImager/AllMC/'

subs = np.array(['1','2','3','4'])
allrms = np.zeros((3, len(subs)))
allrmserr = np.zeros((3, len(subs)))

#### Investigate PCG Tol parameter

clf()
xlim(0,5)
ylim(0.,2)
ff = '150'
title(ff)
allreps = ['MC_Freqs_tol3_nfb7','MC_Freqs_tol4_nfb7',  
		 'MC_Freqs_tol4.5_nfb7', 'MC_Freqs_tol5_nfb7', 'MC_Freqs_tol5.5_nfb7']
for rep in allreps:
	nn=[]
	for isub in xrange(len(subs)):
		a,b,n=get_rms_sim(allrep+rep,'sub_'+subs[isub]+'_freq_'+ff+'_')
		nn.append(n)
		allrms[:,isub]=a 
		allrmserr[:,isub] = b
	errorbar(arange(len(subs))+1, allrms[1,:]/allrms[1,0], 
		yerr=allrmserr[1,:]/allrms[1,0],fmt='o-', label=rep+': nbmc = {}'.format(np.mean(nn)))
legend(loc='upper left',frameon=False,fontsize=10)



clf()
xlim(0,5)
ylim(0.,3)
ff = '150'
title(ff)
allreps = ['MC_Freqs_tol3_nfb7','MC_Freqs_tol4_nfb7',  
		 'MC_Freqs_tol4.5_nfb7', 'MC_Freqs_tol5_nfb7', 'MC_Freqs_tol5.5_nfb7']
for rep in allreps:
	nn=[]
	for isub in xrange(len(subs)):
		a,b,n=get_rms_sim(allrep+rep,'sub_'+subs[isub]+'_freq_'+ff+'_')
		nn.append(n)
		allrms[:,isub]=a 
		allrmserr[:,isub] = b
	errorbar(arange(len(subs))+1, allrms[2,:], 
		yerr=allrmserr[2,:],fmt='o-', label=rep+': nbmc = {}'.format(np.mean(nn)))
legend(loc='upper left',frameon=False,fontsize=10)


#### Investigate number of subbands for construction

clf()
xlim(0,5)
ylim(0.1,0.4)
ff = '150'
title(ff)
allreps = ['MC_Freqs_tol5_nfb5','MC_Freqs_tol5_nfb7',  
		 'MC_Freqs_tol5_nfb11', 'MC_Freqs_tol5_nfb15']
for rep in allreps:
	nn=[]
	for isub in xrange(len(subs)):
		a,b,n=get_rms_sim(allrep+rep,'sub_'+subs[isub]+'_freq_'+ff+'_')
		nn.append(n)
		allrms[:,isub]=a 
		allrmserr[:,isub] = b
	errorbar(arange(len(subs))+1, allrms[1,:], 
		yerr=allrmserr[1,:],fmt='o-', label=rep+': nbmc = {}'.format(np.mean(nn)))
legend(loc='upper left',frameon=False,fontsize=10)


clf()
xlim(0,5)
ylim(0.9,3)
ff = '150'
title(ff)
allreps = ['MC_Freqs_tol5_nfb5','MC_Freqs_tol5_nfb7',  
		 'MC_Freqs_tol5_nfb11', 'MC_Freqs_tol5_nfb15']
for rep in allreps:
	nn=[]
	for isub in xrange(len(subs)):
		a,b,n=get_rms_sim(allrep+rep,'sub_'+subs[isub]+'_freq_'+ff+'_')
		nn.append(n)
		allrms[:,isub]=a 
		allrmserr[:,isub] = b
	errorbar(arange(len(subs))+1, allrms[1,:]/allrms[1,0], 
		yerr=allrmserr[1,:]/allrms[1,0],fmt='o-', label=rep+': nbmc = {}'.format(np.mean(nn)))
legend(loc='upper left',frameon=False,fontsize=10)



allrep = '/Users/hamilton/CMB/Interfero/SpectroImager/AllMC/'
subs = np.array(['1','2','3','4'])
allrms = np.zeros((3, len(subs)))
allrmserr = np.zeros((3, len(subs)))


clf()
#xlim(0,5)
#ylim(0.1,0.4)
ff = '150'
title(ff)
allreps = ['MC_Freqs_tol5_nfb7']
for rep in allreps:
	nn=[]
	for isub in xrange(len(subs)):
		a,b,n=get_rms_sim(allrep+rep,'sub_'+subs[isub]+'_freq_'+ff+'_')
		nn.append(n)
		allrms[:,isub]=a 
		allrmserr[:,isub] = b
	errorbar(arange(len(subs))+1, allrms[1,:], 
		yerr=allrmserr[1,:],fmt='o-', label=rep+': nbmc = {}'.format(np.mean(nn)))
legend(loc='upper left',frameon=False,fontsize=10)





#### Investigate number of pointings... might play a role ... hopefully not

clf()
xlim(0,5)
ylim(0.,0.6)
ff = '150'
title(ff)
allreps = ['MC_Freqs_tol5_nfb7','MC_Ptg2000_Freqs_tol5_nfb7','MC_Ptg4000_Freqs_tol5_nfb7']
for rep in allreps:
	nn=[]
	for isub in xrange(len(subs)):
		a,b,n=get_rms_sim(allrep+rep,'sub_'+subs[isub]+'_freq_'+ff+'_')
		nn.append(n)
		allrms[:,isub]=a 
		allrmserr[:,isub] = b
	errorbar(arange(len(subs))+1, allrms[1,:], 
		yerr=allrmserr[1,:],fmt='o-', label=rep+': nbmc = {}'.format(np.mean(nn)))
legend(loc='upper left',frameon=False,fontsize=10)


clf()
xlim(0,5)
ylim(0.9,3)
ff = '150'
title(ff)
allreps = ['MC_Freqs_tol5_nfb7','MC_Ptg2000_Freqs_tol5_nfb7','MC_Ptg4000_Freqs_tol5_nfb7']
for rep in allreps:
	nn=[]
	for isub in xrange(len(subs)):
		a,b,n=get_rms_sim(allrep+rep,'sub_'+subs[isub]+'_freq_'+ff+'_')
		nn.append(n)
		allrms[:,isub]=a 
		allrmserr[:,isub] = b
	errorbar(arange(len(subs))+1, allrms[1,:]/allrms[1,0], 
		yerr=allrmserr[1,:]/allrms[1,0],fmt='o-', label=rep+': nbmc = {}'.format(np.mean(nn)))
legend(loc='upper left',frameon=False,fontsize=10)




############### Run MC outside
command = 'python /Users/hamilton/Python/Qubic/SpectroImager/script_mc_si.py '
#allcommand = command + 'toto2' + ' {0} {1} {2} {3}'.format('tol', 1e-2, 'nf_sub_build', 3)
#os.system(allcommand)

import random
import string
def random_string(nchars):
    lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(nchars)]
    str = "".join(lst)
    return(str)


nbmc = 40
freq = 150
subs = arange(1,5)
nbsubs = len(subs)

logmtol = 5
nfb = 7
tol = 10**(-logmtol)
repname = 'MC_Freqs_tol{0}_nfb{1}'.format(logmtol, nfb)
os.system('mkdir /Volumes/Data/Qubic/SpectroImager/AllMc/'+repname)

for kmc in xrange(nbmc):
	for jsubs in xrange(nbsubs):
		### Update parameters
		print('')
		print('')
		print('')
		print('################################################################################')
		print('Now doing: MC {0:5d} for nu {1:3d} and subs {2:3d} / {3:3d}'.format(kmc, freq, jsubs, nbsubs))
		print(' Central Frequency : {0:6.2f}'.format(freq))
		print(' Nbsubs : {0:5.1f}'.format(subs[jsubs]))
		rnd_str = random_string(10)
		cst_str = '/Volumes/Data/Qubic/SpectroImager/AllMc/MC_Freqs_tol{0}_nfb{1}/sub_{2:d}_freq_{3:d}'.format(logmtol, nfb, subs[jsubs], freq)
		all_str = cst_str + '_' + rnd_str
		allcommand = command + all_str + ' {0} {1} {2} {3} {4} {5} {6} {7} '.format(
			'band', freq, 'nf_sub_rec', subs[jsubs], 'nf_sub_build', nfb, 'tol', tol)
		os.system(allcommand)




