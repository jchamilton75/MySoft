import os
import glob
from pysimulators import FitsArray
import healpy as hp
from qubic import equ2gal
import sys
from Homogeneity import fitting

racenter = 0.0      # deg
deccenter = -57.0   # deg
center = equ2gal(racenter, deccenter)

directory = '/Users/hamilton/Qubic/SpectroImager/SimsMC/npts3000_subdelta_0.02_realisation_ZMhI69XD9D/'

def read_one_sim(directory):
	x0conv_file = glob.glob(directory+'x0convolved_*.fits')
	the_maps_files = glob.glob(directory+'OUTMAPS_*.fits')
	allrecmaps = []
	allresmaps = []
	allconvmaps =[ ]
	nsubs = []
	stdres = []
	stdtot = []
	stdtot2 = []
	stdstdtot2 = []
	for i in xrange(len(the_maps_files)):
		mapfile = the_maps_files[i]
		x0file = x0conv_file[i]
		maps = FitsArray(mapfile)
		x0_conv = FitsArray(x0file)
		obs = maps != 0
		x0_conv[~obs] = 0
		res = maps-x0_conv
		sh = shape(maps)
		nsubs.append(sh[0])
		allrecmaps.append(maps)
		allresmaps.append(res)
		allconvmaps.append(x0_conv)
		obs1 = res[0,:,0] != 0
		sigres = np.std(res[:,obs1,:], axis=1)
		stdres.append(sigres)
		sigtot = np.sqrt(np.sum(sigres**2, axis=0))/nsubs[i]
		stdtot.append(sigtot)
	stdtot = np.array(stdtot)
	nsubs=np.array(nsubs)
	return nsubs, stdres, stdtot



def get_averages(npts, subdelta):
	alldirs = glob.glob('/Users/hamilton/Qubic/SpectroImager/SimsMC/npts{0:}_subdelta_{1:}*'.format(npts, subdelta))
	allres = []
	for thedir in alldirs:
		print(thedir)
		files = glob.glob(thedir+'/*')
		print(len(files))
		if len(files) ==14:
			print('reading')
			allres.append(read_one_sim(thedir+'/'))

	nbmc = len(allres)
	nsubs = allres[0][0]

	allres_av = np.zeros((len(nsubs), 3))
	allres_std = np.zeros((len(nsubs), 3))
	allres_sum_av = np.zeros((len(nsubs), 3))
	allres_sum_std = np.zeros((len(nsubs), 3))
	for i in xrange(len(nsubs)):
		blo = np.zeros((nbmc, 3))
		bla = np.zeros((nbmc, nsubs[i], 3))
		for j in xrange(nbmc):
			blo[j,:] = allres[j][2][i]
			bla[j,:,:] = allres[j][1][i]
		allres_av[i,:] = np.mean(bla, axis=(0,1))
		allres_std[i,:] = np.std(bla, axis=(0,1))
		allres_sum_av[i,:] = np.mean(blo, axis=0)
		allres_sum_std[i,:] = np.std(blo, axis=0)
	return nsubs, allres_sum_av, allres_sum_std, allres_av, allres_std

allnpts = [3000, 6500, 10000]
subdelta = [0.02, 0.02, 0.02]

#allnpts = [3000, 3000, 3000]
#subdelta = [0.01, 0.02, 0.05]

res = []
for i in xrange(len(allnpts)):
	res.append(get_averages(allnpts[i], subdelta[i]))



def plaw(x,pars):
	return pars[0]*x**pars[1]

colors = ['r', 'g', 'b', 'y', 'm', 'k']


clf()
xlim(0.5,6)
ylim(0.5, 5)
iqu = 1
xx = linspace(0,6,100)
plot(xx, np.ones(100),'k--')
for i in xrange(len(allnpts)):
	nsubs = res[i][0]
	allresav = res[i][1]
	allresstd = res[i][2]
	resfit = fitting.dothefit(nsubs, allresav[:,iqu], allresstd[:,iqu], [1., 0.5], functname=plaw)
	plot(xx, plaw(xx,resfit[1]),colors[i])
	thepow=resfit[1][1]
	theerrpow = resfit[2][1]
	errorbar(nsubs, allresav[:,iqu], yerr=allresstd[:,iqu], 
		label='Q Npts:{0:} SubDelta:{1:} Power: {2:4.3f} +/- {3:4.3f}'.format(allnpts[i], subdelta[i], thepow, theerrpow), fmt='o',color=colors[i])
legend(loc='upper left', fontsize=10)
xlabel('Number of sub-bands')
ylabel('Summed RMS residuals ratio')
#xscale('log')
#yscale('log')

clf()
xlim(0.5,6)
ylim(0.5, 10)
iqu = 1
plot(xx, np.ones(100),'k--')
for i in xrange(len(allnpts)):
	nsubs = res[i][0]
	allresav_sum = res[i][3]
	allresstd_sum = res[i][4]
	resfit = fitting.dothefit(nsubs, allresav_sum[:,iqu], allresstd_sum[:,iqu], [1., 1.], functname=plaw)
	plot(xx, plaw(xx,resfit[1]),colors[i])
	thepow=resfit[1][1]
	theerrpow = resfit[2][1]
	errorbar(nsubs, allresav_sum[:,iqu], yerr=allresstd_sum[:,iqu], 
		label='Q Npts:{0:} SubDelta:{1:} Power: {2:4.3f} +/- {3:4.3f}'.format(allnpts[i], subdelta[i], thepow, theerrpow), fmt='o', color=colors[i])
legend(loc='upper left', fontsize=10)
xlabel('Number of sub-bands')
ylabel('RMS residuals')
#xscale('log')
#yscale('log')











