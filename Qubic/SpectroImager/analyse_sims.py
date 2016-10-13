
import os
import glob
from pysimulators import FitsArray
import healpy as hp
from qubic import equ2gal
import sys
from Homogeneity import fitting



directory = '/Users/hamilton/Qubic/SpectroImager/Sims/npts3000/'
x0conv_file = glob.glob(directory+'x0convolved_*.fits')
the_maps_files = glob.glob(directory+'OUTMAPS_*.fits')

#x0conv_file = x0conv_file[1:]

#### Analyse data
racenter = 0.0      # deg
deccenter = -57.0   # deg
center = equ2gal(racenter, deccenter)


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
	sigtot2 = np.mean(sigres,axis=0)/np.sqrt(nsubs[i])
	sigsigtot2 = np.std(sigres,axis=0)/np.sqrt(nsubs[i])/np.sqrt(nsubs[i])
	stdtot2.append(sigtot2)
	stdstdtot2.append(sigsigtot2)

stdtot = np.array(stdtot)
stdtot2 = np.array(stdtot2)
stdstdtot2 = np.array(stdstdtot2)
nsubs=np.array(nsubs)

def plaw(x,pars):
	return pars[0]*x**pars[1]

res = fitting.dothefit(nsubs, stdtot2[:,1]/stdtot2[-1,1], stdstdtot2[:,1]/stdtot2[-1,1]+1e-3, [1., 0.5], functname=plaw)

clf()
xscale('log')
yscale('log')
plot(nsubs, np.sqrt(nsubs),'k--', label='$\sqrt{N_b}$')
ylim(0.9,np.sqrt(10)*2)
xlim(np.min(nsubs)*0.9, np.max(nsubs)*1.1)
errorbar(nsubs, stdtot2[:,0]/stdtot2[-1,0], yerr=stdstdtot2[:,0]/stdtot2[-1,0], fmt='bo', label ='I')
errorbar(nsubs, stdtot2[:,1]/stdtot2[-1,1], yerr=stdstdtot2[:,1]/stdtot2[-1,1], fmt='ro', label ='Q')
errorbar(nsubs, stdtot2[:,2]/stdtot2[-1,2], yerr=stdstdtot2[:,2]/stdtot2[-1,2], fmt='go', label ='U')
thepow=res[1][1]
therrpow = res[2][1]
plot(nsubs, np.array(nsubs)**thepow,'m--', label='Power: {0:4.3f} +/- {1:4.3f}'.format(thepow, therrpow))
legend(loc='upper left')


# 3000  0.02 0.655 +/- 0.012
# 6500  0.02 0.631 +/- 0.014
# 10000 0.02 0.629 +/- 0.011
# 3001 0.01 0.725 +/- 0.017


clf()
xlim(np.min(nsubs)*0.9, np.max(nsubs)*1.1)
errorbar(nsubs, stdtot2[:,0]/stdtot2[-1,0]/np.sqrt(nsubs), yerr=stdstdtot2[:,0]/stdtot2[-1,0]/np.sqrt(nsubs), fmt='bo', label ='I')
errorbar(nsubs, stdtot2[:,1]/stdtot2[-1,1]/np.sqrt(nsubs), yerr=stdstdtot2[:,1]/stdtot2[-1,1]/np.sqrt(nsubs), fmt='ro', label ='Q')
errorbar(nsubs, stdtot2[:,2]/stdtot2[-1,2]/np.sqrt(nsubs), yerr=stdstdtot2[:,2]/stdtot2[-1,2]/np.sqrt(nsubs), fmt='go', label ='U')
legend(loc='upper left')
ylim(0,2)







i=6
clf()
reso=20
mm = 6.
for j in xrange(int(nsubs[i])):
	hp.gnomview(allconvmaps[i][j,:,1], rot=center, reso=reso,title='Input', sub=(nsubs[i],3, 3*j+1), fig=1,min=-mm,max=mm)
	hp.gnomview(allrecmaps[i][j,:,1], rot=center, reso=reso, title='Reconstructed', sub=(nsubs[i],3, 3*j+2), fig=1,min=-mm,max=mm)
	hp.gnomview(allresmaps[i][j,:,1], rot=center, reso=reso, title='Residuals', sub=(nsubs[i],3, 3*j+3),fig=1,min=-mm,max=mm)





