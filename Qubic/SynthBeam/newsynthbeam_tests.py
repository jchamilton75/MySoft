from pylab import *
import numpy as np
import healpy as hp
from matplotlib import rc




from qubic import QubicScene
nside = 1024
scene = QubicScene(nside)
idet = 231
band = 150 
relative_bandwidth = 0.25


##### Monochromatic
from qubic import QubicInstrument
inst = QubicInstrument(filter_nu=band*1e9)
sbidealFI_mono = inst[idet].get_synthbeam(scene)[0]
sbidealFI_mono /= np.max(sbidealFI_mono)
mini=-30
map_mono = hp.gnomview(np.log10(sbidealFI_mono)*10, rot=[0,90], reso=5, 
	title='First Instrument', min=mini, max=0)


##### Integrate Monochromatic over frequency
def getsb(nu0, idet, dnu_nu=None, detector_integrate=None, nsubnus = 1, nside=256, nuvals=None):
	scene = QubicScene(nside)
	sb = np.zeros(12*nside**2)
	if dnu_nu:
		if nuvals is None:
			print('taking regularly space nus:')
			numin = nu0*(1-dnu_nu/2)
			numax = nu0*(1+dnu_nu/2)
			nuedge = linspace(numin, numax, nsubnus+1)
			nuvals = 0.5 * (nuedge[1:]+nuedge[:-1])
			print(nuvals)
		else:
			print('taking input nu values')
			print(nuvals)
		for i in xrange(nsubnus):
			print('nu={} number {} over {}'.format(nuvals[i], i, nsubnus))
			theinst = QubicInstrument(filter_nu=nuvals[i]*1e9)
			sb += theinst[idet].get_synthbeam(scene)[0]/nsubnus
	else:
		theinst = QubicInstrument(filter_nu=nu0*1e9)
		sb = theinst[idet].get_synthbeam(scene)[0]
	return sb

sbidealFI_mono_int = getsb(band, idet, dnu_nu=relative_bandwidth, nsubnus=10, nside=nside)
sbidealFI_mono_int /= np.max(sbidealFI_mono_int)

map_mono_int = hp.gnomview(np.log10(sbidealFI_mono_int)*10, rot=[0,90], reso=5, 
	title='First Instrument', min=mini, max=0)


######## MultiBandInstrument
from qubic import QubicMultibandInstrument
from qubic import compute_freq
# multiband  instrument, 15 bandes de frequences
Nf=10
Nbfreq_in, nus_edge_in, nus_in, deltas_in, Delta_in, Nbbands_in = compute_freq(band, relative_bandwidth, Nf)
q = QubicMultibandInstrument(filter_nus=nus_in * 1e9,
                             filter_relative_bandwidths=nus_in / deltas_in,
                             ripples=None) 
sbidealFI_multi = q.get_synthbeam(scene,idet)
sbidealFI_multi /= np.max(sbidealFI_multi)

map_multi = hp.gnomview(np.log10(sbidealFI_multi)*10, rot=[0,90], reso=5, 
	title='First Instrument', min=mini, max=0)



# multiband instrument, more bandes de frequences
Nfnew=Nf*10
Nbfreq_in, nus_edge_in, nus_in, deltas_in, Delta_in, Nbbands_in = compute_freq(band, relative_bandwidth, Nfnew)
qvery = QubicMultibandInstrument(filter_nus=nus_in * 1e9,
                             filter_relative_bandwidths=nus_in / deltas_in,
                             ripples=None) 
sbidealFI_multi_very = qvery.get_synthbeam(scene,idet)
sbidealFI_multi_very /= np.max(sbidealFI_multi_very)


map_multi_very = hp.gnomview(np.log10(sbidealFI_multi_very)*10, rot=[0,90], reso=5, 
	title='First Instrument', min=mini, max=0)



#### Comparison 2D
reso = 5
map_mono = hp.gnomview(np.log10(sbidealFI_mono)*10, rot=[0,90], reso=reso, 
	title='FI Mono', min=mini, max=0, sub=(2,2,1), return_projected_map=True)

map_mono_int = hp.gnomview(np.log10(sbidealFI_mono_int)*10, rot=[0,90], reso=reso, 
	title='FI Mono + Int(10)', min=mini, max=0, sub=(2,2,2), return_projected_map=True)

map_multi = hp.gnomview(np.log10(sbidealFI_multi)*10, rot=[0,90], reso=reso, 
	title='FI Multi 10', min=mini, max=0, sub=(2,2,3), return_projected_map=True)

map_multi_very = hp.gnomview(np.log10(sbidealFI_multi_very)*10, rot=[0,90], reso=reso, 
	title='FI Multi 100', min=mini, max=0, sub=(2,2,4), return_projected_map=True)

#### Comparison 1D
maxx, maxy = np.unravel_index(np.argmax(map_mono), dims=shape(map_mono))

cut_mono = np.diag(np.roll(np.roll(map_mono, shape(map_mono)[0]/2-maxx, axis=0), shape(map_mono)[1]/2-maxy, axis=1))
cut_mono_int = np.diag(np.roll(np.roll(map_mono_int, shape(map_mono)[0]/2-maxx, axis=0), shape(map_mono)[1]/2-maxy, axis=1))
cut_multi = np.diag(np.roll(np.roll(map_multi, shape(map_mono)[0]/2-maxx, axis=0), shape(map_mono)[1]/2-maxy, axis=1))
cut_multi_very = np.diag(np.roll(np.roll(map_multi_very, shape(map_mono)[0]/2-maxx, axis=0), shape(map_mono)[1]/2-maxy, axis=1))


xxinit = linspace(-100,100, len(cut_mono))*reso*sqrt(2)/60

clf()
plot(xxinit, cut_mono, label='FI Mono')
plot(xxinit, cut_mono_int, label='FI Mono Int(10)')
plot(xxinit, cut_multi, label='FI Multi(10)')
plot(xxinit, cut_multi_very, '--', label='FI Multi(100)')
xlim(0,12)
ylim(-45,0)
legend()
xlabel(r'$\theta$ [deg]')
ylabel('Synthesized Beam [dB]')
title('nside=1024')
savefig('sbnuall.pdf')


####### SO one needs to compare my integration with multiband for nf=10

sbidealFI_mono_int_0 = getsb(band, idet, dnu_nu=relative_bandwidth, nsubnus=10, nside=nside)
sbidealFI_mono_int_0 /= np.max(sbidealFI_mono_int_0)


Nf=10
Nbfreq_in, nus_edge_in, nus_in, deltas_in, Delta_in, Nbbands_in = compute_freq(band, relative_bandwidth, Nf)
sbidealFI_mono_int_1 = getsb(band, idet, dnu_nu=relative_bandwidth, nsubnus=10, nside=nside, nuvals = nus_in)
sbidealFI_mono_int_1 /= np.max(sbidealFI_mono_int_1)

clf()
map_mono_int_0 = hp.gnomview(np.log10(sbidealFI_mono_int_0)*10, rot=[0,90], reso=reso, 
	title='FI Mono + Int(10) not linear', min=mini, max=0, sub=(2,2,1), return_projected_map=True)

map_mono_int_1 = hp.gnomview(np.log10(sbidealFI_mono_int_1)*10, rot=[0,90], reso=reso, 
	title='FI Mono + Int(10) not linear', min=mini, max=0, sub=(2,2,2), return_projected_map=True)

map_multi = hp.gnomview(np.log10(sbidealFI_multi)*10, rot=[0,90], reso=reso, 
	title='FI Multi 10', min=mini, max=0, sub=(2,2,3), return_projected_map=True)


cut_mono_int_0 = np.diag(np.roll(np.roll(map_mono_int_0, shape(map_mono)[0]/2-maxx, axis=0), shape(map_mono)[1]/2-maxy, axis=1))
cut_mono_int_1 = np.diag(np.roll(np.roll(map_mono_int_1, shape(map_mono)[0]/2-maxx, axis=0), shape(map_mono)[1]/2-maxy, axis=1))


clf()
plot(xxinit, cut_mono_int_0, label='FI Mono Int(10) Linear')
plot(xxinit, cut_mono_int_1, label='FI Mono Int(10) not Linear')
plot(xxinit, cut_multi, '--',label='FI Multi(10) - not Linear')
plot(xxinit, cut_multi_very, '--', label='FI Multi(100)')
xlim(0,12)
ylim(-45,0)
legend()
xlabel(r'$\theta$ [deg]')
ylabel('Synthesized Beam [dB]')
title('nside=1024')





