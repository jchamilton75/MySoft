from qubic import QubicScene, QubicInstrument
from SynthBeam import myinstrument
import healpy as hp


###### First Instrument
nside = 512
scene = QubicScene(nside)
inst = myinstrument.QubicInstrument(filter_nu=150e9)



#### Detecteurs
clf()
inst.detector.plot()
plot(inst.detector[231].center[0,0], inst.detector[231].center[0,1],'ro')
plot(inst.detector[27].center[0,0], inst.detector[27].center[0,1],'ro')


### horns
clf()
inst.horn.plot()
centers = inst.horn.center[:,0:2]
col = inst.horn.column
row = inst.horn.row
for i in xrange(len(centers)):
    text(centers[i,0]-0.006, centers[i,1], 'c{0:}'.format(col[i]), color='r',fontsize=6)
    text(centers[i,0]+0.001, centers[i,1], 'r{0:}'.format(row[i]), color='b',fontsize=6)


### Horns for First Instrument
hornsFI = inst.horn.open

### Horns for TD
hornsTD = (col >= 8) & (col <= 15) & (row >= 8) & (row <= 15)


#### Now create First Instrument and TD monochromatic
instFI = myinstrument.QubicInstrument(filter_nu=150e9)
instFI.horn.open[~hornsFI] = False

instTD = myinstrument.QubicInstrument(filter_nu=150e9)
instTD.horn.open[~hornsTD] = False

### Now Synthesized beams for a detector (chosen close to center)
idet = 231
sbidealFI = instFI[idet].get_synthbeam(scene)[0]
sbidealTD = instTD[idet].get_synthbeam(scene)[0]

clf()
mini=-30
hp.gnomview(np.log10(sbidealFI/np.max(sbidealFI))*10, rot=[0,90], reso=5, 
	sub=(1,2,1), title='First Instrument', min=mini, max=0)
hp.gnomview(np.log10(sbidealTD/np.max(sbidealTD))*10, rot=[0,90], reso=5, 
	sub=(1,2,2), title='Technological Demonstrator', min=mini, max=0)


#### now integrate over pixel size with nsub x nsub sub pix
nsub = 4
idet = 231
sbidealFI = instFI[idet].get_synthbeam(scene, detector_integrate=nsub)[0]
sbidealTD = instTD[idet].get_synthbeam(scene, detector_integrate=nsub)[0]

figure()
clf()
mini=-30
hp.gnomview(np.log10(sbidealFI/np.max(sbidealFI))*10, rot=[0,90], reso=5, 
	sub=(1,2,1), title='First Instrument', min=mini, max=0)
hp.gnomview(np.log10(sbidealTD/np.max(sbidealTD))*10, rot=[0,90], reso=5, 
	sub=(1,2,2), title='Technological Demonstrator', min=mini, max=0)



#### Now build a function to integrate over bandwidth as well
def getsb(hornsOK, nu0, idet, dnu_nu=None, detector_integrate=None, nsubnus = 1, nside=256):
	scene = QubicScene(nside)
	sb = np.zeros(12*nside**2)
	if dnu_nu:
		numin = nu0*(1-dnu_nu/2)
		numax = nu0*(1+dnu_nu/2)
		nuvals = linspace(numin, numax, nsubnus)
		for i in xrange(nsubnus):
			print('nu={} number {} over {}'.format(nuvals[i], i, nsubnus))
			theinst = myinstrument.QubicInstrument(filter_nu=nuvals[i]*1e9)
			theinst.horn.open[~hornsOK] = False
			sb += theinst[idet].get_synthbeam(scene, detector_integrate=detector_integrate)[0]/nsubnus
	else:
		theinst = myinstrument.QubicInstrument(filter_nu=nu0*1e9)
		theinst.horn.open[~hornsOK] = False
		sb = theinst[idet].get_synthbeam(scene, detector_integrate=detector_integrate)[0]
	return sb

nsub = 4
idet = 231
nside = 256
sbidealFI_150 = getsb(hornsFI, 150., idet, dnu_nu=0.25, nsubnus=10, detector_integrate=nsub, nside=nside)
sbidealTD_150 = getsb(hornsTD, 150., idet, dnu_nu=0.25, nsubnus=10, detector_integrate=nsub, nside=nside)

clf()
reso = 7.
mini=-30
mapFI = hp.gnomview(np.log10(sbidealFI_150/np.max(sbidealFI_150))*10, rot=[0,90], reso=reso, 
	sub=(1,2,1), title='FI - 150 GHz - Det + Nu Integ.', min=mini, max=0, return_projected_map=True)
mapTD = hp.gnomview(np.log10(sbidealTD_150/np.max(sbidealTD_150))*10, rot=[0,90], reso=reso, 
	sub=(1,2,2), title='TD - 150 GHz - Det + Nu Integ.', min=mini, max=0, return_projected_map=True)

# location of maximum 
maxx, maxy = np.unravel_index(np.argmax(mapFI), dims=(200,200))
# diagonal cut of array shifted so that maximum is at center
cutFI = np.diag(np.roll(np.roll(mapFI, 99-maxx, axis=0), 99-maxy, axis=1))
cutTD = np.diag(np.roll(np.roll(mapTD, 99-maxx, axis=0), 99-maxy, axis=1))

xx = linspace(-100,100, 200)*reso*sqrt(2)/60
clf()
xlabel('Angle (deg)')
ylabel('Synthesized Beam (dB)')
plot(xx,cutFI, label = 'FI - 150 GHz - Det + Nu Integ.')
plot(xx,cutTD, label = 'TD - 150 GHz - Det + Nu Integ.')
legend(loc='lower right', fontsize=10)

#### Angular resolution
xx = linspace(-100,100, 200)*reso*sqrt(2)
halfmaxFI = cutFI > (np.log10(0.5)*10)
halfmaxTD = cutTD > (np.log10(0.5)*10)

fwhmFI = np.max(xx[halfmaxFI]) - np.min(xx[halfmaxFI])
fwhmTD = np.max(xx[halfmaxTD]) - np.min(xx[halfmaxTD])

clf()
xlabel('Angle (arcmin)')
ylabel('Synthesized Beam (dB)')
xlim(-60,60)
ylim(-10,0)
plot(xx,cutFI, label = 'FI - 150 GHz - Det + Nu Integ. - FWHM = {0:5.1f} arcmin'.format(fwhmFI))
plot(xx,cutTD, label = 'TD - 150 GHz - Det + Nu Integ. - FWHM = {0:5.1f} arcmin'.format(fwhmTD))
plot(xx, xx*0+np.log10(0.5)*10, 'k--')
legend(loc='lower right', fontsize=10)








