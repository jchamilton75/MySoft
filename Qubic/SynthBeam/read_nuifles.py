from qubic import QubicScene
from SynthBeam import myinstrument
import glob
import healpy as hp


scene = QubicScene(256)
inst = myinstrument.QubicInstrument(filter_nu=150e9)



### horns
clf()
inst.horn.plot()
centers = inst.horn.center[:,0:2]
col = inst.horn.column
row = inst.horn.row
for i in xrange(len(centers)):
    text(centers[i,0]-0.006, centers[i,1], 'c{0:}'.format(col[i]), color='r',fontsize=6)
    text(centers[i,0]+0.001, centers[i,1], 'r{0:}'.format(row[i]), color='b',fontsize=6)


### detectors
clf()
#inst.detector.plot()
vertex = inst.detector.vertex[..., :2]
detcenters = inst.detector.center[..., :2]
#plot(detcenters[:,0], detcenters[:,1],'ro')
for i in xrange(992): plot(vertex[i,:,0], vertex[i,:,1], color='blue')
plot(detcenters[231,0], detcenters[231,1],'ro')
savefig('detector_231.pdf')

rep = '/Users/hamilton/Qubic/SynthBeam/NewSimsMaynooth/QUBIC Basic/'
#rep = '/Users/hamilton/Qubic/SynthBeam/SimsMaynooth_OLD/GREGF300Xnorm/'
files = glob.glob(rep+'*.dat')
# Image Plane @ 161x161
# X amplitude normed to X common peak
# Y amplitude normed to Y common peak
#X Amplitude     X Phase Y Amplitude     Y Phase

nn = 161
#nn = 101
xmin = -60./1000
xmax = 60./1000
ymin = -60./1000
ymax = 60./1000
xx = np.linspace(-60,60,nn)/1000
yy = np.linspace(-60,60,nn)/1000

allampX = np.zeros((400,nn,nn))
allphiX = np.zeros((400,nn,nn))
allampY = np.zeros((400,nn,nn))
allphiY = np.zeros((400,nn,nn))



#### Read the files
for i in xrange(len(files)):
    print(i)
    data = np.loadtxt(files[i], skiprows=4)
    allampX[i,:,:] = np.reshape(data[:,0],(nn,nn))
    allphiX[i,:,:] = np.reshape(data[:,1],(nn,nn))
    allampY[i,:,:] = np.reshape(data[:,2],(nn,nn))
    allphiY[i,:,:] = np.reshape(data[:,3],(nn,nn))

#### Just some display, can be skipped
for i in xrange(len(files)):
    print(i)
    clf()
    subplot(2,2,1)
    imshow(allampX[i,:,:],vmin=0,vmax=1, extent = [xmin,xmax, ymin, ymax])
    colorbar()
    title(i)
    subplot(2,2,2)
    imshow(allphiX[i,:,:],vmin=-np.pi,vmax=np.pi, extent = [xmin,xmax, ymin, ymax])
    colorbar()
    subplot(2,2,3)
    imshow(allampY[i,:,:],vmin=0,vmax=1, extent = [xmin,xmax, ymin, ymax])
    colorbar()
    subplot(2,2,4)
    imshow(allphiY[i,:,:],vmin=-np.pi,vmax=np.pi, extent = [xmin,xmax, ymin, ymax])
    colorbar()
    draw()


#### Just some display, can be skipped
for i in xrange(len(files)):
    print(i)
    clf()
    imshow(allampX[i,:,:],vmin=0,vmax=1, extent = [xmin,xmax, ymin, ymax])
    plot(detcenters[:,0], detcenters[:,1],'ro')
    colorbar()
    draw()


for i in xrange(len(files)):
    print(i)
    clf()
    imshow(allphiX[i,:,:],vmin=-np.pi,vmax=np.pi, extent = [xmin,xmax, ymin, ymax])
    plot(detcenters[:,0], detcenters[:,1],'ro')
    colorbar()
    draw()



#### Electric field
Ax = allampX * (cos(allphiX) + 1j*sin(allphiX))
Ay = allampY * (cos(allphiY) + 1j*sin(allphiY))

#### Image in the FP
clf()
imshow(np.log10(np.abs(np.sum(Ax,axis=0))))
colorbar()
#### Image in the FP
clf()
imshow((np.abs(np.sum(Ax,axis=0))))
colorbar()

#### With integration
external_A = [-xx, yy, allampX, allphiX]
reload(myinstrument)
scene = QubicScene(512)
inst = myinstrument.QubicInstrument(filter_nu=150e9)



######## Créidhe's code
idet=231
sbideal = inst[idet].get_synthbeam(scene)[0]
sbnew = inst[idet].get_synthbeam(scene, external_A=external_A)[0]
blm_ideal = hp.anafast(sbideal)
blm_sim = hp.anafast(sbnew)

figure()
hp.gnomview(sbideal, rot=[0,90], reso=5, sub=(2,2,1),
    title='Ideal - No Pix. Int.')
hp.gnomview(sbnew, rot=[0,90], reso=5, sub=(2,2,2),
    title='Sim - No Pix. Int.')
subplot(2,2,3)
plot(blm_ideal, label='ideal')
xlim(0,300)
subplot(2,2,4)
plot(blm_sim, label='Sim')
xlim(0,300)


idet=231
sbideal = inst[idet].get_synthbeam(scene)[0]
sbideal = sbideal/np.max(sbideal)

sbnew = inst[idet].get_synthbeam(scene, external_A=external_A)[0]
sbnew *= np.sum(np.nan_to_num(sbideal))/np.sum(np.nan_to_num(sbnew))


blm_ideal = hp.anafast(sbideal)
blm_sim = hp.anafast(sbnew)

figure()
hp.gnomview(sbideal, rot=[0,90], reso=5, sub=(2,2,1),
    title='Ideal - No Pix. Int.')
hp.gnomview(sbnew, rot=[0,90], reso=5, sub=(2,2,2),
    title='Sim - No Pix. Int.')
subplot(2,1,2)
plot(blm_ideal, label='ideal')
plot(blm_sim, label='Sim')
legend()
xlim(0,300)






#### Which detector is chosen ?
#idet=231
idet=0

#### we use integration over bolometer square are: nint x nint sub-points are averaged out
nint = 4
sbideal = inst[idet].get_synthbeam(scene)[0]
sbnew = inst[idet].get_synthbeam(scene, external_A=external_A)[0]
sbideal_int = inst[idet].get_synthbeam(scene, detector_integrate=nint)[0]
sbnew_int = inst[idet].get_synthbeam(scene, external_A=external_A, detector_integrate=nint)[0]

#### Normalize
sbnew *= np.sum(np.nan_to_num(sbideal))/np.sum(np.nan_to_num(sbnew))
sbnew_int *= np.sum(np.nan_to_num(sbideal_int))/np.sum(np.nan_to_num(sbnew_int))


##### Compute window functions
blm_ideal = hp.anafast(sbideal)
blm_sim = hp.anafast(sbnew)
blm_ideal_int = hp.anafast(sbideal_int)
blm_sim_int = hp.anafast(sbnew_int)


hp.mollview(np.log10(sbideal_int/np.max(sbideal_int))*10, rot=[0,90],
            title='Ideal - with Pix. Int.', min=mini,max=0, unit='dB')


#### Display results
clf()
mini=-30
hp.gnomview(np.log10(sbideal_int/np.max(sbideal_int))*10, rot=[0,90], reso=5, sub=(2,2,1),
            title='Ideal - with Pix. Int.', min=mini,max=0, unit='dB')
hp.gnomview(np.log10(sbnew_int/np.max(sbnew_int))*10, rot=[0,90], reso=5, sub=(2,2,2),
            title='Sim - with Pix. Int.',min=mini, max=0, unit='dB')
subplot(2,1,2)
plot(blm_sim / blm_ideal, label='Sim / ideal')
plot(blm_sim_int / blm_ideal_int, label='Sim / ideal Int')
ylim(0.,1.2)
plot(np.ones(len(blm_sim_int)),'k:')
xlim(0,300)
legend(loc='lower left')
savefig('wf_pixel_231.pdf')


clf()
hp.gnomview(np.log10(sbideal/np.max(sbideal))*10, rot=[0,90], reso=5, sub=(2,2,1),title='Ideal - no Pix. Int.',min=mini, max=0, unit='dB')
hp.gnomview(np.log10(sbnew/np.max(sbnew))*10, rot=[0,90], reso=5, sub=(2,2,2),title='Sim - no Pix. Int.',min=mini, max=0, unit='dB')
hp.gnomview(np.log10(sbideal_int/np.max(sbideal_int))*10, rot=[0,90], reso=5, sub=(2,2,3),title='Ideal - with Pix. Int.',min=mini, max=0, unit='dB')
hp.gnomview(np.log10(sbnew_int/np.max(sbnew_int))*10, rot=[0,90], reso=5, sub=(2,2,4),title='Sim - with Pix. Int.',min=mini, max=0, unit='dB')
savefig('maps_pixel_231.pdf')

clf()
hp.gnomview(np.log10(sbideal_int), rot=[0,90], reso=5, sub=(2,2,1),title='Ideal - with Pix. Int.')
hp.gnomview(np.log10(sbnew_int), rot=[0,90], reso=5, sub=(2,2,2),title='Sim - with Pix. Int.')
subplot(2,1,2)
plot(blm_ideal, label='ideal')
plot(blm_ideal_int, label='ideal Int')
plot(blm_sim, label='Sim')
plot(blm_sim_int, label='Sim Int')
legend()





