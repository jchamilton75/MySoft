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
plot(detcenters[:,0], detcenters[:,1],'ro')
for i in xrange(992): plot(vertex[i,:,0], vertex[i,:,1], color='blue')



rep = '/Users/hamilton/Qubic/SynthBeam/NewSimsMaynooth/QUBIC Basic/'
files = glob.glob(rep+'*.dat')
# Image Plane @ 161x161
# X amplitude normed to X common peak
# Y amplitude normed to Y common peak
#X Amplitude     X Phase Y Amplitude     Y Phase

nn = 161
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




for i in xrange(len(files)):
    print(i)
    data = np.loadtxt(files[i], skiprows=4)
    allampX[i,:,:] = np.reshape(data[:,0],(nn,nn))
    allphiX[i,:,:] = np.reshape(data[:,1],(nn,nn))
    allampY[i,:,:] = np.reshape(data[:,2],(nn,nn))
    allphiY[i,:,:] = np.reshape(data[:,3],(nn,nn))


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



for i in xrange(len(files)):
    print(i)
    clf()
    imshow(allampX[i,:,:],vmin=0,vmax=1, extent = [xmin,xmax, ymin, ymax])
    plot(detcenters[:,0], detcenters[:,1],'ro')
    colorbar()
    draw()


Ax = allampX * (cos(allphiX) + 1j*sin(allphiX))
Ay = allampY * (cos(allphiY) + 1j*sin(allphiY))



#### With integration
external_A = [-xx, yy, allampX, allphiX]
reload(myinstrument)
scene = QubicScene(512)
inst = myinstrument.QubicInstrument(filter_nu=150e9)
#idet=231
idet=0
nint = 4
sbideal = inst[idet].get_synthbeam(scene)[0]
sbnew = inst[idet].get_synthbeam(scene, external_A=external_A)[0]
sbideal_int = inst[idet].get_synthbeam(scene, detector_integrate=nint)[0]
sbnew_int = inst[idet].get_synthbeam(scene, external_A=external_A, detector_integrate=nint)[0]

sbnew *= np.sum(np.nan_to_num(sbideal))/np.sum(np.nan_to_num(sbnew))
sbnew_int *= np.sum(np.nan_to_num(sbideal_int))/np.sum(np.nan_to_num(sbnew_int))

blm_ideal = hp.anafast(sbideal)
blm_sim = hp.anafast(sbnew)
blm_ideal_int = hp.anafast(sbideal_int)
blm_sim_int = hp.anafast(sbnew_int)

clf()
mini=-25
hp.gnomview(np.log10(sbideal_int/np.max(sbideal_int))*10, rot=[0,90], reso=5, sub=(2,2,1),
            title='Ideal - with Pix. Int.', min=mini,max=0, unit='dB')
hp.gnomview(np.log10(sbnew_int/np.max(sbnew_int))*10, rot=[0,90], reso=5, sub=(2,2,2),
            title='Sim - with Pix. Int.',min=mini, max=0, unit='dB')
subplot(2,1,2)
plot(blm_sim / blm_ideal, label='Sim / ideal')
plot(blm_sim_int / blm_ideal_int, label='Sim / ideal Int')
ylim(0.5,1.5)
plot(np.ones(len(blm_sim_int)),'k:')
xlim(0,300)
legend()

xx = linspace(0,199,200)*5.
map = hp.gnomview(sbideal_int/np.max(sbideal_int), rot=[0,90], reso=5, return_projected_map=True)
plot(xx,map[100,:])


clf()
hp.gnomview(sbideal, rot=[0,90], reso=5, sub=(2,2,1),title='Ideal - no Pix. Int.')
hp.gnomview(sbnew, rot=[0,90], reso=5, sub=(2,2,2),title='Sim - no Pix. Int.')
hp.gnomview(sbideal_int, rot=[0,90], reso=5, sub=(2,2,3),title='Ideal - with Pix. Int.')
hp.gnomview(sbnew_int, rot=[0,90], reso=5, sub=(2,2,4),title='Sim - with Pix. Int.')

clf()
hp.gnomview(np.log10(sbideal), rot=[0,90], reso=5, sub=(2,2,1),title='Ideal - no Pix. Int.')
hp.gnomview(np.log10(sbnew), rot=[0,90], reso=5, sub=(2,2,2),title='Sim - no Pix. Int.')
hp.gnomview(np.log10(sbideal_int), rot=[0,90], reso=5, sub=(2,2,3),title='Ideal - with Pix. Int.')
hp.gnomview(np.log10(sbnew_int), rot=[0,90], reso=5, sub=(2,2,4),title='Sim - with Pix. Int.')


clf()
hp.gnomview(np.log10(sbideal_int), rot=[0,90], reso=5, sub=(2,2,1),title='Ideal - with Pix. Int.')
hp.gnomview(np.log10(sbnew_int), rot=[0,90], reso=5, sub=(2,2,2),title='Sim - with Pix. Int.')
subplot(2,1,2)
plot(blm_ideal, label='ideal')
plot(blm_ideal_int, label='ideal Int')
plot(blm_sim, label='Sim')
plot(blm_sim_int, label='Sim Int')
legend()





