import healpy as hp
import glob

dir = '/Users/hamilton/CMB/Interfero/ScanStrategy/WhiteNoise_NoSignal/'
angspeeds = [2.6]
delta_az = [15., 25., 35., 45.]



files = glob.glob(dir+'rec_map_angspeed{}_delta_az{}_realization*_hp.fits'.format(angspeeds[0],delta_az[0]))
covmap = hp.read_map(dir+'cov_map_coverage_angspeed{}_delta_az{}_hp.fits'.format(angspeeds[0],delta_az[0]))

maps = hp.read_map(files[0], field=[0,1,2])


############## Big problem here !!!!!
### first go through maps and find the pixels that are always touched
m = np.array(hp.read_map(files[0], field=[0,1,2]))
m = np.nan_to_num(m/m)
for f in files:
    print(f)
    mi = np.array(hp.read_map(f, field=[0,1,2]))
    m *= np.nan_to_num(mi/mi)
    
hp.mollview(m[0])
hp.mollview(m[1])
hp.mollview(m[2])

m = np.array(hp.read_map(files[0], field=[0,1,2]))
m = np.nan_to_num(m/m)

m1 = np.array(hp.read_map(files[1], field=[0,1,2]))
m1 = np.nan_to_num(m1/m1)


#### new maps
dir = '/Users/hamilton/CMB/Interfero/ScanStrategy/1dayNoise_nosignal/'
import qubic
mcov = qubic.io.read_map(dir+'cov_map_coverage_angspeed2.6_delta_az35.0.fits')
m0 = qubic.io.read_map(dir+'rec_map_angspeed2.6_delta_az35.0_realization0.fits')
m1 = qubic.io.read_map(dir+'rec_map_angspeed2.6_delta_az35.0_realization1.fits')

bla = np.nan_to_num(mcov / mcov)
bla0 = np.nan_to_num(m0[:,0] / m0[:,0])
bla1 = np.nan_to_num(m1[:,0] / m1[:,0])
#######################################################################


### calculate covariance matrix
maskok = covmap != 0
npixok = np.sum(maskok)

mapsi = np.zeros((len(maskok)))
