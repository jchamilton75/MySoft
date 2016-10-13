import os
import glob
from pysimulators import FitsArray
import healpy as hp
from qubic import equ2gal
import sys
import numpy as np
import string
import random

def random_string(nchars):
    lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(nchars)]
    str = "".join(lst)
    return(str)

def boolconv(thestr):
	if thestr == 'True':
		return True
	else:
		return False

### TOD Parameters
photon_noise = True
detector_nep = 0.
dnu_nu_all = 0.25
nu_center = 150.
effective_duration = 5e-28
subdelta_construction = np.float(sys.argv[4])
npts = np.int(np.float(sys.argv[1]))
ang = 20
build_TOD = boolconv(sys.argv[2])
dirinit = sys.argv[5]



if len(glob.glob(dirinit))==0:
	os.mkdir(dirinit)

### MAPS Parameters
subdelta_reconstruction = subdelta_construction
#nsubfreqs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
nsubfreqs = [1., 2., 3., 4., 5.]
nside = 256
build_maps=boolconv(sys.argv[3])

print('NPTS = {0:}'.format(npts))
print('BUILD TOD  = {0:}'.format(build_TOD))
print('BUILD MAPS = {0:}'.format(build_maps))



nu_min=nu_center*(1.-dnu_nu_all/2)
nu_max=nu_center*(1.+dnu_nu_all/2)

realisation_string = random_string(10)

thedirectory = dirinit+'/npts{0:}_subdelta_{1:}_realisation_{2:}/'.format(npts, 
	subdelta_reconstruction, realisation_string)
if len(glob.glob(thedirectory))==0: 
	os.mkdir(thedirectory)
os.chdir(thedirectory)


######### TOD Fabrication
tod_code = '~/Python/Qubic/SpectroImager/create_tod.py'
tod_command = 'python {0:} {1:} {2:} {3:} {4:} {5:} {6:} {7:} {8:}'.format(tod_code, photon_noise, detector_nep, 
	dnu_nu_all, nu_center, effective_duration, subdelta_construction, npts, ang)
sampling_file = 'SAMPLING_{0:}_{1:}_{2:}_{3:}_{4:}_{5:}_{6:}_{7:}.fits'.format(photon_noise, 
	detector_nep, dnu_nu_all, nu_center, effective_duration, subdelta_construction, npts, ang)
tod_file = 'TOD_{0:}_{1:}_{2:}_{3:}_{4:}_{5:}_{6:}_{7:}.fits'.format(photon_noise, 
	detector_nep, dnu_nu_all, nu_center, effective_duration, subdelta_construction, npts, ang)
x0conv_file = 'x0convolved_{0:}_{1:}_{2:}_{3:}_{4:}_{5:}_{6:}_{7:}.fits'.format(photon_noise, 
	detector_nep, dnu_nu_all, nu_center, effective_duration, subdelta_construction, npts, ang)

if build_TOD:
	os.system(tod_command)


######### Maps Fabrication
maps_code = '~/Python/Qubic/SpectroImager/make_multifreq_maps.py'
for i in xrange(len(nsubfreqs)):
	thensubfreq = nsubfreqs[i]
	maps_command = 'python {0:} {1:} {2:} {3:} {4:} {5:} {6:} {7:}'.format(maps_code, tod_file, sampling_file, 
		dnu_nu_all, thensubfreq, nu_center, subdelta_reconstruction, nside)
	if build_maps:
		os.system(maps_command)








