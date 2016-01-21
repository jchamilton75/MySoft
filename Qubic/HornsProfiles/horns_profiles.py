from __future__ import division
from pylab import *
import healpy as hp
from matplotlib.pyplot import *
import numpy as np
import os
import sys
import qubic
import pycamb
import string
import random
from pyoperators import DenseBlockDiagonalOperator, Rotation3dOperator
from pysimulators import FitsArray
from pyoperators import MPI

from qubic import (
    QubicAcquisition, create_sweeping_pointings, equ2gal, map2tod, tod2map_all,
    tod2map_each, create_random_pointings, QubicInstrument)

rank = MPI.COMM_WORLD.rank


def random_string(nchars):
    lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(nchars)]
    str = "".join(lst)
    return(str)



def make_a_map(x0, pointing, instrument, nside, coverage_threshold=0.01, todnoise=None, fits_string=None, noiseless=False):
    ############# Make TODs ###########################################################
    acquisition = QubicAcquisition(instrument, pointing,
                                 nside=nside,
                                 synthbeam_fraction=0.99)
    tod, x0_convolved = map2tod(acquisition, x0, convolution=True)
    if todnoise is None:
        todnoise = acquisition.get_noise()
    factnoise=1
    if noiseless:
        factnoise=0
    ##################################################################################

    
    ############# Make mapss ###########################################################
    print('Making map')
    maps, cov = tod2map_all(acquisition, tod + todnoise * factnoise, tol=1e-4, coverage_threshold=coverage_threshold)
    if MPI.COMM_WORLD.rank == 0:
        fitsmapname = 'maps_'+fits_string+'.fits'
        fitscovname = 'cov_'+fits_string+'.fits'
        print('Saving the map: '+fitsmapname)
        qubic.io.write_map(fitsmapname,maps)
        print('Saving the coverage: '+fitsmapname)
        qubic.io.write_map(fitscovname,cov)
    ##################################################################################

    return maps, cov, todnoise
    
class AltPrimaryBeam(object):
    def __init__(self, fwhm_deg, colnum):
        data = np.loadtxt('Horn_beam_patterns.dat',skiprows=3)
        self.th = data[:,0]
        mask = np.abs(self.th) > 90
        data[mask,1:] = np.log10(0)
        data[:,1] *= 1.09
        data[:,2] *= 0.75
        data[:,3] *= 0.98
        data[:,4] *= 1.13
        self.beam_db = data[:,colnum]
        self.beam = 10**(data[:,colnum]/10)
        abovehalf = self.beam >= 0.5
        self.fwhm_deg = np.max(self.th[abovehalf]) - np.min(self.th[abovehalf])
        #print(self.fwhm_deg)
        self.sigma = np.radians(self.fwhm_deg) / np.sqrt(8 * np.log(2))
        self.fwhm_sr = 2 * pi * self.sigma**2
    def __call__(self, theta):
        return 10**(np.interp(theta, np.radians(self.th), self.beam_db/10))



############# Input Power spectrum ##############################################
H0 = 67.04
omegab = 0.022032
omegac = 0.12038
h2 = (H0/100.)**2
scalar_amp = np.exp(3.098)/1.E10
omegav = h2 - omegab - omegac
Omegab = omegab/h2
Omegac = omegac/h2
print 'Omegab = ',omegab/h2,'Omegam = ',(omegac+omegab)/h2,'Omegav = ',omegav/h2

params = {'H0':H0,'omegab':Omegab,'omegac':Omegac,'omegak':0,'scalar_index':0.9624,
         'reion__use_optical_depth':True,'reion__optical_depth':0.0925,
         'tensor_ratio':0.1,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':True}

lmaxcamb = 3*512
T,E,B,X = pycamb.camb(lmaxcamb+1,**params)
lll = np.arange(1,lmaxcamb+1)
fact = (lll*(lll+1))/(2*np.pi)
spectra = [lll, T/fact, E/fact, B/fact, X/fact]
##################################################################################





############# Reading arguments ################################################
print 'Number of arguments:', len(sys.argv), 'arguments.'
if len(sys.argv) > 1:
    print 'Argument List:', str(sys.argv)
    noise = np.float(sys.argv[1])
    nside = np.int(sys.argv[2])
    ts = np.float(sys.argv[3])
    hornprofile = np.int(sys.argv[4])
else:
    ts = 30.
    noise = 1.
    nside = 256
    hornprofile = 0

print('Noise level is set to '+np.str(noise))
print('nside is set to '+np.str(nside))
print('ts is set to '+np.str(ts))
print('Selected Horn profile '+np.str(hornprofile))

hprofiles = ['GaussInit', 'Clover', 'Conical', 'Gaussian', 'HybridConical']
##################################################################################




############# Parameters ##########################################################
racenter = 0.0      # deg
deccenter = -57.0   # deg
center = equ2gal(racenter, deccenter)
duration = 24       # hours
ts = 30.            # seconds
ang = 20            # degrees
##################################################################################


############## True Pointing #####################################################
pointing = create_random_pointings([racenter, deccenter], duration*3600/ts, ang, period=ts)
hwp_angles = np.random.random_integers(0, 7, len(pointing)) * 11.25 
pointing.pitch = 0
pointing.angle_hwp = hwp_angles
npoints = len(pointing)
##################################################################################


############# Instrument with Alternative Primary Beam ###########################
inst = QubicInstrument(detector_tau=0.0001,
                    detector_sigma=noise,
                    detector_fknee=0.,
                    detector_fslope=1)
if hornprofile != 0:
    pb = AltPrimaryBeam(14,hornprofile)
    inst.primary_beam=pb
##################################################################################


############## Input maps ########################################################
# x0 = None
# if rank == 0:
#     print('Rank '+str(rank)+' is Running Synfast')
#     x0 = np.array(hp.synfast(spectra[1:],nside,fwhm=0,pixwin=True,new=True)).T

# x0 = MPI.COMM_WORLD.bcast(x0)
# x0_noI = x0.copy()
# x0_noI[:,0] = 0
# print('Initially I map RMS is : '+str(np.std(x0[:,0])))
# print('Initially Q map RMS is : '+str(np.std(x0[:,1])))
# print('Initially U map RMS is : '+str(np.std(x0[:,2])))
# print('new I map RMS is : '+str(np.std(x0_noI[:,0])))
# print('new Q map RMS is : '+str(np.std(x0_noI[:,1])))
# print('new U map RMS is : '+str(np.std(x0_noI[:,2])))
# #### save them to disk
# qubic.io.write_map('maps_cmb.fits',x0)
# qubic.io.write_map('maps_cmb_noI.fits',x0_noI)
#### Read the maps from disk
x0 = qubic.io.read_map('maps_cmb.fits')
x0_noI = qubic.io.read_map('maps_cmb_noI.fits')
##################################################################################



############## Make maps #########################################################
todnoise = None
#### instGaussInit
maps, cov, todnoise = make_a_map(x0, pointing, inst, nside, todnoise=todnoise, fits_string=hprofiles[hornprofile], noiseless=False)
maps, cov, todnoise = make_a_map(x0, pointing, inst, nside, todnoise=todnoise, fits_string=hprofiles[hornprofile]+'_noiseless', noiseless=True)
maps, cov, todnoise = make_a_map(x0_noI, pointing, inst, nside, todnoise=todnoise, fits_string=hprofiles[hornprofile]+'_noI', noiseless=False)
maps, cov, todnoise = make_a_map(x0_noI, pointing, inst, nside, todnoise=todnoise, fits_string=hprofiles[hornprofile]+'_noiseless_noI', noiseless=True)






# ############## Show Horn Profiles ################################################
# pbclover = AltPrimaryBeam(14,1)
# pbconical = AltPrimaryBeam(14,2)
# pbgaussian = AltPrimaryBeam(14,3)
# pbhybrid_conical = AltPrimaryBeam(14,4)

# ## Initial Gaussian Profiles (analytical)
# instGaussInit = QubicInstrument(detector_tau=0.0001,
#                                 detector_sigma=noise,
#                                 detector_fknee=0.,
#                                 detector_fslope=1)
# ## Instrument with Clover Horns
# instClover = QubicInstrument(detector_tau=0.0001,
#                                 detector_sigma=noise,
#                                 detector_fknee=0.,
#                                 detector_fslope=1)
# instClover.primary_beam=pbclover
# ## Instrument with Conical Horns
# instConical = QubicInstrument(detector_tau=0.0001,
#                                 detector_sigma=noise,
#                                 detector_fknee=0.,
#                                 detector_fslope=1)
# instConical.primary_beam=pbconical
# ## Instrument with Gaussian Horns (from Daniele Buzi)
# instGaussian = QubicInstrument(detector_tau=0.0001,
#                                 detector_sigma=noise,
#                                 detector_fknee=0.,
#                                 detector_fslope=1)
# instGaussian.primary_beam=pbgaussian
# ## Instrument with Hybrid Conical Horns
# instHybrid = QubicInstrument(detector_tau=0.0001,
#                                 detector_sigma=noise,
#                                 detector_fknee=0.,
#                                 detector_fslope=1)
# instHybrid.primary_beam=pbhybrid_conical
# ##################################################################################


# ########################## Plot for Checking ######################################
# def db(data):
#     return 10*np.log10(data)

# theta = np.linspace(-90,90,5001)

# clf()
# subplot(2,1,1)
# ylim(0,1)
# xlim(-20,20)
# ylabel('Profile')
# xlabel('$\\theta$ [Deg.]')
# plot(theta,instClover.primary_beam(np.radians(theta)), label='Clover Horns',lw=3)
# plot(theta,instConical.primary_beam(np.radians(theta)), label='Conical Horns',lw=3)
# plot(theta,instGaussian.primary_beam(np.radians(theta)), label='Gaussian Horns',lw=3)
# plot(theta,instHybrid.primary_beam(np.radians(theta)), label='Hybrid Conical Horns',lw=3)
# plot(theta,instGaussInit.primary_beam(np.radians(theta)),'--', label='Current Instrument Model',lw=3)
# legend(loc='upper left',frameon=False, fontsize=10)

# subplot(2,1,2)
# ylim(-80,0)
# xlim(-90,90)
# ylabel('Profile (dB)')
# xlabel('$\\theta$ [Deg.]')
# plot(theta,db(instClover.primary_beam(np.radians(theta))), label='Clover Horns',lw=3)
# plot(theta,db(instConical.primary_beam(np.radians(theta))), label='Conical Horns',lw=3)
# plot(theta,db(instGaussian.primary_beam(np.radians(theta))), label='Gaussian Horns',lw=3)
# plot(theta,db(instHybrid.primary_beam(np.radians(theta))), label='Hybrid Conical Horns',lw=3)
# plot(theta,db(instGaussInit.primary_beam(np.radians(theta))),'--', label='Current Instrument Model',lw=3)
# ##################################################################################



