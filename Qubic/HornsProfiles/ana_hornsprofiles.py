from __future__ import division
from pylab import *
import healpy as hp
from matplotlib.pyplot import *
import numpy as np
import os
import sys
import qubic
import pycamb
from pyoperators import DenseBlockDiagonalOperator, Rotation3dOperator
from pysimulators import FitsArray
from pyoperators import MPI
import glob
from qubic import (
    QubicAcquisition, create_sweeping_pointings, equ2gal, map2tod, tod2map_all,
    tod2map_each, create_random_pointings, QubicInstrument)



def profile(x,y,range=None,nbins=10,fmt=None,plot=True, dispersion=True):
  if range == None:
    mini = np.min(x)
    maxi = np.max(x)
  else:
    mini = range[0]
    maxi = range[1]
  dx = (maxi - mini) / nbins
  xmin = np.linspace(mini,maxi-dx,nbins)
  xmax = xmin + dx
  xc = xmin + dx / 2
  yval = np.zeros(nbins)
  dy = np.zeros(nbins)
  dx = np.zeros(nbins) + dx / 2
  for i in np.arange(nbins):
    ok = (x > xmin[i]) & (x < xmax[i])
    yval[i] = np.mean(y[ok])
    if dispersion: 
      fact = 1
    else:
      fact = np.sqrt(len(y[ok]))
    dy[i] = np.std(y[ok])/fact
  if plot: errorbar(xc, yval, xerr=dx, yerr=dy, fmt=fmt)
  return xc, yval, dx, dy




######################################################################################
from scipy.ndimage import gaussian_filter1d
th, hybconical_nofb, hybconical_14deg_05m, hybconical_14deg_1m, hybconical_14deg_2m = np.loadtxt('Beam_patterns_FB_different_heights.dat', skiprows=5).T

clf()
subplot(2,1,1)
title('Beam_patterns_FB_different_heights.dat')
ylim(-80,0)
plot(th, hybconical_nofb, label='No Forebaffle')
plot(th, hybconical_14deg_05m, label='Forebaffle 14 deg - 0.5m')
plot(th, hybconical_14deg_1m, label='Forebaffle 14 deg - 1m')
plot(th, hybconical_14deg_2m, label='Forebaffle 14 deg - 2m')
legend()
subplot(2,1,2)
ylim(-20,0)
xlim(-20,20)
plot(th, hybconical_nofb, label='No Forebaffle')
plot(th, hybconical_14deg_05m, label='Forebaffle 14 deg - 0.5m')
plot(th, hybconical_14deg_1m, label='Forebaffle 14 deg - 1m')
plot(th, hybconical_14deg_2m, label='Forebaffle 14 deg - 2m')
legend()

sm=5
clf()
subplot(2,1,1)
title('Beam_patterns_FB_different_heights.dat')
ylim(-80,0)
plot(th, hybconical_nofb, label='No Forebaffle', color='black')
plot(th, gaussian_filter1d(hybconical_14deg_05m,sm), label='Forebaffle 14 deg - 0.5m', color='blue')
plot(th, gaussian_filter1d(hybconical_14deg_1m,sm), label='Forebaffle 14 deg - 1m',color='red')
plot(th, gaussian_filter1d(hybconical_14deg_2m,sm), label='Forebaffle 14 deg - 2m',color='green')
legend()
subplot(2,1,2)
ylim(-20,0)
xlim(-20,20)
plot(th, hybconical_nofb, label='No Forebaffle', color='black')
plot(th, gaussian_filter1d(hybconical_14deg_05m,sm), label='Forebaffle 14 deg - 0.5m', color='blue')
plot(th, gaussian_filter1d(hybconical_14deg_1m,sm), label='Forebaffle 14 deg - 1m', color='red')
plot(th, gaussian_filter1d(hybconical_14deg_2m,sm), label='Forebaffle 14 deg - 2m', color='green')
legend()


th2, hybconical_nofb2, hybconical_fb14deg_1m = np.loadtxt('Beam_patterns_with_without_FB.dat', skiprows=5).T
clf()
title('Beam_patterns_with_without_FB.dat')
ylim(-60,30)
plot(th2, hybconical_nofb2)
plot(th2, gaussian_filter1d(hybconical_fb14deg_1m,10))

th2, hybconical_nofb2, hybconical_fb14deg_1m = np.loadtxt('Beam_patterns_with_without_FB.dat', skiprows=5).T
clf()
title('Beam_patterns_with_without_FB.dat')
ylim(-20,5)
plot(th2,th2*0,'k:')
plot(th2, gaussian_filter1d(hybconical_fb14deg_1m,10)-hybconical_nofb2)

######################################################################################



racenter = 0.0      # deg
deccenter = -57.0   # deg
center = equ2gal(racenter, deccenter)
nbins = 20



#hprofiles = ['GaussInit', 'Clover', 'Conical', 'Gaussian', 'HybridConical']
#hprofiles = ['GaussInit13', 'GaussInit14', '17mm.pat', '17mm_plate.pat', '17mm_plate_131GHz.pat', '17mm_plate_169GHz.pat', '22mm.pat', '22mm_plate.pat', '30mm.pat', '30mm_plate.pat', 'Full_length.pat', 'Full_length_plate.pat']
hprofiles = ['GaussInit13', '22mm_plate.pat', '30mm_plate.pat', 'Full_length_plate.pat']
rep = 'NerscBruno/'

sig_true = np.zeros((len(hprofiles),3 , nbins))
errsig_true = np.zeros((len(hprofiles),3 , nbins))
sig_true_noI = np.zeros((len(hprofiles),3 , nbins))
errsig_true_noI = np.zeros((len(hprofiles),3 , nbins))
sig_noiseless = np.zeros((len(hprofiles),3 , nbins))
errsig_noiseless = np.zeros((len(hprofiles),3 , nbins))
sig_noiseless_noI = np.zeros((len(hprofiles),3 , nbins))
errsig_noiseless_noI = np.zeros((len(hprofiles),3 , nbins))

for i in np.arange(len(hprofiles)):
  print('Doing profile: '+hprofiles[i])
  available_files = glob.glob(rep+'maps_'+hprofiles[i]+'_noiseless_noI*.fits')
  allcov = []
  allres_true = []
  allres_true_noI =[]
  allres_noiseless = []
  allres_noiseless_noI =[]
  allsig_true = np.zeros((len(available_files), 3, nbins))
  allsig_true_noI = np.zeros((len(available_files), 3, nbins))
  allsig_noiseless = np.zeros((len(available_files), 3, nbins))
  allsig_noiseless_noI = np.zeros((len(available_files), 3, nbins))
  for j in np.arange(len(available_files)): 
    #### Get the random string
    strrnd = (available_files[j].split('.')[-2]).split('_')[-1]
    print('    - Doing extension '+strrnd)
    #### read the maps
    thecov = qubic.io.read_map(rep+'cov_'+hprofiles[i]+'_'+strrnd+'.fits')
    themaps = qubic.io.read_map(rep+'maps_'+hprofiles[i]+'_'+strrnd+'.fits')
    themaps_noiseless = qubic.io.read_map(rep+'maps_'+hprofiles[i]+'_noiseless_'+strrnd+'.fits')
    themaps_noI = qubic.io.read_map(rep+'maps_'+hprofiles[i]+'_noI_'+strrnd+'.fits')
    themaps_noiseless_noI = qubic.io.read_map(rep+'maps_'+hprofiles[i]+'_noiseless_noI_'+strrnd+'.fits')
    theinit = qubic.io.read_map(rep+'initconv_'+hprofiles[i]+'_'+strrnd+'.fits')
    theinit_noI = qubic.io.read_map(rep+'initconv_'+hprofiles[i]+'_noI_'+strrnd+'.fits')
    mask = themaps[:,0] == 0
    theinit[mask] = 0
    mask_noI = themaps_noI[:,0] == 0
    theinit_noI[mask_noI] = 0
    #### Calculate residuals
    res_true = themaps - theinit
    res_true_noI = themaps_noI - theinit_noI
    res_noiseless = themaps - themaps_noiseless
    res_noiseless_noI = themaps_noI - themaps_noiseless_noI
    #### Lopp on Stokes to make the profiles
    for iqu in np.arange(3):  
      xx,yy,dx,allsig_true[j,iqu,:] = profile(thecov,res_true[:,iqu],range=[0, np.max(thecov)], nbins=nbins, plot=False, dispersion=False)
      xx,yy,dx,allsig_true_noI[j,iqu,:] = profile(thecov,res_true_noI[:,iqu],range=[0, np.max(thecov)], nbins=nbins, plot=False, dispersion=False)
      xx,yy,dx,allsig_noiseless[j,iqu,:] = profile(thecov,res_noiseless[:,iqu],range=[0, np.max(thecov)], nbins=nbins, plot=False, dispersion=False)
      xx,yy,dx,allsig_noiseless_noI[j,iqu,:] = profile(thecov,res_noiseless_noI[:,iqu],range=[0, np.max(thecov)], nbins=nbins, plot=False, dispersion=False)
    #### store the maps
    allres_true.append(res_true)
    allres_true_noI.append(res_true_noI)
    allres_noiseless.append(res_noiseless)
    allres_noiseless_noI.append(res_noiseless_noI)
    allcov.append(thecov)
  #### Make average and sig for various realizations
  sig_true[i, :, :] = np.mean(allsig_true, axis=0)
  errsig_true[i, :, :] = np.std(allsig_true, axis=0)/np.sqrt(len(available_files))
  sig_true_noI[i, :, :] = np.mean(allsig_true_noI, axis=0)
  errsig_true_noI[i, :, :] = np.std(allsig_true_noI, axis=0)/np.sqrt(len(available_files))
  sig_noiseless[i, :, :] = np.mean(allsig_noiseless, axis=0)
  errsig_noiseless[i, :, :] = np.std(allsig_noiseless, axis=0)/np.sqrt(len(available_files))
  sig_noiseless_noI[i, :, :] = np.mean(allsig_noiseless_noI, axis=0)
  errsig_noiseless_noI[i, :, :] = np.std(allsig_noiseless_noI, axis=0)/np.sqrt(len(available_files))


class AltPrimaryBeam(object):
    def __init__(self, filename):
        if filename == 'GaussInit13':
            self.th = np.arange(91)
            self.beam = np.exp(-0.5*self.th**2/(2*(13./2.35)**2))**2
            self.beam_db = 10*np.log10(self.beam)
            abovehalf = self.beam >= 0.5
            self.fwhm_deg = np.max(self.th[abovehalf]) - np.min(self.th[abovehalf])
            #print(self.fwhm_deg)
            self.sigma = np.radians(self.fwhm_deg) / np.sqrt(8 * np.log(2))
            self.fwhm_sr = 2 * pi * self.sigma**2
        elif filename == 'GaussInit14':
            self.th = np.arange(91)
            self.beam = np.exp(-0.5*self.th**2/(2*(14./2.35)**2))**2
            self.beam_db = 10*np.log10(self.beam)
            abovehalf = self.beam >= 0.5
            self.fwhm_deg = np.max(self.th[abovehalf]) - np.min(self.th[abovehalf])
            #print(self.fwhm_deg)
            self.sigma = np.radians(self.fwhm_deg) / np.sqrt(8 * np.log(2))
            self.fwhm_sr = 2 * pi * self.sigma**2
        else:
            data = np.loadtxt('beam/'+filename, skiprows=4)
            self.th = data[:,0]
            self.beam_db = data[:,3]
            self.beam = 10**(data[:,3]/10)
            abovehalf = self.beam >= 0.5
            self.fwhm_deg = np.max(self.th[abovehalf]) - np.min(self.th[abovehalf])
            self.sigma = np.radians(self.fwhm_deg) / np.sqrt(8 * np.log(2))
            self.fwhm_sr = 2 * pi * self.sigma**2
    def __call__(self, theta):
        return 10**(np.interp(theta, np.radians(self.th), self.beam_db/10))


theta = np.linspace(0,90,1000)

iquname=['I','Q','U']
col = get_cmap('jet')(np.linspace(0, 1.0, len(hprofiles)))

clf()

for i in np.arange(len(hprofiles)):
  subplot(1,3,1)
  xlim(0,40)
  ylim(-30,0)
  pb = AltPrimaryBeam(hprofiles[i])
  plot(theta, 10*np.log10(pb(np.radians(theta))), color=col[i], lw=3, label=hprofiles[i])
legend(fontsize=8, loc='upper right')
xlabel(r'$\theta$')
ylabel('dB')

for i in np.arange(len(hprofiles)-1)+1:
  subplot(1,3,1)
  ylim(-50,0)
  pb = AltPrimaryBeam(hprofiles[i])
  plot(theta, 10*np.log10(pb(np.radians(theta))), color=col[i], lw=3)
  for iqu in np.arange(2)+1:
    subplot(1,3,iqu+1)
    ylim(0,2)
    plot(xx/np.max(xx),xx*0+1,'k:')
    ratio = sig_true_noI[i,iqu,:]/sig_true_noI[0,iqu,:]
    dratio = ratio * (errsig_true_noI[i,iqu,:] / sig_true_noI[i,iqu,:] + errsig_true_noI[0,iqu,:]/sig_true_noI[0,iqu,:])
    plot(xx/np.max(xx),ratio,label=hprofiles[i],lw=3, color=col[i])
    #plot(xx/np.max(xx),ractio+dratio, color=col[i])
    #plot(xx/np.max(xx),ratio-dratio, color=col[i])
    #fill_between(xx/np.max(xx), ratio+dratio, y2 =ratio-dratio,alpha=0.05)
    legend(fontsize=8, loc='lower right')
    xlabel('Normalized coverage')
    ylabel('Maps RMS ratio w.r.t. '+hprofiles[0])
    title(iquname[iqu])

savefig('beams_bruno.png')








clf()
col = get_cmap('jet')(np.linspace(0, 1.0, 4))
bla=0

for i in np.arange(len(hprofiles)):
  subplot(1,3,1)
  xlim(0,40)
  ylim(-30,0)
  pb = AltPrimaryBeam(hprofiles[i])
  plot(theta, 10*np.log10(pb(np.radians(theta))), color=col[bla], lw=3, label=hprofiles[i])
  bla+=1
legend(fontsize=8, loc='upper right')
xlabel(r'$\theta$')
ylabel('dB')

bla=1
for i in np.arange(len(hprofiles)-1)+1:
  subplot(1,3,1)
  ylim(-50,0)
  pb = AltPrimaryBeam(hprofiles[i])
  plot(theta, 10*np.log10(pb(np.radians(theta))), color=col[bla], lw=3)
  for iqu in np.arange(2)+1:
    subplot(1,3,iqu+1)
    ylim(0,2)
    plot(xx/np.max(xx),xx*0+1,'k:')
    ratio = sig_true_noI[i,iqu,:]/sig_true_noI[0,iqu,:]
    dratio = ratio * (errsig_true_noI[i,iqu,:] / sig_true_noI[i,iqu,:] + errsig_true_noI[0,iqu,:]/sig_true_noI[0,iqu,:])
    #plot(xx/np.max(xx),ratio,label=hprofiles[i],lw=3, color=col[bla])
    #plot(xx/np.max(xx),ratio+dratio, color=col[i])
    #plot(xx/np.max(xx),ratio-dratio, color=col[i])
    fill_between(xx/np.max(xx), ratio+dratio, y2 =ratio-dratio,alpha=0.5, color=col[bla])
    legend(fontsize=8, loc='lower right')
    xlabel('Normalized coverage')
    ylabel('Maps RMS ratio w.r.t. '+hprofiles[0])
    title(iquname[iqu])
  bla+=1

savefig('beams_bruno_30_42.png')











hprofiles = ['GaussInit', 'Clover', 'Conical', 'Gaussian', 'HybridConical']
rep = 'NERSC2/'
maps = []
maps_noiseless = []
maps_noI = []
maps_noiseless_noI = []
cov = []
cov_noiseless = []
cov_noI = []
cov_noiseless_noI = []
res = []
res_noI = []
nbins=10
sigvals = np.zeros((nbins, len(hprofiles), 3))
sigvals_noI = np.zeros((nbins, len(hprofiles), 3))
for i in np.arange(len(hprofiles)):
  maps.append(qubic.io.read_map(rep+'maps_'+hprofiles[i]+'.fits'))
  maps_noiseless.append(qubic.io.read_map(rep+'maps_'+hprofiles[i]+'_noiseless.fits'))
  maps_noI.append(qubic.io.read_map(rep+'maps_'+hprofiles[i]+'_noI.fits'))
  maps_noiseless_noI.append(qubic.io.read_map(rep+'maps_'+hprofiles[i]+'_noiseless_noI.fits'))
  cov.append(qubic.io.read_map(rep+'cov_'+hprofiles[i]+'.fits'))
  cov_noiseless.append(qubic.io.read_map(rep+'cov_'+hprofiles[i]+'_noiseless.fits'))
  cov_noI.append(qubic.io.read_map(rep+'cov_'+hprofiles[i]+'_noI.fits'))
  cov_noiseless_noI.append(qubic.io.read_map(rep+'cov_'+hprofiles[i]+'_noiseless_noI.fits'))
  res.append(maps[i] - maps_noiseless[i])
  res_noI.append(maps_noI[i] - maps_noiseless_noI[i])
  for iqu in np.arange(3):
      xx,yy,dx,sigvals[:,i,iqu] = profile(cov[i],res[i][:,iqu],range=[0, np.max(cov[0])], nbins=nbins, plot=False, dispersion=False)
      xx,yy,dx,sigvals_noI[:,i,iqu] = profile(cov[i],res_noI[i][:,iqu],range=[0, np.max(cov[0])], nbins=nbins, plot=False, dispersion=False)

iquname=['I','Q','U']
col=['magenta','blue','green','red','cyan']
clf()
for i in np.arange(len(hprofiles)-1)+1:
  for iqu in np.arange(2)+1:
    subplot(1,2,iqu)
    ylim(0,1.5)
    plot(xx/np.max(xx),xx*0+1,'k:')
    plot(xx/np.max(xx),sigvals_noI[:,i,iqu]/sigvals_noI[:,0,iqu], color=col[i],label=hprofiles[i],lw=3)
    legend(fontsize=8, loc='lower right')
    xlabel('Normalized coverage')
    ylabel('Maps RMS ratio')
    title(iquname[iqu])



col=['magenta','blue','green','red','cyan']
clf()
for i in np.arange(len(hprofiles)-1)+1:
  for iqu in np.arange(2)+1:
    subplot(1,2,iqu)
    ylim(0,1.5)
    plot(xx/np.max(xx),xx*0+1,'k:')
    plot(xx/np.max(xx),sigvals[:,i,iqu]/sigvals[:,0,iqu], color=col[i],label=hprofiles[i],lw=3)
    legend(fontsize=8, loc='lower right')
    title(iquname[iqu])


hp.gnomview(res[0][:,0],rot=center,reso=30)

clf()
mask = cov[0] == 0
hist(res[0][~mask,0],bins=100, range=[-1.5,1.5],alpha=0.5)
hist(res[0][~mask,1],bins=100, range=[-1.5,1.5],alpha=0.5)
hist(res[0][~mask,2],bins=100, range=[-1.5,1.5],alpha=0.5)


clf()
mask = cov[0] == 0
hist(res[0][~mask,1],bins=100, range=[-1.5,1.5],alpha=0.5)
hist(res_noI[0][~mask,1],bins=100, range=[-1.5,1.5],alpha=0.5)


clf()
mask = cov[0] == 0
hist(res[0][~mask,0]-res_noI[0][~mask,0],bins=100,alpha=0.5)
clf()
mask = cov[0] == 0
hist(res[0][~mask,1]-res_noI[0][~mask,1],bins=100,alpha=0.5)
clf()
mask = cov[0] == 0
hist(res[0][~mask,2]-res_noI[0][~mask,2],bins=100,alpha=0.5)








