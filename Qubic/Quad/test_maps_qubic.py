from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
from Quad import qml
from Quad import pyquad
import healpy as hp
from pysimulators import FitsArray
from qubic.io import read_map
from qubic import equ2gal
import glob
from qubic.utils import progress_bar

directory = '/Volumes/Data/Qubic/SimulCurie_2015-05-30/tce120_dead_time5.0_period0.05_fknee0.1_nside256/'

allfiles = glob.glob(directory+'rec*.fits')
covmap = read_map(directory+'cov_map_coverage_angspeed1.0_delta_az30.0.fits')
bla = read_map(allfiles[0])

racenter = -0.0    #deg
deccenter = -45 #deg
center = equ2gal(racenter, deccenter) 
hp.gnomview(covmap, rot=center, reso=10, xsize=40*10)
hp.gnomview(np.log(covmap), rot=center, reso=10, xsize=40*10)


def display(input, msg='', iplot=1, range=[30, 5, 5]):
    out = []
    for i, (kind, lim) in enumerate(zip('IQU', range)):
        map = input[..., i]
        out += [hp.gnomview(map, rot=center, reso=5, xsize=800, min=-lim,
                            max=lim, title=msg + ' ' + kind,
                            sub=(1, 3, iplot + i), return_projected_map=True)]
    return out



newns = 64
newcovmap = hp.ud_grade(covmap, nside_out=newns, power=-2)
hitmin = 0.2
pixok = newcovmap > (hitmin * np.max(covmap))
npix = np.sum(pixok)
newcovmap[~pixok] = np.nan

def getmapsready(file, newns, pixok):
    maps = read_map(file)
    newmaps_I = hp.ud_grade(maps[:,0], nside_out=newns)
    newmaps_Q = hp.ud_grade(maps[:,1], nside_out=newns)
    newmaps_U = hp.ud_grade(maps[:,2], nside_out=newns)
    newmaps_I[~pixok] = np.nan
    newmaps_Q[~pixok] = np.nan
    newmaps_U[~pixok] = np.nan
    #newmaps_I[pixok] -= np.mean(newmaps_I[pixok])
    #newmaps_Q[pixok] -= np.mean(newmaps_Q[pixok])
    #newmaps_U[pixok] -= np.mean(newmaps_U[pixok])
    newmaps = np.array([newmaps_I, newmaps_Q, newmaps_U]).T
    return newmaps

maps = getmapsready(allfiles[0], newns, pixok)
display(maps, range=[1, 1, 1])
hp.gnomview(newcovmap, rot=center, reso=10, xsize=40*10)

### put all maps in memory
allmaps = np.zeros((len(allfiles), 3, npix))
for i in xrange(len(allfiles)):
    print(i, len(allfiles))
    maps = getmapsready(allfiles[i], newns, pixok)
    allmaps[i,:,:] = maps[pixok,:].T

meanmaps = np.mean(allmaps,axis=0)
iqumeanmaps = np.zeros((12*newns**2, 3))
for i in [0,1,2]: iqumeanmaps[pixok, i] = meanmaps[i,:]
display(iqumeanmaps, range=[1, 1, 1])


from qubic import (QubicAcquisition,
                   PlanckAcquisition,
                   QubicPlanckAcquisition, QubicScene)
nside = 256
scene = QubicScene(nside)
sky = scene.zeros()
acq_planck = PlanckAcquisition(150, scene, true_sky=sky)
obs_planck = acq_planck.get_observation()
Iplanck = hp.ud_grade(obs_planck[:,0], nside_out=newns)
Qplanck = hp.ud_grade(obs_planck[:,1], nside_out=newns)
Uplanck = hp.ud_grade(obs_planck[:,2], nside_out=newns)
planckmaps = np.array([Iplanck, Qplanck, Uplanck]).T

figure()
clf()
display(iqumeanmaps, range=[1, 1, 1])
figure()
clf()
display(iqumeanmaps-planckmaps, range=[1, 1, 1])


clf()
plot(allmaps[:,0,0], alpha=0.2, color='red')
plot(allmaps[:,0,0]*0+meanmaps[0,0], color='red')
plot(allmaps[:,0,100], alpha=0.2, color='blue')
plot(allmaps[:,0,100]*0+meanmaps[0,100], color='blue')
plot(allmaps[:,0,1000], alpha=0.2, color='green')
plot(allmaps[:,0,1000]*0+meanmaps[0,1000], color='green')

clf()
plot(allmaps[:,0,0]-meanmaps[0,0], alpha=0.2, color='red')
plot(allmaps[:,0,0]*0, color='red')
plot(allmaps[:,0,100]-meanmaps[0,100], alpha=0.2, color='blue')
plot(allmaps[:,0,100]*0, color='blue')
plot(allmaps[:,0,1000]-meanmaps[0,1000], alpha=0.2, color='green')
plot(allmaps[:,0,1000]*0, color='green')


allmaps_zero_t = (allmaps-meanmaps).T

npx=npix
cm = np.zeros((3*npx,3*npx))
for iqup in [0,1,2]:
    for iquq in [0,1,2]:
        print(iqup, iquq)
        bar = progress_bar(npx)
        for p in xrange(npx):
            bar.update(p)
            for q in xrange(npx):
                cm[iqup*npx+p, iquq*npx+q] = np.dot( allmaps_zero_t[p, iqup, :], allmaps_zero_t[q, iquq, :])/len(allfiles)


npx=npix
cm = np.zeros((3*npx,3*npx))
for iqup in xrange(2):
    for iquq in xrange(iqup,2):
        print(iqup, iquq)
        bar = progress_bar(npx)
        for p in xrange(npx):
            bar.update(p)
            for q in xrange(p,npx):
                cm[iqup*npx+p, iquq*npx+q] = np.dot( allmaps_zero_t[p, iqup, :], allmaps_zero_t[q, iquq, :])/len(allfiles)
                cm[iquq*npx+q, iqup*npx+p] = cm[iqup*npx+p, iquq*npx+q]

clf()
imshow(cm, interpolation='nearest')

FitsArray(cm).save('covmat0.fits')

clf()
imshow(np.log(np.abs(cm)/np.max(cm)),vmin=-6, vmax=-2, interpolation='nearest')
colorbar()



