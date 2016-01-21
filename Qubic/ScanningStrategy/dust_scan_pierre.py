from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import sin, cos, pi
from pyoperators import Rotation3dOperator
from pysimulators import FitsArray
from qubic import (
    QubicInstrument, create_random_pointings, equ2gal, gal2equ, map2tod,
    tod2map_all, tod2map_each)
from qubic.utils import progress_bar


def euler_rotation(angXdeg, angYdeg, angZdeg):
    thetaX = np.radians(angXdeg)
    thetaY = np.radians(angYdeg)
    thetaZ = np.radians(angZdeg)
    Rx = np.array([[1, 0, 0], [0, cos(thetaX), sin(thetaX)],
                   [0, -sin(thetaX), cos(thetaX)]])
    Ry = np.array([[cos(thetaY), 0, sin(thetaY)], [0, 1, 0],
                   [-sin(thetaY), 0, cos(thetaY)]])
    Rz = np.array([[cos(thetaZ), -sin(thetaZ), 0],
                   [sin(thetaZ), cos(thetaZ), 0], [0, 0, 1]])
    mat = np.dot(Rz, np.dot(Ry, Rx))
    return mat


def circ_around_radec(radeccenter, radius, npoints = 1000):
    #### circle around north pole first
    lat = pi/2-np.radians(np.zeros(npoints) + radius)
    lon = np.linspace(-np.pi, np.pi, npoints)
    #### Corresponding Unit vectors
    uvinitx = np.sin(pi/2-lat) * np.cos(lon)
    uvinity = np.sin(pi/2-lat) * np.sin(lon)
    uvinitz = np.cos(pi/2-lat)
    uvinit = np.array([uvinitx, uvinity, uvinitz]).T
    #### Get Rotation Matrix
#    mat = euler_rotation(0., 90.-radeccenter[1], radeccenter[0])
    #### apply it to unit vector
#    new = np.dot(mat, uvinit)
    mat = Rotation3dOperator('YZ', 90-radeccenter[1], radeccenter[0],
                             degrees=True)
    new = mat(uvinit)
    #### Get the result
    newdec = pi/2 - np.arccos(new[:, 2])
    newra = np.arctan2(new[:, 1], new[:, 0])
    return [np.degrees(newra), np.degrees(newdec)]




######## Input Planck map at 143 GHz, Q and U maps are set to zero
nside = 128
init143 = hp.read_map('HFI_SkyMap_143_2048_R1.10_nominal.fits')
map143 = hp.ud_grade(init143, nside, order_out='RING')
#hp.mollview(map143)
maps = np.transpose(np.array([map143, map143*0, map143*0]))

mapcmb  = 1e-6*hp.ud_grade(hp.read_map('COM_CompMap_CMB-smica_2048_R1.20.fits'), nside, order_out='RING')
mapscmb = np.transpose(np.array([mapcmb, mapcmb*0, mapcmb*0]))
hp.mollview(mapcmb)

mapdust = map143 - mapcmb
mapsdust = np.transpose(np.array([mapdust, mapdust*0, mapdust*0]))



################# Qubic Instrument #########################################
qubicinst = QubicInstrument('monochromatic',nside=nside)
detectors=qubicinst.detector.packed
mp.clf()
mp.plot(detectors.center[:,0], detectors.center[:,1],'ro')

################# random pointings ##########################################
racenter = 0.0
deccenter = -57.0
center = equ2gal(racenter, deccenter)

################## new way of making the MC
nsmap = 16
thgal, phgal = hp.pix2ang(nsmap, np.arange(12 * nsmap**2))
llgal = np.degrees(phgal) - 180
bbgal = 90 - np.degrees(thgal)
ra, dec = gal2equ(llgal, bbgal)

mp.clf()
mp.subplot(211, projection='mollweide')
mp.plot(np.radians(llgal),np.radians(bbgal), '.')
mp.title('Galactic')
mp.subplot(212, projection='mollweide')
mp.plot(np.radians(ra-180), np.radians(dec), '.')
mp.title('Equatorial')

#### create pointings at the center of each healpix pixel
npointings = 12 * nsmap**2
initpointing = create_random_pointings([ra[0], dec[0]], npointings, 0.001)
bar = progress_bar(npointings)
for i in np.arange(npointings):
    bla = create_random_pointings([ra[i], dec[i]], 2, 0.01)
    initpointing[i] = bla[0]
    bar.update()

fracvals = [0.3, 0.5, 0.9, 0.999]
mapmean_cmb = np.zeros((len(fracvals), npointings))
mapmean_dust = np.zeros((len(fracvals), npointings))
mapmean_all = np.zeros((len(fracvals), npointings))
maprms_cmb = np.zeros((len(fracvals), npointings))
maprms_dust = np.zeros((len(fracvals), npointings))
maprms_all = np.zeros((len(fracvals), npointings))
for j in np.arange(len(fracvals)):
    frac = fracvals[j]
    print(j, len(fracvals))
    theinst = QubicInstrument('monochromatic', nside=nside,
                              synthbeam_fraction=frac)
    tod_cmb, toto = map2tod(theinst, initpointing, mapscmb, convolution=True)
    tod_dust, toto = map2tod(theinst, initpointing, mapsdust, convolution=True)
    tod_all, toto = map2tod(theinst, initpointing, maps, convolution=True)
    mapmean_cmb[j] = np.mean(tod_cmb, axis=0)
    maprms_cmb[j] = np.std(tod_cmb, axis=0)
    mapmean_dust[j] = np.mean(tod_dust, axis=0)
    maprms_dust[j] = np.std(tod_dust, axis=0)
    mapmean_all[j] = np.mean(tod_all, axis=0)
    maprms_all[j] = np.std(tod_all, axis=0)


FitsArray(mapmean_cmb, copy=False).save('small2_mapmean_cmb.fits')
FitsArray(mapmean_dust, copy=False).save('small2_mapmean_dust.fits')
FitsArray(mapmean_all, copy=False).save('small2_mapmean_all.fits')
FitsArray(maprms_cmb, copy=False).save('small2_maprms_cmb.fits')
FitsArray(maprms_dust, copy=False).save('small2_maprms_dust.fits')
FitsArray(maprms_all, copy=False).save('small2_maprms_all.fits')





mapmean_cmb = FitsArray('small2_mapmean_cmb.fits')
mapmean_dust = FitsArray('small2_mapmean_dust.fits')
mapmean_all = FitsArray('small2_mapmean_all.fits')
maprms_cmb = FitsArray('small2_maprms_cmb.fits')
maprms_dust = FitsArray('small2_maprms_dust.fits')
maprms_all = FitsArray('small2_maprms_all.fits')

racenter = 0
deccenter = -57.0
deltatheta = 15.
deltatheta2 = 20.
deltatheta3 = 25.
racirc, deccirc = circ_around_radec([racenter, deccenter], deltatheta)
racirc2, deccirc2 = circ_around_radec([racenter, deccenter], deltatheta2)
racirc3, deccirc3 = circ_around_radec([racenter, deccenter], deltatheta3)

# Initial dust map 
clf()
l,b = equ2gal(racenter, deccenter)
vv = hp.ang2vec(np.radians(90-b), np.radians(l))
pixels = hp.query_disc(hp.npix2nside(len(mapdust)), vv, np.radians(15))
mm = np.min(mapdust[pixels])
#hp.gnomview(mapdust-mm, coord='GE', title='Planck Dust Map', rot=[racenter,deccenter], fig=1, reso=25,min=0,max=1e-4)
hp.mollview(mapdust-mm, coord='GE', title='Planck Dust Map', rot=[racenter,deccenter], fig=2, min=0,max=1e-4)
hp.projplot(pi/2-np.radians(deccenter), np.radians(racenter), 'kx')
hp.projplot(pi/2-np.radians(deccirc), np.radians(racirc), 'k')
hp.projplot(pi/2-np.radians(deccirc2), np.radians(racirc2), 'k')
hp.projplot(pi/2-np.radians(deccirc3), np.radians(racirc3), 'k')
hp.projplot(pi/2-np.radians(-45), np.radians(4*15+40/60+12/3600), 'ko')
hp.projplot(pi/2-np.radians(-30/60), np.radians(11*15+453/60+0/3600), 'ko')
hp.projplot(pi/2 - np.radians(-32-48/60), np.radians(23*15+1/60+48/3600), 'ko')
bicepalpha = np.ravel([np.zeros(10)-60, np.linspace(-60, 60, 10), np.zeros(10)+60, np.linspace(60, -60, 10)])
bicepdelta = np.ravel([np.linspace(-70, -45, 10),np.zeros(10)-45, np.linspace(-45, -70, 10),np.zeros(10)-70])
hp.projplot(pi/2-np.radians(bicepdelta), np.radians(bicepalpha), 'k--')

# Initial 143 GHz map 
clf()
hp.gnomview(init143, coord='GE', title='Planck 143 GHz Map', rot=[racenter,deccenter], fig=2, reso=25,min=-5e-4,max=5e-4)
hp.projplot(pi/2-np.radians(deccenter), np.radians(racenter), 'kx')
hp.projplot(pi/2-np.radians(deccirc), np.radians(racirc), 'k')
hp.projplot(pi/2-np.radians(deccirc2), np.radians(racirc2), 'k')
hp.projplot(pi/2-np.radians(deccirc3), np.radians(racirc3), 'k')
hp.projplot(pi/2-np.radians(-45), np.radians(4*15+40/60+12/3600), 'ko')
hp.projplot(pi/2-np.radians(-30/60), np.radians(11*15+453/60+0/3600), 'ko')
hp.projplot(pi/2 - np.radians(-32-48/60), np.radians(23*15+1/60+48/3600), 'ko')
bicepalpha = np.ravel([np.zeros(10)-60, np.linspace(-60, 60, 10), np.zeros(10)+60, np.linspace(60, -60, 10)])
bicepdelta = np.ravel([np.linspace(-70, -45, 10),np.zeros(10)-45, np.linspace(-45, -70, 10),np.zeros(10)-70])
hp.projplot(pi/2-np.radians(bicepdelta), np.radians(bicepalpha), 'k--')


##################### DUST/CMB mollview
mp.clf()
mn = -1
mx = 1
for numk in np.arange(len(fracvals)):
    l,b = equ2gal(racenter, deccenter)
    vv = hp.ang2vec(np.radians(90-b), np.radians(l))
    pixels = hp.query_disc(nsmap, vv, np.radians(15))
    mm = np.mean(mapmean_dust[numk,pixels])
    hp.mollview((mapmean_dust[numk]-mm) / np.mean(maprms_cmb[numk]), coord='GE',
                title='Mean dust / RMS CMB - frac='+str(fracvals[numk]),
                fig=1,min=mn, max=mx, sub=(2, 2, numk+1))
    hp.projplot(pi/2-np.radians(deccenter), np.radians(racenter), 'kx')
    hp.projplot(pi/2-np.radians(deccirc), np.radians(racirc), 'k')
    hp.projplot(pi/2-np.radians(deccirc2), np.radians(racirc2), 'k')
    hp.projplot(pi/2-np.radians(deccirc3), np.radians(racirc3), 'k')
    hp.projplot(pi/2-np.radians(-45), np.radians(4*15+40/60+12/3600), 'ko')
    hp.projplot(pi/2-np.radians(-30/60), np.radians(11*15+453/60+0/3600), 'ko')
    hp.projplot(pi/2 - np.radians(-32-48/60), np.radians(23*15+1/60+48/3600), 'ko')
    bicepalpha = np.ravel([np.zeros(10)-60, np.linspace(-60, 60, 10), np.zeros(10)+60, np.linspace(60, -60, 10)])
    bicepdelta = np.ravel([np.linspace(-70, -45, 10), np.zeros(10)-45, np.linspace(-45, -70, 10), np.zeros(10)-70])
    hp.projplot(pi/2-np.radians(bicepdelta), np.radians(bicepalpha), 'k--')

mp.savefig('mean_dust_ratio.pdf')

clf()
mx=3
numk=0
mn = -1e-5
mx = 1e-5
for numk in np.arange(len(fracvals)):
        l,b = equ2gal(racenter, deccenter)
        vv = hp.ang2vec(np.radians(90-b), np.radians(l))
        pixels = hp.query_disc(nsmap, vv, np.radians(15))
        mm = np.mean(mapmean_dust[numk,pixels])
        a=hp.gnomview(mapmean_dust[numk,:]-mm,coord='GE',rot=[racenter,deccenter],
                title='Mean dust - frac='+str(fracvals[numk]),fig=1, min=mn, max=mx,
                sub=(2,2,numk+1),reso=20,return_projected_map=True)
        hp.projplot(pi/2-np.radians(deccenter),np.radians(racenter),'kx')
        hp.projplot(pi/2-np.radians(deccirc),np.radians(racirc),'k')
        hp.projplot(pi/2-np.radians(deccirc2),np.radians(racirc2),'k')
        hp.projplot(pi/2-np.radians(deccirc3),np.radians(racirc3),'k')
        hp.projplot(pi/2-np.radians(-45),np.radians(4*15+40/60+12/3600),'ko')
        hp.projplot(pi/2-np.radians(-30/60),np.radians(11*15+453/60+0/3600),'ko')
        hp.projplot(pi/2-np.radians(-32-48/60),np.radians(23*15+1/60+48/3600),'ko')
        bicepalpha=np.ravel([np.zeros(10)-60,np.linspace(-60,60,10),np.zeros(10)+60,np.linspace(60,-60,10)])
        bicepdelta=np.ravel([np.linspace(-70,-45,10),np.zeros(10)-45,np.linspace(-45,-70,10),np.zeros(10)-70])
        hp.projplot(pi/2-np.radians(bicepdelta),np.radians(bicepalpha),'k--')


clf()
mx=3
numk=0
mn = -1
mx = 1
for numk in np.arange(len(fracvals)):
        l,b = equ2gal(racenter, deccenter)
        vv = hp.ang2vec(np.radians(90-b), np.radians(l))
        pixels = hp.query_disc(nsmap, vv, np.radians(15))
        mm = np.mean(mapmean_dust[numk,pixels])
        a=hp.gnomview((mapmean_dust[numk,:] - mm)/np.mean(maprms_cmb[numk,:]),coord='GE',rot=[racenter,deccenter],
                title='Mean dust / RMS CMB - frac='+str(fracvals[numk]),fig=1, min =mn, max=mx,
                sub=(2,2,numk+1),reso=20,return_projected_map=True)
        hp.projplot(pi/2-np.radians(deccenter),np.radians(racenter),'kx')
        hp.projplot(pi/2-np.radians(deccirc),np.radians(racirc),'k')
        hp.projplot(pi/2-np.radians(deccirc2),np.radians(racirc2),'k')
        hp.projplot(pi/2-np.radians(deccirc3),np.radians(racirc3),'k')
        hp.projplot(pi/2-np.radians(-45),np.radians(4*15+40/60+12/3600),'ko')
        hp.projplot(pi/2-np.radians(-30/60),np.radians(11*15+453/60+0/3600),'ko')
        hp.projplot(pi/2-np.radians(-32-48/60),np.radians(23*15+1/60+48/3600),'ko')
        bicepalpha=np.ravel([np.zeros(10)-60,np.linspace(-60,60,10),np.zeros(10)+60,np.linspace(60,-60,10)])
        bicepdelta=np.ravel([np.linspace(-70,-45,10),np.zeros(10)-45,np.linspace(-45,-70,10),np.zeros(10)-70])
        hp.projplot(pi/2-np.radians(bicepdelta),np.radians(bicepalpha),'k--')
