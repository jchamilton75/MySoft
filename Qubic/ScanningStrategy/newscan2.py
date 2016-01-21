from __future__ import division
from numpy import *
from matplotlib.pyplot import *

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import pysimulators
import pyoperators
from qubic import QubicAcquisition, QubicInstrument, create_sweeping_pointings
from astropy.time import Time, TimeDelta

### Dome C
domec = np.array([-(75 + 6 / 60), 123 + 20 / 60])
#SITELAT = -(75 + 6 / 60)
#SITELON = 123 + 20 / 60
### San Antonio de los Cobres
sadlc = np.array([-24.18947, -66.472016])
#SITELAT = -24.18947
#SITELON = -66.472016

q2g = pysimulators.SphericalEquatorial2GalacticOperator(degrees=True)
g2q = pysimulators.SphericalGalactic2EquatorialOperator(degrees=True)


################### Get a center around racenter, deccenter
nside = 256
racenter = 0
deccenter = -57.0
deltatheta = 15.

spot_bicep2 = [racenter, deccenter]
newspot = g2q([315, -75])
galbicep2 = q2g(spot_bicep2)
galnew = q2g(newspot)



def euler_rotation(angXdeg, angYdeg, angZdeg):
    thetaX = np.radians(angXdeg)
    thetaY = np.radians(angYdeg)
    thetaZ = np.radians(angZdeg)
    Rx = np.array([ [1, 0, 0], [0, cos(thetaX), sin(thetaX)], [0, -sin(thetaX), cos(thetaX)] ])
    Ry = np.array([ [cos(thetaY), 0, sin(thetaY)], [0, 1, 0], [-sin(thetaY), 0, cos(thetaY)] ])
    Rz = np.array([ [cos(thetaZ), -sin(thetaZ), 0], [sin(thetaZ), cos(thetaZ), 0], [0, 0, 1] ])
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
    uvinit = np.array([uvinitx,uvinity,uvinitz])
    #### Get Rotation Matrix
    mat = euler_rotation(0., 90.-radeccenter[1], radeccenter[0])
    #### apply it to unit vector
    new = np.dot(mat, uvinit)
    #### Get the result
    newdec = pi/2-np.arccos(new[2,:])
    newra = np.arctan2(new[1,:],new[0,:])
    return [np.degrees(newra),np.degrees(newdec)]
    
ra,dec = circ_around_radec([racenter,deccenter],15.)
clf()
subplot(111,projection='mollweide')
plot(np.radians(racenter),np.radians(deccenter),'ro')
plot(np.radians(ra),np.radians(dec),'b.')

#################### Constraints for scanning strategy
fsampling = 100 #Hz
######## Atmosphere knee frequency 
#### from Bicep Biermann 20122 fig 12
#### knee frequency atm without pair differencing ~ 1Hz
#### with pair differencing goes to 0.1 Hz
#### ===> need to come back at the same pixel as fast as possible
######## HWP duty cycle
#### Change HWP orientation as often as possible: after each scan
#### 5 sec dead time each time => want the duty cycle to be ~ 90%
#### a scan needs to last more than 1 min


def observe_onespot(spot, site, nt=1000, DEFAULT_DATE_OBS='2016-01-01 00:00:00'):
    racenter = spot[0]
    deccenter = spot[1]
    SITELAT = site[0]
    SITELON = site[1]
    time = TimeDelta(linspace(0,24*60*60,nt), format='sec')
    op3 = pyoperators.Cartesian2SphericalOperator('azimuth,elevation', degrees=True) 
    op2 = pysimulators.CartesianEquatorial2HorizontalOperator('NE', time, SITELAT, SITELON) 
    op1 = pyoperators.Rotation3dOperator("ZY'", racenter, 90 - deccenter, degrees=True)
    op0 = pyoperators.Spherical2CartesianOperator('zenith,azimuth', degrees=True)
    theta = np.zeros(nt)
    phi = np.zeros(nt)
    inputdata = np.asarray(np.asarray([theta.T, phi.T]).T)
    coords = op3(op2(op1(op0(inputdata))))
    azimuth = coords[:, 0]
    elevation = coords[:, 1]
    return time, azimuth, elevation

def plot_onespot(spot, site, nt=1000, DEFAULT_DATE_OBS='2016-01-01 00:00:00'):
    racenter = spot[0]
    deccenter = spot[1]
    time, azimuth, elevation = observe_onespot(spot, site, nt=nt, DEFAULT_DATE_OBS=DEFAULT_DATE_OBS)
    subplot(2,2,1)
    hour = time.sec/3600
    plot(hour,elevation,'.')
    bad = elevation < 30
    print(1.-np.sum(bad)*1./len(bad))
    plot(hour[bad], elevation[bad], 'k.')
    plot(hour,hour*0+30, 'r:')
    xlabel('Time [Hours]')
    ylabel('Elevation [deg]')
    xlim(0,24)
    ylim(0,90)
    subplot(2,2,2)
    #ylim(-180,180)
    #plot(hour, ((azimuth + 180 + 360) % 360)-180,'.')
    ylim(0,360)
    plot(hour, azimuth,'.')
    plot(hour[bad], azimuth[bad],'k.')
    xlabel('Time [Hours]')
    ylabel('Azimuth [deg]')
    xlim(0,24)
    subplot(2,2,3, projection='mollweide')
    plot(np.radians(((racenter + 180 + 360) % 360)-180), np.radians(deccenter),'o')
    title('Equatorial')
    op = pysimulators.SphericalEquatorial2GalacticOperator(degrees=True) 
    lbcenter = op(spot)
    subplot(2,2,4, projection='mollweide')
    title('Galactic')
    plot(np.radians(((lbcenter[0] + 180 + 360) % 360)-180), np.radians(lbcenter[1]),'o')


###### BICEP2 Spot
clf()
plot_onespot(spot_bicep2, domec)
savefig('Concordia.png')


nbra = 2
all_ra = np.arange(nbra)*360./nbra

clf()
for thera in all_ra: plot_onespot(np.array([thera, deccenter]), sadlc)
savefig('Chorillos.png')

####### New Spot
clf()
plot_onespot(newspot, domec)
savefig('Concordia_newspot.png')


nbra = 2
all_ra = (np.arange(nbra)*360./nbra + newspot[0] ) % 360

clf()
for thera in all_ra: plot_onespot(np.array([thera, newspot[1]]), sadlc)
savefig('Chorillos_newspot.png')





####now with TOD
fech=1. # Hz
deltaaz=30
angspeed= 1. #deg/s
nsw = 100
#ptg = create_sweeping_pointings(spot_bicep2, 24., 1./fech, angspeed, deltaaz, nsw, 0, 0, 
#    date_obs = '2016-01-01 00:00:00', latitude=domec[0], longitude=domec[1])
ptg = create_sweeping_pointings(spot_bicep2, 24., 1./fech, angspeed, deltaaz, nsw, 0, 0, 
    date_obs = '2016-01-01 00:00:00', latitude=sadlc[0], longitude=sadlc[1])
ok = ptg.elevation > 30
fractime = ok.sum()/len(ok)*100
ptg = ptg[ptg.elevation > 30]

clf()
subplot(3,1,1)
plot(ptg.azimuth, ptg.elevation)
subplot(3,1,2)
plot(ptg.time, ptg.azimuth)
subplot(3,1,3)
plot(ptg.time, ptg.elevation)

clf()
subplot(2,2,1)
title('Equatorial')
plot(((ptg.equatorial[:,0]+180+360) % 360)-180, ptg.equatorial[:,1],',')
plot(spot_bicep2[0], spot_bicep2[1], 'ro')
plot(newspot[0], newspot[1], 'go')
subplot(2,2,2)
title('Galactic')
plot(((ptg.galactic[:,0]+180+360) % 360)-180, ptg.galactic[:,1],',')
plot(galbicep2[0]-360, galbicep2[1], 'ro')
plot(galnew[0]-360, galnew[1], 'go')
subplot(2,2,3, projection='mollweide')
title('Equatorial')
plot(np.radians(((ptg.equatorial[:,0]+180+360) % 360)-180), np.radians(ptg.equatorial[:,1]),',')
plot(np.radians(spot_bicep2[0]), np.radians(spot_bicep2[1]), 'ro')
plot(np.radians(newspot[0]), np.radians(newspot[1]), 'go')
subplot(2,2,4, projection='mollweide')
title('Galactic')
plot(np.radians(((ptg.galactic[:,0]+180+360) % 360)-180), np.radians(ptg.galactic[:,1]),',')
plot(np.radians(galbicep2[0]-360), np.radians(galbicep2[1]), 'ro')
plot(np.radians(galnew[0]-360), np.radians(galnew[1]), 'go')


import healpy
ns = 256
ip = ptg.healpix(ns)
map = np.zeros(12*ns**2)
for i in ip: map[i]+=1
map = hp.smoothing(map, fwhm=np.radians(14))
map = map/np.max(map)
fsky = map.sum()/len(map)

hp.mollview(map, title='Fractime: {0:4.1f}% - fsky: {1:4.1f}%'.format(fractime, fsky*100))



nside = 8
map = np.arange(12*8**2)
idbicep2 = hp.ang2pix(nside, np.radians(90-galbicep2[1]), np.radians(galbicep2[0]))
map[idbicep2]+=100
idnew = hp.ang2pix(nside, np.radians(90-galnew[1]), np.radians(galnew[0]))
map[idnew]+=100
hp.orthview(map, rot=[0,90])




############# Tower
time, azimuth, elevation = observe_onespot(spot_bicep2, domec,  nt=1000, DEFAULT_DATE_OBS='2016-01-01 00:00:00')
time2, azimuth2, elevation2 = observe_onespot(newspot, domec,  nt=1000, DEFAULT_DATE_OBS='2016-01-01 00:00:00')

htower = 45.
dtower = np.array([45., 70.])
thetatower = np.degrees(arctan2(htower, dtower))

clf()
plot(azimuth, elevation)

clf()
subplot(111, projection='polar')
ylim(0,90)
fill_between(np.radians(azimuth), 90-elevation-6.5, y2=90-elevation+6.5, color='red', alpha=0.3)
plot(np.radians(azimuth), 90-elevation-6.5, 'r', label='BICEP2 Spot')
plot(np.radians(azimuth), 90-elevation+6.5, 'r')
fill_between(np.radians(azimuth2), 90-elevation2-6.5, y2=90-elevation2+6.5, color='blue', alpha=0.3)
plot(np.radians(azimuth2), 90-elevation2-6.5, 'b', label='New Spot')
plot(np.radians(azimuth2), 90-elevation2+6.5, 'b')
plot(np.radians(azimuth), azimuth*0+90-thetatower[0], 'm', lw=3, label='Tower at 45m')
plot(np.radians(azimuth), azimuth*0+90-thetatower[1], 'g', lw=3, label='Tower at 70m')
plot(np.radians(azimuth), azimuth*0+90-30, 'k--', lw=3, label='Observation Limit')
legend()
#savefig('tower_fields_azel.png')



def plot_region(az, el, deltael, color=None, label=None):
    fill_between(np.radians(az), 90-el-deltael*0.5, y2=90-el+deltael*0.5, color=color, alpha=0.3)
    plot(np.radians(az), 90-el+deltael*0.5, color=color, label=label)
    plot(np.radians(az), 90-el-deltael*0.5, color=color)
    
htower = 45.
hqubic = 5.
dtower = 57.
aztower = 100.
thetatower = np.degrees(arctan2(htower-hqubic, dtower))




av_el = 50.
delta_pt = 20.
fwhm =  13.
clf()
subplot(111, projection='polar')
ylim(0,90)
plot_region(azimuth, azimuth*0+av_el, 2*delta_pt, color='green')
plot_region(azimuth, azimuth*0+av_el, 2*delta_pt+fwhm/2, color='green', 
    label='QUBIC {0:4.0f}$^\circ$ +/- {1:4.1f}$^\circ$ (PT) +/- {2:4.1f}$^\circ$ (FWHM)'.format(av_el, delta_pt, fwhm/2))
plot(np.radians(azimuth), 90-elevation, 'r', label='BICEP2 Spot', lw=3)
fill_between(np.radians(azimuth), 90-elevation-6.5, y2=90-elevation+6.5, color='red', alpha=0.3)
plot(np.radians(azimuth2), 90-elevation2, 'b', label='New Spot', lw=3)
fill_between(np.radians(azimuth2), 90-elevation2-6.5, y2=90-elevation2+6.5, color='blue', alpha=0.3)
#plot(np.radians(azimuth), azimuth*0+90-thetatower[0], 'm--', lw=3, label='{0:2.0f}m Tower at {1:2.0f}m'.format(htower, dtower[0]))
#plot(np.radians(azimuth), azimuth*0+90-thetatower[1], 'k--', lw=3, label='{0:2.0f}m Tower at {1:2.0f}m'.format(htower, dtower[1]))
plot(np.radians(aztower),90.-thetatower,'*',color='gold', markersize=20, label='Possible {0:2.0f}m Tower location: Az={1:3.0f}deg. D={2:3.0f}m'.format(htower, aztower, dtower))
legend(fontsize=10, loc='lower right',numpoints=1)
title('QUBIC Height: {0:2.0f}m'.format(hqubic))
savefig('newplot.png')
#savefig('qubic_azel_{0:4.1f}_{1:4.1f}.png'.format(av_el, delta_pt))




