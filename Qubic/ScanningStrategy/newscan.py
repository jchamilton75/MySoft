from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from pyoperators import DiagonalOperator, PackOperator, pcg, rules_inplace
from pysimulators import (SphericalEquatorial2GalacticOperator, CartesianEquatorial2GalacticOperator,
                          CartesianHorizontal2EquatorialOperator)
from pyoperators import (
    Cartesian2SphericalOperator, Rotation3dOperator,
    Spherical2CartesianOperator)
from pysimulators import (
    PointingHorizontal, CartesianEquatorial2GalacticOperator,
    CartesianEquatorialHorizontalOperator,
    CartesianHorizontal2EquatorialOperator)
import pysimulators
from pyoperators import Spherical2CartesianOperator
from qubic import QubicAcquisition, QubicInstrument, create_sweeping_pointings
from astropy.time import Time, TimeDelta


from ScanningStrategy import pointings_modbyJC
#from ScanningStrategy.pointings_modbyJC import create_sweeping_pointings


################### Get a center around racenter, deccenter
nside = 256
racenter = 0
deccenter = -57.0
deltatheta = 15.

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

#### Co-scan and Cross-scan sampling w.r.t. ang speed and ang range
nn = 100
angspeed = linspace(0., 1., nn)
delta_scan = linspace(10, 30., nn)
# Co-scan :
co_scan_sampling = angspeed / fsampling * 60 #arcmin/sample
clf()
plot(angspeed, co_scan_sampling)
xlabel('Azimuth Angular speed [deg/sec]')
ylabel('Co-Scan sampling [arcmin]')

angspeed2, delta_scan2 = meshgrid(angspeed, delta_scan)
co_scan_sampling = angspeed2 / fsampling * 60 #arcmin/sample
clf()
imshow(co_scan_sampling, interpolation = 'nearest', extent = [min(angspeed), max(angspeed), min(delta_scan), max(delta_scan)], aspect='auto',origin='lower')
colorbar()
xlabel('Azimuth Angular speed [deg/sec]')
ylabel('Azimuth scan extent [deg]')
title('Co-scan sampling [arcmin]')
levels=[0.1,0.5,1.,2,5,10]
cs=contour(angspeed, delta_scan,co_scan_sampling, levels=levels,colors='w')
fmt = '%r arcmin'
clabel(cs,levels,inline=True,fmt=fmt,fontsize=10,colors='w')



# Cross scan sampling
angspeed2, delta_scan2 = meshgrid(angspeed, delta_scan)
dtscan = delta_scan2 / angspeed2
clf()
imshow(dtscan, interpolation = 'nearest', extent = [min(angspeed), max(angspeed), min(delta_scan), max(delta_scan)], aspect='auto',origin='lower',vmin=60,vmax=200)
colorbar()
xlabel('Azimuth Angular speed [deg/sec]')
ylabel('Azimuth scan extent [deg]')
title('Scan duration [sec]')
levels=[30, 60, 90, 120, 180]
cs=contour(angspeed, delta_scan, dtscan, levels=levels,colors='w')
fmt = '%r sec'
clabel(cs,levels,inline=True,fmt=fmt,fontsize=10,colors='w')

nt = 1000
DOMECLAT = -(75 + 6 / 60)
DOMECLON = 123 + 20 / 60
DEFAULT_DATE_OBS = '2016-01-01 00:00:00'
time = TimeDelta(linspace(0,24*60*60,nt), format='sec')
rotation = (
    Cartesian2SphericalOperator('azimuth,elevation', degrees=True) *
    CartesianEquatorial2HorizontalOperator('NE', time, DOMECLAT, DOMECLON) *
    Rotation3dOperator("ZY'", racenter, 90 - deccenter, degrees=True) *
    Spherical2CartesianOperator('zenith,azimuth', degrees=True))
theta = np.zeros(nt)
phi = np.zeros(nt)
coords = rotation(np.asarray(np.asarray([theta.T, phi.T]).T))
azimuth = coords[:, 0]
elevation = coords[:, 1]

clf()
subplot(2,1,1)
plot(time.sec,elevation)
xlabel('Time [sec]')
ylabel('Elevation [deg]')
subplot(2,1,2)
plot(time.sec[:-1],60*60*np.abs(numpy.diff(elevation)/numpy.diff(time.sec)))
xlabel('Time [sec]')
ylabel('dEl/dt [arcmin/min]')

mdel_dt=np.mean(abs(numpy.diff(elevation)/numpy.diff(time.sec)))
cross_scan_sampling = dtscan*mdel_dt*60
plot(time.sec[:-1],time.sec[:-1]*0+mdel_dt*3600,'r--')

clf()
imshow(cross_scan_sampling, interpolation = 'nearest', extent = [min(angspeed), max(angspeed), min(delta_scan), max(delta_scan)], aspect='auto',origin='lower',vmin=0.1,vmax=10)
colorbar()
xlabel('Azimuth Angular speed [deg/sec]')
ylabel('Azimuth scan extent [deg]')
title('Mean Cross-scan sampling [arcmin]')
levels=[0.1,0.5,1.,2,5,10]
cs=contour(angspeed, delta_scan, cross_scan_sampling, levels=levels,colors='w')
fmt = '%r arcmin'
clabel(cs,levels,inline=True,fmt=fmt,fontsize=10,colors='w')


clf()
subplot(2,2,1)
imshow(co_scan_sampling, interpolation = 'nearest', extent = [min(angspeed), max(angspeed), min(delta_scan), max(delta_scan)], aspect='auto',origin='lower')
colorbar()
xlabel('Azimuth Angular speed [deg/sec]')
ylabel('Azimuth scan extent [deg]')
title('Co-scan sampling [arcmin]')
levels=[0.1,0.5,1.,2,5,10]
cs=contour(angspeed, delta_scan,co_scan_sampling, levels=levels,colors='w')
fmt = '%r arcmin'
clabel(cs,levels,inline=True,fmt=fmt,fontsize=10,colors='w')

subplot(2,2,2)
imshow(cross_scan_sampling, interpolation = 'nearest', extent = [min(angspeed), max(angspeed), min(delta_scan), max(delta_scan)], aspect='auto',origin='lower',vmin=0.1,vmax=10)
colorbar()
xlabel('Azimuth Angular speed [deg/sec]')
ylabel('Azimuth scan extent [deg]')
title('Mean Cross-scan sampling [arcmin]')
levels=[0.1,0.5,1.,2,5,10]
cs=contour(angspeed, delta_scan, cross_scan_sampling, levels=levels,colors='w')
fmt = '%r arcmin'
clabel(cs,levels,inline=True,fmt=fmt,fontsize=10,colors='w')

subplot(2,2,3)
imshow(co_scan_sampling/cross_scan_sampling, interpolation = 'nearest', extent = [min(angspeed), max(angspeed), min(delta_scan), max(delta_scan)], aspect='auto',origin='lower')
colorbar()
xlabel('Azimuth Angular speed [deg/sec]')
ylabel('Azimuth scan extent [deg]')
title('Ratio Co Sampling / <Cross sampling>')
levels=[0.1,0.5,1.,2,5,10]
cs=contour(angspeed, delta_scan, co_scan_sampling/cross_scan_sampling , levels=levels,colors='w')
fmt = '%r'
clabel(cs,levels,inline=True,fmt=fmt,fontsize=10,colors='w')

subplot(2,2,4)
imshow(dtscan, interpolation = 'nearest', extent = [min(angspeed), max(angspeed), min(delta_scan), max(delta_scan)], aspect='auto',origin='lower',vmin=10,vmax=200)
colorbar()
xlabel('Azimuth Angular speed [deg/sec]')
ylabel('Azimuth scan extent [deg]')
title('Scan duration [sec]')
levels=[30, 60, 90, 120, 180]
cs=contour(angspeed, delta_scan, dtscan, levels=levels,colors='w')
fmt = '%r sec'
clabel(cs,levels,inline=True,fmt=fmt,fontsize=10,colors='w')

savefig('sky_sampling_100Hz.png')






###### let's try
angspeed = 0.3
delta_az = 15
DOMECLAT = -(75 + 6 / 60)
DOMECLON = 123 + 20 / 60
nside = 256
racenter = 0
deccenter = -57.0
sampling_period = 0.01
duration = 24*3600
nelt = duration/sampling_period
racirc, deccirc = circ_around_radec([racenter,deccenter],delta_az)
racirc2, deccirc2 = circ_around_radec([racenter,deccenter],delta_az/5)
clf()
subplot(111,projection='mollweide')
plot(np.radians(racenter),np.radians(deccenter),'ro')
plot(np.radians(racirc), np.radians(deccirc),'b.')
plot(np.radians(racirc2), np.radians(deccirc2),'g.')

###### Sampling at each time sample
t0 = Time(['2016-01-01 00:00:00.0'], scale='utc')
dt = TimeDelta(np.arange(nelt)*sampling_period, format='sec')
op = (pysimulators.SphericalEquatorial2HorizontalOperator('NE', t0 + dt, pointings_modbyJC.DOMECLAT, pointings_modbyJC.DOMECLAT, degrees=True))
azelcenter = op(np.array([np.zeros(nelt)+racenter, np.zeros(nelt)+deccenter]).T)

clf()
subplot(211)
plot(dt.jd,azelcenter[:,0])
subplot(212)
plot(dt.jd,azelcenter[:,1])


###### Sampling at each end of the const elevation sweeps
scan_duration = delta_az / angspeed
nelt_scan = scan_duration / sampling_period
nscan = duration / scan_duration
dt_scan = TimeDelta(np.arange(nscan) * scan_duration, format='sec')
op_scan = (pysimulators.SphericalEquatorial2HorizontalOperator('NE', t0 + dt_scan, pointings_modbyJC.DOMECLAT, pointings_modbyJC.DOMECLAT, degrees=True, block_column=True))
azelcenter_scan = op_scan(np.array([racenter, deccenter]).T)

clf()
subplot(211)
plot(dt_scan.jd,azelcenter_scan[:,0])
subplot(212)
plot(dt_scan.jd,azelcenter_scan[:,1])

azelcenter_scan = op_scan(np.array([racenter,deccenter]).T)
azelcirc = np.zeros((len(racirc), nscan, 2))
azelcirc2 = np.zeros((len(racirc), nscan, 2))
for i in np.arange(len(racirc)):
    azelcirc[i,:,:] = op_scan(np.array([racirc[i], deccirc[i]]).T)
    azelcirc2[i,:,:] = op_scan(np.array([racirc2[i], deccirc2[i]]).T)

clf()
plot(azelcenter_scan[:,0],azelcenter_scan[:,1],'k.')
plot(azelcirc[:,0,0],azelcirc[:,0,1],',')
xlim(0,360)
ylim(0,90)
title('Location of the patch')
xlabel('Azimuth [deg]')
ylabel('Elevation [deg]')
nn=100
for i in np.arange(nscan):
    if (nn*int(i/nn) == i):
        plot(azelcirc[:,i,0],azelcirc[:,i,1],',')
        plot(azelcirc2[:,i,0],azelcirc2[:,i,1],',')

savefig('patchlocation.png')

minaz = np.zeros(nscan)
maxaz = np.zeros(nscan)
minel = np.zeros(nscan)
maxel = np.zeros(nscan)
minaz2 = np.zeros(nscan)
maxaz2 = np.zeros(nscan)
minel2 = np.zeros(nscan)
maxel2 = np.zeros(nscan)
for i in np.arange(nscan):
    minaz[i] = np.min(azelcirc[:,i,0])
    minel[i] = np.min(azelcirc[:,i,1])
    maxaz[i] = np.max(azelcirc[:,i,0])
    maxel[i] = np.max(azelcirc[:,i,1])
    minaz2[i] = np.min(azelcirc2[:,i,0])
    minel2[i] = np.min(azelcirc2[:,i,1])
    maxaz2[i] = np.max(azelcirc2[:,i,0])
    maxel2[i] = np.max(azelcirc2[:,i,1])

# derivative of elevation
der_el = numpy.diff(azelcenter_scan[:,1])/numpy.diff(dt_scan.sec)

## Start a UTC = 0 with constant elevation scans between minaz and maxaz and at minel or maxel depending on sign of derivative
if sign(der_el[0]) == 1:
    start_el = maxel2[0]
    end_el = minel2[0]
else:
    start_el = minel2[0]
    end_el = maxel2[0]

the_el = np.zeros(nscan)+start_el
for i in np.arange(nscan):
    if (start_el < minel2[i]) or (start_el > maxel2[i]):
        if sign(der_el[i]) == 1:
            start_el = maxel2[i]
        else:
            start_el = minel2[i]
    the_el[i] = start_el
        
    
clf()
plot(dt_scan.sec/3600,azelcenter_scan[:,1])
plot(dt_scan.sec/3600,minel,':')
plot(dt_scan.sec/3600,maxel,':')
plot(dt_scan.sec/3600,minel2,':')
plot(dt_scan.sec/3600,maxel2,':')
plot(dt_scan.sec/3600,the_el)





