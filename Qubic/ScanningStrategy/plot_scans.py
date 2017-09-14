from __future__ import division
from pyoperators import pcg
from pysimulators import profile
from qubic import (
    create_random_pointings, equ2gal, QubicAcquisition, PlanckAcquisition,
    QubicPlanckAcquisition, create_sweeping_pointings)
from qubic.data import PATH
from qubic.io import read_map
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np

nside = 128
maxiter = 1000
tol = 5e-6
racenter = 0.0      # deg
deccenter = -57.0   # deg
angspeed = 1        # deg/sec
delta_az = 30.      # deg
angspeed_psi = 0.   # deg/sec
maxpsi = 45.        # deg
nsweeps_el = 80
duration = 24       # hours
ts = 1           # seconds
np.random.seed(0)
sky = read_map(PATH + 'syn256_pol.fits')

center = equ2gal(racenter, deccenter)

# get the sampling model for San Antonio de los Cobres
sadlc = np.array([-24.18947, -66.472016])
sampling = create_sweeping_pointings(
        [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
        angspeed_psi, maxpsi, latitude=sadlc[0], longitude=sadlc[1])

mask = sampling.elevation > 30
sampling = sampling[mask]
clf()
plot(sampling.azimuth, sampling.elevation ,',')

radec = sampling.equatorial
ra = (( radec[:,0] + 720 + 180 ) % 360) - 180

clf()
scatter(ra, radec[:,1], c=sampling.elevation, marker=0, alpha=0.2)
cb=colorbar()
cb.set_label('Elevation [deg.]')
xlabel('RA [deg.]')
ylabel('DEC [deg.]')
savefig('sweeps_radec_sac.png')




# get the sampling model for Concordia
sampling = create_sweeping_pointings(
        [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
        angspeed_psi, maxpsi)


radec = sampling.equatorial
ra = (( radec[:,0] + 720 + 180 ) % 360) - 180

clf()
scatter(ra, radec[:,1], c=sampling.elevation, marker=0, alpha=0.2)
cb=colorbar()
cb.set_label('Elevation [deg.]')
xlabel('RA [deg.]')
ylabel('DEC [deg.]')
savefig('sweeps_radec.png')


clf()
plot(sampling.azimuth, sampling.elevation ,',')

clf()
subplot(1,1,1,polar=True)
plot(np.radians(sampling.azimuth), 90-sampling.elevation,'b.')
ylim(0,90)
savefig('azel.png')


ts=5.
nsweeps_el = 300
delta_az = 40.
sampling = create_sweeping_pointings(
        [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
        angspeed_psi, maxpsi)


detector_nep = 4.7e-17*np.sqrt(len(sampling) * sampling.period / (365 * 86400))
acq_qubic = QubicAcquisition(150, sampling[np.abs(sampling.elevation-50) < 20], 
    #nside=nside,
                             detector_nep=detector_nep)
                            
coverage_map = acq_qubic.get_coverage()
coverage_map = coverage_map / np.max(coverage_map)
angmax = hp.pix2ang(nside, coverage_map.argmax())
maxloc = np.array([np.degrees(angmax[1]), 90.-np.degrees(angmax[0])])

figure(0)
clf()
cov = hp.gnomview(coverage_map, rot=maxloc, reso=5, xsize=800, return_projected_map=True,sub=(2,1,1),min=0.0)
contour(cov,[0,0.01, 0.1])
subplot(2,1,2)
x, y = profile(cov)
x *= 5 / 60
plot(x, y/np.max(y))

maskok = coverage_map > 0.1
fsky=np.sum(coverage_map[maskok]) *1./len(maskok)


hp.mollview(coverage_map)
title('QUBIC : fsky > 0.1 : {0:3.1f}%'.format(fsky*100))
savefig('moll_qubic.png')


import scipy.ndimage
clf()
figure(0)
sh = np.shape(cov)
thex = np.arange(sh[0])*5./60
imshow(cov, extent=[np.min(thex), np.max(thex), np.min(thex), np.max(thex)], origin='lower')
xx, yy = meshgrid(np.array(thex),np.array(thex))
colorbar()
cs=contour(xx, yy, scipy.ndimage.gaussian_filter(cov, sigma=3), [0,0.01, 0.1, 0.5], colors=['w','w','w', 'w'])
clabel(cs, inline=1, fontsize=10)
title('QUBIC : fsky > 0.1 : {0:3.1f}%'.format(fsky*100))
xlabel('Angle in RA [deg.]')
ylabel('Angle in DEC [deg.]')
savefig('qubic_cov.png')






##### exemple of various fields
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
    
ndays = 10
ra_c,dec_c = circ_around_radec([racenter,deccenter],12., npoints=ndays)



ts=60.
nsweeps_el = 300
delta_az = 40.
coverage_map = np.zeros(12*nside**2)
for i in xrange(ndays):
    print(i)
    sampling = create_sweeping_pointings(
            [ra_c[i], dec_c[i]], duration, ts, angspeed, delta_az, nsweeps_el,
            angspeed_psi, maxpsi)
    detector_nep = 4.7e-17*np.sqrt(len(sampling) * sampling.period / (365 * 86400))
    acq_qubic = QubicAcquisition(150, sampling[np.abs(sampling.elevation-50) < 20], nside=nside,
                             detector_nep=detector_nep)                        
    coverage_map += acq_qubic.get_coverage()

coverage_map = coverage_map / np.max(coverage_map)
angmax = hp.pix2ang(nside, coverage_map.argmax())
maxloc = np.array([np.degrees(angmax[1]), 90.-np.degrees(angmax[0])])

figure(0)
clf()

res=5.
cov = hp.gnomview(coverage_map, rot=maxloc, reso=res, xsize=800, return_projected_map=True,sub=(2,1,1),min=0.0)
contour(cov,[0,0.01, 0.1])
subplot(2,1,2)
x, y = profile(cov)
x *= res / 60
plot(x, y/np.max(y))

maskok = coverage_map > 0.1
fsky=np.sum(coverage_map[maskok]) *1./len(maskok)


hp.mollview(coverage_map)
title('QUBIC : fsky > 0.1 : {0:3.1f}%'.format(fsky*100))
savefig('moll_qubic_large.png')

import scipy.ndimage
figure(0)
sh = np.shape(cov)
thex = np.arange(sh[0])*res/60
imshow(cov, extent=[np.min(thex), np.max(thex), np.min(thex), np.max(thex)], origin='lower')
xx, yy = meshgrid(np.array(thex),np.array(thex))
colorbar()
cs=contour(xx, yy, scipy.ndimage.gaussian_filter(cov, sigma=3), [0,0.01, 0.1, 0.5], colors=['w','w','w', 'w'])
clabel(cs, inline=1, fontsize=10)
title('QUBIC : fsky > 0.1 : {0:3.1f}%'.format(fsky*100))
xlabel('Angle in RA [deg.]')
ylabel('Angle in DEC [deg.]')
savefig('qubic_cov_large.png')








