from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from pyoperators import DiagonalOperator, PackOperator, pcg, rules_inplace
from pysimulators import SphericalEquatorial2GalacticOperator, CartesianEquatorial2GalacticOperator, CartesianHorizontal2EquatorialOperator
from pyoperators import Spherical2CartesianOperator
from qubic import QubicAcquisition, QubicInstrument, create_sweeping_pointings

from ScanningStrategy import pointings_modbyJC
#from ScanningStrategy.pointings_modbyJC import create_sweeping_pointings



nside = 256
racenter = 0.0
deccenter = -57.0
angspeed = 0.3    # deg/sec
delta_az = 20.
angspeed_psi = 0
maxpsi = 15.
nsweeps_el = int(120*angspeed)
duration = 24   # hours
ts = 0.1         # seconds
decrange=0
decspeed=0
pointing = pointings_modbyJC.create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi, decrange=decrange, decspeed=decspeed,recenter=True)
pointing.angle_hwp = np.random.random_integers(0, 7, pointing.size) * 22.5
ntimes = len(pointing)

# get instrument model with only one detector
idetector = 231
instrument = QubicInstrument('monochromatic')
instrument.plot()
mask_packed = np.ones(len(instrument.detector.packed), bool)
mask_packed[idetector] = False
mask_unpacked = instrument.unpack(mask_packed)
instrument = QubicInstrument('monochromatic', removed=mask_unpacked)

obs = QubicAcquisition(instrument, pointing)
convolution = obs.get_convolution_peak_operator()
projection = obs.get_projection_peak_operator(kmax=0)
hwp = obs.get_hwp_operator()
polarizer = obs.get_polarizer_operator()
coverage = projection.pT1()
mask = coverage > 0
projection.restrict(mask)
pack = PackOperator(mask, broadcast='rightward')
ra,dec=pointings_modbyJC._hor2equ(pointing.azimuth, pointing.elevation, pointing.latitude, pointing.time/3600)
ns_scan = int(delta_az / angspeed / 0.1)
ns_tot = ns_scan * 2
ichunk = (np.arange(len(pointing)) / ns_tot / nsweeps_el).astype(numpy.int64)
isweep = ((np.arange(len(pointing)) / ns_tot).astype(numpy.int64)) % nsweeps_el
clf()
subplot(2,2,1)
plot(pointing.azimuth,pointing.elevation)
e2g = SphericalEquatorial2GalacticOperator(degrees=True)
center = e2g([racenter, deccenter])
hp.gnomview(coverage, rot=[racenter,deccenter], reso=3, xsize=400, min=0, max=np.max(coverage),
                    title='Coverage',sub=(2,2,2),coord=['G','C'])
subplot(2,1,2)
step=10
import time
for chunk in np.arange(np.max(ichunk)):
    mask = ichunk == chunk
    thera = ra[mask]
    thedec = dec[mask]
    theisweep = isweep[mask]
    masksweep = (theisweep % step) == 0
    plot(thera[masksweep],thedec[masksweep])
    bla=gca()
    #bla.set_aspect(1./cos(np.radians(deccenter)))
    draw()
    time.sleep(0.01)

clf()
max=10000
plot(ra[0:max],dec[0:max],',')


def get_coverage_onedet(idetector=231, angspeed=1., delta_az=15., angspeed_psi=0., maxpsi=15., nsweeps_el=100, duration=24, ts=1., decrange=2., decspeed=2., recenter=False):
    print('##### Getting Coverage for: ') 
    print('## idetector = '+str(idetector))
    print('## duration = '+str(duration))
    print('## ts = '+str(ts))
    print('## recenter = '+str(recenter))
    print('## angspeed = '+str(angspeed))
    print('## delta_az = '+str(delta_az))
    print('## nsweeps_el = '+str(nsweeps_el))
    print('## decrange = '+str(decrange))
    print('## decspeed = '+str(decspeed))
    print('## angspeed_psi = '+str(angspeed_psi))
    print('## maxpsi = '+str(maxpsi))
    print('##########################')
    nside = 256
    racenter = 0.0
    deccenter = -57.0
    pointing = pointings_modbyJC.create_sweeping_pointings(
        [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
        angspeed_psi, maxpsi, decrange=decrange, decspeed=decspeed, recenter=recenter)
    pointing.angle_hwp = np.random.random_integers(0, 7, pointing.size) * 22.5
    ntimes = len(pointing)

    # get instrument model with only one detector
    instrument = QubicInstrument('monochromatic')
    mask_packed = np.ones(len(instrument.detector.packed), bool)
    mask_packed[idetector] = False
    mask_unpacked = instrument.unpack(mask_packed)
    instrument = QubicInstrument('monochromatic', removed=mask_unpacked)
    
    obs = QubicAcquisition(instrument, pointing)
    convolution = obs.get_convolution_peak_operator()
    projection = obs.get_projection_peak_operator(kmax=0)
    coverage = projection.pT1()
    return coverage

idetector=231
cov=get_coverage_onedet(idetector=idetector,decrange=1,decspeed=2)
clf()
hp.gnomview(cov,rot=[racenter,deccenter], reso=3, xsize=400, min=0, max=np.max(cov),
                    title='Coverage',sub=(1,1,1),coord=['G','C'])

# Flatness estimator to be mimnimized (the closer to 1, the better)
# 2 for a gaussian
thr,phr=hp.pix2ang(nside,np.arange(12*nside**2))
covg=np.exp(-thr**2/(2*(np.radians(14)/2.35)**2))
covg=covg/np.max(covg)
np.sum(covg)/np.sum(covg**2)
# 1 for a top-hat
thr,phr=hp.pix2ang(nside,np.arange(12*nside**2))
covg=np.zeros(12*nside**2)
covg[thr < 0.05]=3
covg=covg/np.max(covg)
np.sum(covg)/np.sum(covg**2)


########## Dec Range
dr=np.linspace(0,6,30)
eta=np.zeros(len(dr))
fsky=np.zeros(len(dr))
for i in np.arange(len(dr)):
    print(i)
    cov=get_coverage_onedet(idetector=idetector,decrange=dr[i])
    cov=cov/np.max(cov)
    eta[i]=np.sum(cov)/np.sum(cov**2)
    fsky[i]=100*np.sum(cov)/(12*nside**2)

clf()
subplot(2,1,1)
plot(dr,eta)
subplot(2,1,2)
plot(dr,fsky)

########## Dec Speed
ds=np.linspace(0,6,100)
eta=np.zeros(len(ds))
fsky=np.zeros(len(ds))
for i in np.arange(len(ds)):
    print(i)
    cov=get_coverage_onedet(idetector=idetector,decspeed=ds[i])
    cov=cov/np.max(cov)
    eta[i]=np.sum(cov)/np.sum(cov**2)
    fsky[i]=100*np.sum(cov)/(12*nside**2)

clf()
subplot(2,1,1)
plot(ds,eta)
subplot(2,1,2)
plot(ds,fsky)

###### minimize eta w.r.t. parameters ###########################################
import iminuit

class MyCost:
    def __init__(self, idetector, recenter, duration, ts, doplot, costdef):
        self.idetector = idetector
        self.recenter = recenter
        self.duration = duration
        self.ts = ts
        self.doplot = doplot
        self.costdef = costdef
    def __call__(self, *pars):
        idetector = self.idetector
        angspeed = pars[0]
        deltaaz = pars[1]
        nsweepsel = int(pars[2])
        decrange = pars[3]
        decspeed = pars[4]
        angspeedpsi = pars[5]
        maxpsi = pars[6]
        cov = get_coverage_onedet(idetector=idetector, angspeed=angspeed,
                                  delta_az=deltaaz, angspeed_psi=angspeedpsi,
                                  maxpsi=maxpsi, nsweeps_el=nsweepsel,
                                  duration=self.duration, ts=self.ts,
                                  decrange=decrange, decspeed=decspeed, recenter=self.recenter)
        cov = cov / np.max(cov)
        eta = np.sum(cov) / np.sum(cov**2)
        self.cov = cov
        if self.costdef is 'stddev':
            cost = np.std(cov[cov != 0])
        elif self.costdef is 'eta':
            cost = eta
        elif self.costdef is 'omega':
            cost = np.sum(cov)
            
        if self.doplot:
            clf()
            hp.gnomview(cov,rot=[racenter,deccenter], reso=5, xsize=400, min=0, max=np.max(cov),
                        title='Coverage: '+self.costdef+'='+str(cost),sub=(1,1,1),coord=['G','C'])
            draw()
        print (angspeed, nsweepsel, decrange, decspeed, cost)
        return cost

### Instatiate function to be minimized
idetector = 231
recenter = False
duration = 24.
ts = 1.
doplot = True
costdef = 'eta'
cost = MyCost(idetector, recenter, duration, ts, doplot, costdef)

### Parameters
parnames = ['angspeed', 'deltaaz', 'nsweepsel', 'decrange', 'decspeed', 'angspeedpsi', 'maxpsi']
guess = [.5, 15., 100, 2., 2., 0., 15.]
kwdargs = dict(zip(parnames,guess))
errornorm=0.001
kwdargs['limit_angspeed'] = (0., 10.)
kwdargs['limit_deltaaz'] = (10., 30.)
kwdargs['limit_nsweepsel'] = (1, 500)
kwdargs['limit_decrange'] = (0., 6.)
kwdargs['limit_decspeed'] = (0., 6.)
kwdargs['limit_angspeedpsi'] = (0., 10.)
kwdargs['limit_maxpsi'] = (0., 45.)
kwdargs['fix_angspeed'] = False
kwdargs['fix_deltaaz'] = False
kwdargs['fix_nsweepel'] = False
kwdargs['fix_decspeed'] = True
kwdargs['fix_decrange'] = True
kwdargs['fix_angspeedpsi'] = True
kwdargs['fix_maxpsi'] = True
kwdargs['error_angspeed'] = 0.2 * errornorm
kwdargs['error_deltaaz'] = 1. * errornorm
kwdargs['error_nsweepel'] = 10. * errornorm
kwdargs['error_decspeed'] = 0.2 * errornorm
kwdargs['error_decrange'] = 0.2 * errornorm
kwdargs['error_angspeedpsi'] = 0.2 * errornorm
kwdargs['error_maxpsi'] = 1. * errornorm

### Minuit
m = iminuit.Minuit(cost, forced_parameters=parnames, errordef=0.001, **kwdargs)
m.migrad()
########################################################################################


############ Brute force exploration ##########
idetector = np.arange(248)
recenter = False
duration = 24.
ts = 1.
doplot = True
costdef = 'eta'
cost = MyCost(idetector, recenter, duration, ts, doplot, costdef)

all_eta = []
all_omega = []
all_sigma = []
angspeed_vals = []
deltaaz_vals = []
nsweepsel_vals = []
decrange_vals = []
decspeed_vals = []
angspeedpsi_vals = []
maxpsi_vals = []
while True:
    angspeed = np.random.uniform(low=0., high=10.)
    deltaaz = np.random.uniform(low=10., high=30.)
    nsweepsel = np.random.uniform(low=1, high=500)
    decrange = np.random.uniform(low=0., high=6.)
    decspeed = np.random.uniform(low=0., high=6.)
    angspeedpsi = np.random.uniform(low=0., high=10.)
    maxpsi = np.random.uniform(low=0., high=45.)

    pars = np.array([angspeed, deltaaz, nsweepsel, decrange,
                     decspeed, angspeedpsi, maxpsi])
    theeta = cost(*pars)
    good = cost.cov != 0

    all_eta.append(theeta)
    angspeed_vals.append(angspeed)
    deltaaz_vals.append(deltaaz)
    nsweepsel_vals.append(nsweepsel)
    decrange_vals.append(decrange)
    decspeed_vals.append(decspeed)
    angspeedpsi_vals.append(angspeedpsi)
    maxpsi_vals.append(maxpsi)
    all_sigma.append(np.std(cost.cov[good]))
    all_omega.append(np.sum(cost.cov[good])*4*pi/len(cost.cov))
    print(len(all_omega))
    
clf()
scatter(all_eta,all_sigma,c=np.array(all_omega))
xlabel('$\eta$')
ylabel('$\sigma$')
title('Colors : $\Omega$')
colorbar()

clf()
scatter(all_eta,all_omega,c=np.array(all_sigma))
xlabel('$\eta$')
ylabel('$\Omega$')
title('Colors : $\sigma$')
colorbar()

clf()
scatter(all_sigma,all_omega,c=np.array(all_eta))
xlabel('$\sigma$')
ylabel('$\Omega$')
title('Colors : $\eta$')
colorbar()

##### Analyse
eta = np.array(all_eta)
omega = np.array(all_omega)
sigma = np.array(all_sigma)
angspeed = np.array(angspeed_vals)
deltaaz = np.array(deltaaz_vals)
nsweepsel = np.array(nsweepsel_vals)
decrange =  np.array(decrange_vals)
decspeed = np.array(decspeed_vals)
angspeedpsi = np.array(angspeedpsi_vals)
maxpsi = np.array(maxpsi_vals)

#cut = (eta < 1.65) & (omega < 0.2) & (maxpsi < 15)
cut = (eta < 1.75) & (omega < 0.05)
angspeed[cut].size


params=np.array([angspeed[cut], deltaaz[cut], nsweepsel[cut], decrange[cut],
                 decspeed[cut], angspeedpsi[cut], maxpsi[cut]])

i=0
theeta = cost(*(params[:,i]))
good = cost.cov != 0
thesigma=np.std(cost.cov[good])
theomega=np.sum(cost.cov[good])*4*pi/len(cost.cov)
print(theeta,thesigma,theomega)
print(eta[cut][i],sigma[cut][i],omega[cut][i])

i=0
cost_alldet=MyCost(np.arange(992*2), recenter, duration, ts, doplot, costdef)
theeta = cost_alldet(*(params[:,i]))
good = cost_alldet.cov != 0
thesigma=np.std(cost_alldet.cov[good])
theomega=np.sum(cost_alldet.cov[good])*4*pi/len(cost.cov)
print(theeta,thesigma,theomega)
print(eta[cut][i],sigma[cut][i],omega[cut][i])

allcov=np.zeros((len(angspeed[cut]),12*nside**2))
alleta=np.zeros(len(angspeed[cut]))
allsigma=np.zeros(len(angspeed[cut]))
allomega=np.zeros(len(angspeed[cut]))
thecost=MyCost(np.arange(992), recenter, duration, ts, doplot, costdef)
for i in np.arange(len(angspeed[cut])):
    print(i,len(angspeed[cut]))
    alleta[i] = thecost(*(params[:,i]))
    good = thecost.cov != 0
    allsigma[i] = np.std(thecost.cov[good])
    allomega[i] = np.sum(thecost.cov[good])*4*pi/len(thecost.cov)
    allcov[i,:]=thecost.cov

clf()
for i in np.arange(9):
    hp.gnomview(allcov[i,:],rot=[racenter,deccenter], reso=5, xsize=400, min=0, max=1,
                        title='Coverage: ',sub=(3,3,i+1),coord=['G','C'])


clf()
i=2
a0=hp.gnomview(allcov0[i,:],rot=[racenter,deccenter], reso=5, xsize=400, min=0, max=1,
            title='Coverage: 0',sub=(2,3,1),coord=['G','C'],return_projected_map=True)
a1=hp.gnomview(allcov1[i,:],rot=[racenter,deccenter], reso=5, xsize=400, min=0, max=1,
            title='Coverage: 1',sub=(2,3,2),coord=['G','C'],return_projected_map=True)
a2=hp.gnomview(allcov2[i,:],rot=[racenter,deccenter], reso=5, xsize=400, min=0, max=1,
            title='Coverage: 2',sub=(2,3,4),coord=['G','C'],return_projected_map=True)
a3=hp.gnomview(allcov3[i,:],rot=[racenter,deccenter], reso=5, xsize=400, min=0, max=1,
            title='Coverage: 3',sub=(2,3,5),coord=['G','C'],return_projected_map=True)
atot=hp.gnomview(allcov[i,:],rot=[racenter,deccenter], reso=5, xsize=400, min=0, max=1,
            title='Coverage: Tot',sub=(2,3,3),coord=['G','C'],return_projected_map=True)
subplot(2,3,6)
imshow(atot)
contour(a0,[0.25,0.5,0.75],colors=['blue'],linestyles=['--','-','--'])
contour(a1,[0.25,0.5,0.75],colors=['red'],linestyles=['--','-','--'])
contour(a2,[0.25,0.5,0.75],colors=['green'],linestyles=['--','-','--'])
contour(a3,[0.25,0.5,0.75],colors=['black'],linestyles=['--','-','--'])



clf()
for i in np.arange(9):
    a0=hp.gnomview(allcov0[i,:],rot=[racenter,deccenter], reso=5, xsize=400, min=0, max=1,
                   title='Coverage: 0',sub=(3,3,i+1),coord=['G','C'],return_projected_map=True)
    a1=hp.gnomview(allcov1[i,:],rot=[racenter,deccenter], reso=5, xsize=400, min=0, max=1,
                   title='Coverage: 1',sub=(3,3,i+1),coord=['G','C'],return_projected_map=True)
    a2=hp.gnomview(allcov2[i,:],rot=[racenter,deccenter], reso=5, xsize=400, min=0, max=1,
                   title='Coverage: 2',sub=(3,3,i+1),coord=['G','C'],return_projected_map=True)
    a3=hp.gnomview(allcov3[i,:],rot=[racenter,deccenter], reso=5, xsize=400, min=0, max=1,
                   title='Coverage: 3',sub=(3,3,i+1),coord=['G','C'],return_projected_map=True)
    atot=hp.gnomview(allcov[i,:],rot=[racenter,deccenter], reso=5, xsize=400, min=0, max=1,
            title='Coverage: Tot',sub=(3,3,i+1),coord=['G','C'],return_projected_map=True)
    subplot(3,3,i+1)
    imshow(atot)
    contour(a0,[0.25,0.5,0.75],colors=['blue'],linestyles=['--','-','--'])
    contour(a1,[0.25,0.5,0.75],colors=['red'],linestyles=['--','-','--'])
    contour(a2,[0.25,0.5,0.75],colors=['green'],linestyles=['--','-','--'])
    contour(a3,[0.25,0.5,0.75],colors=['black'],linestyles=['--','-','--'])
