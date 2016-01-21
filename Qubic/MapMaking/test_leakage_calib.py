#source ~/software_pola_tmp/bin/activate

from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import pycamb
from pyoperators import (
    DenseOperator, DegreesOperator, DiagonalOperator, RadiansOperator,
    Cartesian2SphericalOperator, Spherical2CartesianOperator, pcg)
from pysimulators import (
    CartesianEquatorial2GalacticOperator, ProjectionOperator, FitsArray)
from pysimulators import (
    ProjectionOperator, SphericalEquatorial2GalacticOperator, FitsArray)
from qubic import QubicAcquisition, QubicInstrument, create_sweeping_pointings, create_random_pointings
from MapMaking import MapMaking as mm

###### Input Power spectrum ###################################
################# Input Power spectrum ###################################
## parameters from Planck 2013 results XV. CMB Power spectra ..., table 8, Planck+WP
## Archiv 1303.5075
import pycamb
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
         'tensor_ratio':0.1,'WantTensors':True,'scalar_amp':scalar_amp,'DoLensing':False}

lmax = 1200
ell = np.arange(1,lmax+1)
T,E,B,X = pycamb.camb(lmax+1,**params)
fact = (ell*(ell+1))/(2*np.pi)
spectra = [ell, T/fact, E/fact, B/fact, X/fact]

clf()
plot(ell,np.sqrt(spectra[1]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TT}$')
plot(ell,np.sqrt(abs(spectra[4])*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TE}$')
plot(ell,np.sqrt(spectra[2]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{EE}$')
plot(ell,np.sqrt(spectra[3]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{BB} (r=0.1)$')
yscale('log')
xlim(0,600)
ylim(0.01,100)
xlabel('$\ell$')
ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
legend(loc='lower right',frameon=False)


################## Input Maps #############################################
nside=128
lmax=3*nside
ell=spectra[0]
maskl=ell<(lmax+1)
bl=np.ones(maskl.size)
fwhmrad=0.5*np.pi/180
mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
hp.mollview(mapi)
hp.mollview(mapq)
hp.mollview(mapu)
maps=np.transpose(np.array([mapi,mapq,mapu]))


################# Qubic Instrument #########################################
qubic = QubicInstrument('monochromatic',nside=nside)
detectors=qubic.detector.packed
ndet = len(detectors)
clf()
subplot(2,1,1,aspect='equal')
plot(detectors.center[0:ndet/2,0],detectors.center[0:ndet/2,1],'ro')
subplot(2,1,2,aspect='equal')
plot(detectors.center[ndet/2:2*ndet/2,0],detectors.center[ndet/2:2*ndet/2,1],'bo')


################# random pointings ##########################################
racenter = 0.0
deccenter = -57.0
dtheta=15.
npointings=2000
rnd_pointing=create_random_pointings([racenter, deccenter],npointings,dtheta)
op = (DegreesOperator() * Cartesian2SphericalOperator('azimuth,elevation') *
      CartesianEquatorial2GalacticOperator() *
      Spherical2CartesianOperator('azimuth,elevation') *
      RadiansOperator())
center = op([racenter, deccenter])

################# TOD
tod=mm.map2tod(maps,rnd_pointing,qubic,kmax=2)

################# Spoil bolometers calibration
calerror=1e-2
tod_spoiled=tod.copy()
rndcal=np.random.normal(loc=1.,scale=calerror,size=(len(detectors),2))
for i in np.arange(len(detectors)):
    for j in np.arange(2):
        tod_spoiled[i,:,j]=rndcal[i,j]*tod[i,:,j]

################# Maps
output_maps_all,call=mm.tod2map(tod,rnd_pointing,qubic,disp=False)
output_maps_all_spoiled,call_spoiled=mm.tod2map(tod_spoiled,rnd_pointing,qubic,disp=False)
output_maps_det,cdet=mm.tod2map_perdet(tod,rnd_pointing,qubic,disp=False)
output_maps_det_spoiled,cdet_spoiled=mm.tod2map_perdet(tod_spoiled,rnd_pointing,qubic,disp=False)

covmin=np.arange(12*nside**2)
for i in np.arange(12*nside**2):
    covmin[i]=np.min([call[i],call_spoiled[i],cdet[i],cdet_spoiled[i]])

clf()
thr = 100
mm.display(output_maps_all,'',center=center,lim=[200,3,3],reso=15,mask=covmin < thr)
mm.display(output_maps_det,'',center=center,lim=[200,3,3],reso=15,mask=covmin < thr)
mm.display(output_maps_all_spoiled,'',center=center,lim=[200,3,3],reso=15,mask=covmin < thr)
mm.display(output_maps_det_spoiled,'',center=center,lim=[200,3,3],reso=15,mask=covmin < thr)

rr=0.2
figure()
mm.display(output_maps_all_spoiled-output_maps_all,'',center=center,lim=[rr,rr,rr],reso=15,mask=covmin < 40)
rr=0.05
figure()
mm.display(output_maps_det_spoiled-output_maps_det,'',center=center,lim=[rr,rr,rr],reso=15,mask=covmin < 40)

def statstr(vec):
    m=np.mean(vec)
    s=np.std(vec)
    return '{0:.4f} +/- {1:.4f}'.format(m,s)

clf()
rr=0.5
cut=covmin > 40
aa=output_maps_all_spoiled-output_maps_all
bb=output_maps_det_spoiled-output_maps_det
subplot(3,1,1)
hist(aa[cut,0],bins=100,range=[-rr,rr],color='blue',alpha=0.2,label='All detectors:'+statstr(aa[cut,0]))
hist(bb[cut,0],bins=100,range=[-rr,rr],color='red',alpha=0.2,label='Per detectors:'+statstr(bb[cut,0]))
legend()
subplot(3,1,2)
hist(aa[cut,1],bins=100,range=[-rr,rr],color='blue',alpha=0.2,label='All detectors:'+statstr(aa[cut,1]))
hist(bb[covmin > 40,1],bins=100,range=[-rr,rr],color='red',alpha=0.2,label='Per detectors:'+statstr(bb[cut,1]))
legend()
subplot(3,1,3)
hist(aa[cut,2],bins=100,range=[-rr,rr],color='blue',alpha=0.2,label='All detectors:'+statstr(aa[cut,2]))
hist(bb[cut,2],bins=100,range=[-rr,rr],color='red',alpha=0.2,label='Per detectors:'+statstr(bb[cut,2]))
legend()

QUall=[output_maps_all[:,1],output_maps_all[:,2]]
QUdet=[output_maps_det[:,1],output_maps_det[:,2]]
QUall_spoiled=[output_maps_all_spoiled[:,1],output_maps_all_spoiled[:,2]]
QUdet_spoiled=[output_maps_det_spoiled[:,1],output_maps_det_spoiled[:,2]]

### Get Rho and epsilon
rho_all,eps_all=mm.rhoepsilon_from_maps(QUall,QUall_spoiled,goodpix=covmin>40)
rho_det,eps_det=mm.rhoepsilon_from_maps(QUdet,QUdet_spoiled,goodpix=covmin>40)


################### Make a MC
dtheta=15.
npointings=5000
calerror=1e-2
rho_all=[]
eps_all=[]
rho_det=[]
eps_det=[]
nbmc=100
for n in np.arange(nbmc):
    print(' ')
    print(' ')
    print(' ')
    print('#############################')
    print('####### mc '+str(n))
    print('#############################')
    mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
    rnd_pointing=create_random_pointings([racenter, deccenter],npointings,dtheta)
    tod=mm.map2tod(maps,rnd_pointing,qubic)
    tod_spoiled=tod.copy()
    rndcal=np.random.normal(loc=1.,scale=calerror,size=(len(detectors),2))
    for i in np.arange(len(detectors)):
        for j in np.arange(2):
            tod_spoiled[i,:,j]=rndcal[i,j]*tod[i,:,j]

    output_maps_all,call=mm.tod2map(tod,rnd_pointing,qubic,disp=False)
    output_maps_all_spoiled,call_spoiled=mm.tod2map(tod_spoiled,rnd_pointing,qubic,disp=False)
    output_maps_det,cdet=mm.tod2map_perdet(tod,rnd_pointing,qubic,disp=False)
    output_maps_det_spoiled,cdet_spoiled=mm.tod2map_perdet(tod_spoiled,rnd_pointing,qubic,disp=False)
    covmin=np.arange(12*nside**2)
    for i in np.arange(12*nside**2):
        covmin[i]=np.min([call[i],call_spoiled[i],cdet[i],cdet_spoiled[i]])

    QUall=[output_maps_all[:,1],output_maps_all[:,2]]
    QUdet=[output_maps_det[:,1],output_maps_det[:,2]]
    QUall_spoiled=[output_maps_all_spoiled[:,1],output_maps_all_spoiled[:,2]]
    QUdet_spoiled=[output_maps_det_spoiled[:,1],output_maps_det_spoiled[:,2]]
    ### Get Rho and epsilon
    therho_all,theeps_all=mm.rhoepsilon_from_maps(QUall,QUall_spoiled,goodpix=covmin>40)
    therho_det,theeps_det=mm.rhoepsilon_from_maps(QUdet,QUdet_spoiled,goodpix=covmin>40)
    rho_all.append(therho_all)
    eps_all.append(theeps_all)
    rho_det.append(therho_det)
    eps_det.append(theeps_det)
    print(therho_all,theeps_all)
    print(therho_det,theeps_det)



clf()
subplot(2,1,1)
m=0.015
xlim(-m,m)
title('Nb={0:.2f} - Cal. Error = {1:.5f}'.format(len(rho_all),calerror))
hist(rho_all,bins=20,range=[-m,m],label='rho All detectors: '+statstr(rho_all),color='blue',alpha=0.2)
hist(rho_det,bins=20,range=[-m,m],label='rho Per detectors: '+statstr(rho_det),color='red',alpha=0.2)
xlabel('rho')
legend()
subplot(2,1,2)
m=0.015
xlim(-m,m)
hist(eps_all,bins=20,range=[-m,m],label='eps All detectors: '+statstr(eps_all),color='blue',alpha=0.2)
hist(eps_det,bins=20,range=[-m,m],label='eps Per detectors: '+statstr(eps_det),color='red',alpha=0.2)
xlabel('epsilon')
legend()
savefig('rho_eps_calerror0.01.png')




######################################### Now more General: Full 3x3 matrix ##########################
#### see second part of leakage_QU_pointing.nb
######################################################################################################

################## Input Maps #############################################
nside=128
lmax=3*nside
ell=spectra[0]
maskl=ell<(lmax+1)
bl=np.ones(maskl.size)
fwhmrad=0.5*np.pi/180
mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
hp.mollview(mapi)
hp.mollview(mapq)
hp.mollview(mapu)
maps=np.transpose(np.array([mapi,mapq,mapu]))


################# Qubic Instrument #########################################
qubic = QubicInstrument('monochromatic',nside=nside)
detectors=qubic.detector.packed
clf()
plot(detectors.center[:,0],detectors.center[:,1],'ro')


################# random pointings ##########################################
racenter = 0.0
deccenter = -57.0
dtheta=15.
npointings=1000
rnd_pointing=create_random_pointings([racenter, deccenter],npointings,dtheta)
op = (DegreesOperator() * Cartesian2SphericalOperator('azimuth,elevation') *
      CartesianEquatorial2GalacticOperator() *
      Spherical2CartesianOperator('azimuth,elevation') *
      RadiansOperator())
center = op([racenter, deccenter])

################# TOD
tod=mm.map2tod(maps,rnd_pointing,qubic,kmax=2)

################# Spoil bolometers calibration
calerror=1e-2
tod_spoiled=tod.copy()
rndcal=np.random.normal(loc=1.,scale=calerror,size=(len(detectors),2))
for i in np.arange(len(detectors)):
    for j in np.arange(2):
        tod_spoiled[i,:,j]=rndcal[i,j]*tod[i,:,j]

################# Maps
output_maps_all,call=mm.tod2map(tod,rnd_pointing,qubic,disp=False,displaytime=True)
output_maps_all_spoiled,call_spoiled=mm.tod2map(tod_spoiled,rnd_pointing,qubic,disp=False,displaytime=True)
output_maps_det,cdet=mm.tod2map_perdet(tod,rnd_pointing,qubic,disp=False,displaytime=True)
output_maps_det_spoiled,cdet_spoiled=mm.tod2map_perdet(tod_spoiled,rnd_pointing,qubic,disp=False,displaytime=True)

covmin=np.arange(12*nside**2)
for i in np.arange(12*nside**2):
    covmin[i]=np.min([call[i],call_spoiled[i]])#,cdet[i],cdet_spoiled[i]])

mm.display(output_maps_all,'',center=center,lim=[200,3,3],reso=15,mask=covmin < 40)
mm.display(output_maps_all_spoiled,'',center=center,lim=[200,3,3],reso=15,mask=covmin < 40)
mm.display(output_maps_det,'',center=center,lim=[200,3,3],reso=15,mask=covmin < 40)
mm.display(output_maps_det_spoiled,'',center=center,lim=[200,3,3],reso=15,mask=covmin < 40)

rr=0.2
figure()
mm.display(output_maps_all_spoiled-output_maps_all,'',center=center,lim=[rr,rr,rr],reso=15,mask=covmin < 40)
figure()
mm.display(output_maps_det_spoiled-output_maps_det,'',center=center,lim=[rr,rr,rr],reso=15,mask=covmin < 40)

clf()
rr=0.2
cut=covmin > 40
aa=output_maps_all_spoiled-output_maps_all
bb=output_maps_det_spoiled-output_maps_det
subplot(3,1,1)
hist(aa[cut,0],bins=100,range=[-rr,rr],color='blue',alpha=0.2,label='All detectors:'+statstr(aa[cut,0]))
hist(bb[cut,0],bins=100,range=[-rr,rr],color='red',alpha=0.2,label='Per detectors:'+statstr(bb[cut,0]))
legend()
title('Residuals I')
subplot(3,1,2)
hist(aa[cut,1],bins=100,range=[-rr,rr],color='blue',alpha=0.2,label='All detectors:'+statstr(aa[cut,1]))
hist(bb[covmin > 40,1],bins=100,range=[-rr,rr],color='red',alpha=0.2,label='Per detectors:'+statstr(bb[cut,1]))
legend()
title('Residuals Q')
subplot(3,1,3)
hist(aa[cut,2],bins=100,range=[-rr,rr],color='blue',alpha=0.2,label='All detectors:'+statstr(aa[cut,2]))
hist(bb[cut,2],bins=100,range=[-rr,rr],color='red',alpha=0.2,label='Per detectors:'+statstr(bb[cut,2]))
legend()
title('Residuals U')


### Get Rho and epsilon
pars_all=mm.mixingmatrix_from_maps(output_maps_all,output_maps_all_spoiled,goodpix=covmin>40)
pars_det=mm.mixingmatrix_from_maps(output_maps_det,output_maps_det_spoiled,goodpix=covmin>40)

################### Make a MC
dtheta=15.
npointings=10000
calerror=1e-2
rhoT_all=[]
rhoQ_all=[]
rhoU_all=[]
epsTQ_all=[]
epsTU_all=[]
epsQU_all=[]
rhoT_det=[]
rhoQ_det=[]
rhoU_det=[]
epsTQ_det=[]
epsTU_det=[]
epsQU_det=[]
kmax=0
nbmc=10
for n in np.arange(nbmc):
    print(' ')
    print(' ')
    print(' ')
    print('#############################')
    print('####### mc '+str(n))
    print('#############################')
    mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
    rnd_pointing=create_random_pointings([racenter, deccenter],npointings,dtheta)
    tod=mm.map2tod(maps,rnd_pointing,qubic,kmax=kmax)
    tod_spoiled=tod.copy()
    rndcal=np.random.normal(loc=1.,scale=calerror,size=(len(detectors),2))
    for i in np.arange(len(detectors)):
        for j in np.arange(2):
            tod_spoiled[i,:,j]=rndcal[i,j]*tod[i,:,j]

    output_maps_all,call=mm.tod2map(tod,rnd_pointing,qubic,disp=False,kmax=kmax)
    output_maps_all_spoiled,call_spoiled=mm.tod2map(tod_spoiled,rnd_pointing,qubic,disp=False,kmax=kmax)
    output_maps_det,cdet=mm.tod2map_perdet(tod,rnd_pointing,qubic,disp=False,kmax=kmax)
    output_maps_det_spoiled,cdet_spoiled=mm.tod2map_perdet(tod_spoiled,rnd_pointing,qubic,disp=False,kmax=kmax)
    covmin=np.arange(12*nside**2)
    for i in np.arange(12*nside**2):
        covmin[i]=np.min([call[i],call_spoiled[i],cdet[i],cdet_spoiled[i]])
    ### Get mixing parameters
    therhoTall,therhoQall,therhoUall,theepsTQall,theepsTUall,theepsQUall=mm.mixingmatrix_from_maps(output_maps_all,output_maps_all_spoiled,goodpix=covmin>40)
    rhoT_all.append(therhoTall)
    rhoQ_all.append(therhoQall)
    rhoU_all.append(therhoUall)
    epsTQ_all.append(theepsTQall)
    epsTU_all.append(theepsTUall)
    epsQU_all.append(theepsQUall)
    therhoTdet,therhoQdet,therhoUdet,theepsTQdet,theepsTUdet,theepsQUdet=mm.mixingmatrix_from_maps(output_maps_det,output_maps_det_spoiled,goodpix=covmin>40)
    rhoT_det.append(therhoTdet)
    rhoQ_det.append(therhoQdet)
    rhoU_det.append(therhoUdet)
    epsTQ_det.append(theepsTQdet)
    epsTU_det.append(theepsTUdet)
    epsQU_det.append(theepsQUdet)
    print('#######################################################')
    print(therhoTall,therhoQall,therhoUall,theepsTQall,theepsTUall,theepsQUall)
    print(therhoTdet,therhoQdet,therhoUdet,theepsTQdet,theepsTUdet,theepsQUdet)
    print('#######################################################')


def statstr(vec):
    m=np.mean(vec)
    s=np.std(vec)
    return '{0:.4f} +/- {1:.4f}'.format(m,s)

clf()
subplot(2,3,1)
m=0.001
xlim(-m,m)
hist(rhoT_all,bins=20,range=[-m,m],label='rhoT All detectors: '+statstr(rhoT_all),color='blue',alpha=0.2)
hist(rhoT_det,bins=20,range=[-m,m],label='rhoT Per detectors: '+statstr(rhoT_det),color='red',alpha=0.2)
xlabel('rhoT')
#legend()
subplot(2,3,2)
m=0.02
xlim(-m,m)
hist(rhoQ_all,bins=20,range=[-m,m],label='rhoQ All detectors: '+statstr(rhoQ_all),color='blue',alpha=0.2)
hist(rhoQ_det,bins=20,range=[-m,m],label='rhoQ Per detectors: '+statstr(rhoQ_det),color='red',alpha=0.2)
xlabel('rhoQ')
#legend()
subplot(2,3,3)
m=0.02
xlim(-m,m)
hist(rhoU_all,bins=20,range=[-m,m],label='rhoU All detectors: '+statstr(rhoU_all),color='blue',alpha=0.2)
hist(rhoU_det,bins=20,range=[-m,m],label='rhoU Per detectors: '+statstr(rhoU_det),color='red',alpha=0.2)
xlabel('rhoU')
#legend()
subplot(2,3,4)
m=0.0005
xlim(-m,m)
hist(epsTQ_all,bins=20,range=[-m,m],label='epsTQ All detectors: '+statstr(epsTQ_all),color='blue',alpha=0.2)
hist(epsTQ_det,bins=20,range=[-m,m],label='epsTQ Per detectors: '+statstr(epsTQ_det),color='red',alpha=0.2)
xlabel('epsTQ')
#legend()
subplot(2,3,5)
m=0.0005
xlim(-m,m)
hist(epsTU_all,bins=20,range=[-m,m],label='epsTU All detectors: '+statstr(epsTU_all),color='blue',alpha=0.2)
hist(epsTU_det,bins=20,range=[-m,m],label='epsTU Per detectors: '+statstr(epsTU_det),color='red',alpha=0.2)
xlabel('epsTU')
#legend()
subplot(2,3,6)
m=0.01
xlim(-m,m)
hist(epsQU_all,bins=20,range=[-m,m],label='epsQU All detectors: '+statstr(epsQU_all),color='blue',alpha=0.2)
hist(epsQU_det,bins=20,range=[-m,m],label='epsQU Per detectors: '+statstr(epsQU_det),color='red',alpha=0.2)
xlabel('epsQU')
#legend()
savefig('matrix9_Cal'+str(calerror)+'_nptg'+str(npointings)+'.png')


matrixAll=np.zeros((3,3))
matrixAll[0,0]=np.std(rhoT_all)
matrixAll[1,1]=np.std(rhoQ_all)
matrixAll[2,2]=np.std(rhoU_all)
matrixAll[0,1]=np.std(epsTQ_all)
matrixAll[0,2]=np.std(epsTU_all)
matrixAll[1,2]=np.std(epsQU_all)
matrixAll[1,0]=np.std(epsTQ_all)
matrixAll[2,0]=np.std(epsTU_all)
matrixAll[2,1]=np.std(epsQU_all)
matrixDet=np.zeros((3,3))
matrixDet[0,0]=np.std(rhoT_det)
matrixDet[1,1]=np.std(rhoQ_det)
matrixDet[2,2]=np.std(rhoU_det)
matrixDet[0,1]=np.std(epsTQ_det)
matrixDet[0,2]=np.std(epsTU_det)
matrixDet[1,2]=np.std(epsQU_det)
matrixDet[1,0]=np.std(epsTQ_det)
matrixDet[2,0]=np.std(epsTU_det)
matrixDet[2,1]=np.std(epsQU_det)

clf()
subplot(1,3,1)
imshow(matrixAll,interpolation='nearest',vmin=0,vmax=1e-2)
title('RMS All detectors')
colorbar()
subplot(1,3,2)
imshow(matrixDet,interpolation='nearest',vmin=0,vmax=1e-2)
title('RMS Per detector')
colorbar()
subplot(1,3,3)
imshow(matrixDet/matrixAll,interpolation='nearest',vmin=0,vmax=1)
title('RMS Ratio')
colorbar()
savefig('matrix9_lin_Cal'+str(calerror)+'_nptg'+str(npointings)+'.png')


clf()
subplot(1,3,1)
imshow(np.log10(matrixAll),interpolation='nearest',vmin=-5,vmax=-2)
title('RMS All detectors (Log)')
colorbar()
subplot(1,3,2)
imshow(np.log10(matrixDet),interpolation='nearest',vmin=-5,vmax=-2)
title('RMS Per detector (Log)')
colorbar()
subplot(1,3,3)
imshow(matrixDet/matrixAll,interpolation='nearest',vmin=0,vmax=1)
colorbar()
title('RMS Ratio')
savefig('matrix9_log_Cal'+str(calerror)+'_nptg'+str(npointings)+'.png')

