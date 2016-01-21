source ~/software_pola_tmp/bin/activate

from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from pyoperators import (
    DenseOperator, DegreesOperator, DiagonalOperator, RadiansOperator,
    Cartesian2SphericalOperator, Spherical2CartesianOperator, pcg)
from pysimulators import (
    CartesianEquatorial2GalacticOperator, ProjectionOperator, FitsArray)
from pysimulators import (
    ProjectionOperator, SphericalEquatorial2GalacticOperator, FitsArray)
from qubic import QubicAcquisition, QubicInstrument, create_sweeping_pointings, create_random_pointings
from MapMaking import MapMaking as mm


################# Input Power spectrum ###################################
a=np.loadtxt('/Users/hamilton/Qubic/cl_r=0.1bis2.txt')
ell=a[:,0]
ctt=np.concatenate([[0,0],a[:,1]*1e12*2*np.pi/(ell*(ell+1))])
cee=np.concatenate([[0,0],a[:,2]*1e12*2*np.pi/(ell*(ell+1))])
cte=np.concatenate([[0,0],a[:,4]*1e12*2*np.pi/(ell*(ell+1))])
cbb=np.concatenate([[0,0],a[:,7]*1e12*2*np.pi/(ell*(ell+1))])
ell=np.concatenate([[0,1],ell])
spectra=[ell,ctt,cee,cbb,cte]   #### This is the new ordering

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
clf()
plot(detectors.center[:,0],detectors.center[:,1],'ro')


################## Create some pointings ###################################
#### Sweeping ####
#racenter = 0.0
#deccenter = -57.0
#angspeed = 0.1    # deg/sec
#delta_az = 25.
#angspeed_psi = 0
#maxpsi = 45.
#nsweeps_el = 300
#duration = 24   # hours
#ts = 10         # seconds
#pointing = create_sweeping_pointings(
#    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
#    angspeed_psi, maxpsi)
#pointing.angle_hwp = np.random.random_integers(0, 7, pointing.size) * 22.5
#ntimes = len(pointing)
##### Random pointing ######
racenter = 0.0
deccenter = -57.0
delta_az = 25.
duration = 1   # hours
ts = 10         # seconds
npointings = duration * 3600 / ts
pointing = create_random_pointings([racenter, deccenter], npointings, delta_az)

#### Conversion en RA DEC
import pysimulators
from astropy.time import Time, TimeDelta
DOMECLAT = -(75 + 6 / 60)
DOMECLON = 123 + 20 / 60
t0 = pointing.date_obs
dt = TimeDelta(pointing.time, format='sec')
op = (pysimulators.SphericalHorizontal2EquatorialOperator('NE', t0 + dt, DOMECLAT, DOMECLON, degrees=True))
radec = op(np.array([pointing.azimuth, pointing.elevation]).T)

clf()
subplot(2,1,1)
plot(((radec[:,0]+180 + 360) % 360) -180,radec[:,1],',')
subplot(2,1,2,projection='mollweide')
plot(np.radians(((radec[:,0]+180 + 360) % 360) -180),np.radians(radec[:,1]),',')

#### Comparaison des map-makings avec tous les det d'un coup ou bien det par det
tod_all=mm.map2tod(maps,pointing,qubic)

clf()
plot(tod_all[0,:,0])


## All det
output_maps_all,coverage_all=mm.tod2map(tod_all,pointing,qubic)
output_maps_all_new=output_maps_all.copy()
hp.mollview(output_maps_all[:,0])
hp.mollview(output_maps_all[:,1])
hp.mollview(output_maps_all[:,2])


## per det
output_maps_det,coverage_det=mm.tod2map_perdet(tod_all,pointing,qubic)
output_maps_det_new=output_maps_det.copy()
hp.mollview(output_maps_det[:,0])
hp.mollview(output_maps_det[:,1])
hp.mollview(output_maps_det[:,2])

mask = coverage_det < 10
output_maps_all_new[mask,:]=0
output_maps_det_new[mask,:]=0
newmapi=mapi.copy()
newmapq=mapq.copy()
newmapu=mapu.copy()
newmapi[mask]=np.nan
newmapq[mask]=np.nan
newmapu[mask]=np.nan

e2g = SphericalEquatorial2GalacticOperator(degrees=True)
center = e2g([racenter, deccenter])


clf()
rng=200
res=15
hp.gnomview(newmapi,rot=center,reso=res,min=-rng,max=rng,title='Input',sub=(2,3,1))
hp.gnomview(output_maps_det_new[:,0],rot=center,reso=res,min=-rng,max=rng,title='Opt Mean',sub=(2,3,2))
hp.gnomview(output_maps_all_new[:,0],rot=center,reso=res,min=-rng,max=rng,title='All',sub=(2,3,3))
hp.gnomview(output_maps_det_new[:,0]-newmapi,rot=center,reso=res,min=-rng,max=rng,title='Res Opt Mean',sub=(2,3,5))
hp.gnomview(output_maps_all_new[:,0]-newmapi,rot=center,reso=res,min=-rng,max=rng,title='Res All',sub=(2,3,6))



clf()
subplot(3,1,1)
rng=200
hist(output_maps_det_new[~mask,0]-mapi[~mask],bins=100,range=[-rng,rng],label='Opt Mean: {0:.2f}'.format(np.std(output_maps_det_new[~mask,0]-mapi[~mask])),alpha=0.5)
hist(output_maps_all_new[~mask,0]-mapi[~mask],bins=100,range=[-rng,rng],label='All: {0:.2f}'.format(np.std(output_maps_all_new[~mask,0]-mapi[~mask])),alpha=0.5)
legend()
title('I')

subplot(3,1,2)
rng=3
hist(output_maps_det_new[~mask,1]-mapq[~mask],bins=100,range=[-rng,rng],label='Opt Mean: {0:.2f}'.format(np.std(output_maps_det_new[~mask,1]-mapq[~mask])),alpha=0.5)
hist(output_maps_all_new[~mask,1]-mapq[~mask],bins=100,range=[-rng,rng],label='All: {0:.2f}'.format(np.std(output_maps_all_new[~mask,1]-mapq[~mask])),alpha=0.5)
legend()
title('Q')

subplot(3,1,3)
hist(output_maps_det_new[~mask,2]-mapu[~mask],bins=100,range=[-rng,rng],label='Opt Mean: {0:.2f}'.format(np.std(output_maps_det_new[~mask,2]-mapu[~mask])),alpha=0.5)
hist(output_maps_all_new[~mask,2]-mapu[~mask],bins=100,range=[-rng,rng],label='All: {0:.2f}'.format(np.std(output_maps_all_new[~mask,2]-mapu[~mask])),alpha=0.5)
legend()
title('U')


##### A faire:
# - pas les deux plans focaux
# - nside pas dans instrument
# - mettre la convolution dans map2tod
# - comprendre effet all detectors / single detector Vs kmax => redondance ?
# - je ne comprends pas ou est l'info sur le bruit... pour le 1/f...



############# now MC to get Cl ratio
nbmc=100
kmax=[0,1,2]
dtheta=15.
npointings=10000
allcl_det=np.zeros((3*nside,nbmc,len(kmax)))
allcl_all=np.zeros((3*nside,nbmc,len(kmax)))
allcl_true=np.zeros((3*nside,nbmc,len(kmax)))
for m in np.arange(67,nbmc):
    for k in np.arange(len(kmax)):
        print('')
        print('')
        print('=========================')
        print('k='+str(k)+'  m='+str(m))
        print('=========================')
        ## create initial maps
        mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=np.radians(38.93/60),pixwin=True,new=True)
        maps=np.transpose(np.array([mapi,mapq,mapu]))
        ## Random pointings
        rnd_pointing=create_random_pointings([racenter, deccenter],npointings,dtheta)
        ## Comparaison des map-makings avec tous les det d'un coup ou bien det par det
        tod_all=mm.map2tod(maps,rnd_pointing,qubic,kmax=kmax[k])
        ## All det
        output_maps_all,coverage_all=mm.tod2map(tod_all,rnd_pointing,qubic,disp=False,kmax=kmax[k])
        output_maps_all_new=output_maps_all.copy()
        ## per det
        output_maps_det,coverage_det=mm.tod2map_perdet(tod_all,rnd_pointing,qubic,disp=False,kmax=kmax[k])
        output_maps_det_new=output_maps_det.copy()
        mask = coverage_det < 500
        output_maps_all_new[mask,:]=0
        output_maps_det_new[mask,:]=0
        newmapconvi=maps[:,0].copy()
        newmapconvi[mask]=0
        cl=hp.anafast(newmapconvi)
        newmapi=maps[:,0].copy()
        newmapi[mask]=0
        cl_all=hp.anafast(output_maps_all_new[:,0])
        cl_det=hp.anafast(output_maps_det_new[:,0])
        allcl_det[:,m,k]=cl_det
        allcl_all[:,m,k]=cl_all
        allcl_true[:,m,k]=cl

clratio_det=allcl_det/allcl_true
clratio_all=allcl_all/allcl_true

ratio_det=np.zeros((3*nside,len(kmax)))
dratio_det=np.zeros((3*nside,len(kmax)))
ratio_all=np.zeros((3*nside,len(kmax)))
dratio_all=np.zeros((3*nside,len(kmax)))
mcldet=np.zeros((3*nside,len(kmax)))
scldet=np.zeros((3*nside,len(kmax)))
mclall=np.zeros((3*nside,len(kmax)))
sclall=np.zeros((3*nside,len(kmax)))
mcltrue=np.zeros((3*nside,len(kmax)))
scltrue=np.zeros((3*nside,len(kmax)))
for k in np.arange(len(kmax)):
    for i in np.arange(3*nside):
        ratio_det[i,k]=np.mean(clratio_det[i,:,k])
        dratio_det[i,k]=np.std(clratio_det[i,:,k])
        ratio_all[i,k]=np.mean(clratio_all[i,:,k])
        dratio_all[i,k]=np.std(clratio_all[i,:,k])
        mcldet[i,k]=np.mean(allcl_det[i,:,k])
        scldet[i,k]=np.std(allcl_det[i,:,k])
        mclall[i,k]=np.mean(allcl_all[i,:,k])
        sclall[i,k]=np.std(allcl_all[i,:,k])
        mcltrue[i,k]=np.mean(allcl_true[i,:,k])
        scltrue[i,k]=np.std(allcl_true[i,:,k])
        


clf()
ll=np.arange(3*nside)
col=['blue','green','red']
ratio=np.zeros(len(kmax))
for k in np.arange(len(kmax)):
    mask = ll > 50
    bla=np.mean(ratio_det[mask,k])
    ratio[k]=bla
    blo=np.std(ratio_det[mask,k])
    plot(ll,ratio_det[:,k],label='$C_\ell^{rec} / C_\ell^{true}$ Per detector - $k_{max} = $'+str(k)+' - Mean = {0:.2f} +/- {1:.2f}'.format(bla,blo),color=col[k],lw=3)
    plot(ll,ll*0+bla,'--',color=col[k],lw=2)
ylim(0,2)
legend(loc='upper right',frameon=False)
xlabel('Multipole')
ylabel('$C_\ell^{rec} / C_\ell^{true}$')
title('T Only - Average over '+str(nbmc)+' realizations - '+str(npointings)+' pointings over '+str(dtheta)+'deg radius')
savefig('clratio_nbmc'+str(nbmc)+'_ptg'+str(npointings)+'_dth'+str(dtheta)+'.png')


clf()
ll=np.arange(3*nside)
col=['blue','green','red']
for k in np.arange(len(kmax)):
    mask = ll > 50
    bla=np.mean(ratio_all[mask,k])
    blo=np.std(ratio_all[mask,k])
    plot(ll,ratio_all[:,k],label='$C_\ell^{rec} / C_\ell^{true}$ All detectors - $k_{max} = $'+str(k)+' - Mean = {0:.2f} +/- {1:.2f}'.format(bla,blo),color=col[k],lw=3)
    plot(ll,ll*0+bla,'--',color=col[k],lw=2)
ylim(0.95,1.05)
legend(loc='upper right',frameon=False)
xlabel('Multipole')
ylabel('$C_\ell^{rec} / C_\ell^{true}$')
title('T Only - Average over '+str(nbmc)+' realizations - '+str(npointings)+' pointings over '+str(dtheta)+'deg radius')
savefig('clratio_alldet_nbmc'+str(nbmc)+'_ptg'+str(npointings)+'_dth'+str(dtheta)+'.png')


clf()
k=2
#plot(ell,spectra[1]*(ell*(ell+1)/(2*np.pi)),label='$C_\ell^{TT}$')
plot(ll,mcltrue[:,k]*ll*(ll+1)/(2*np.pi),'k',label='Initial Map')
plot(ll,(mcltrue[:,k]+scltrue[:,k])*ll*(ll+1)/(2*np.pi),'k:')
plot(ll,(mcltrue[:,k]-scltrue[:,k])*ll*(ll+1)/(2*np.pi),'k:')
plot(ll,mclall[:,k]*ll*(ll+1)/(2*np.pi),'b',label='Reconstructed Map (All detectors)')
plot(ll,(mclall[:,k]+scltrue[:,k])*ll*(ll+1)/(2*np.pi),'b:')
plot(ll,(mclall[:,k]-scltrue[:,k])*ll*(ll+1)/(2*np.pi),'b:')
plot(ll,mcldet[:,k]*ll*(ll+1)/(2*np.pi)/ratio[k],'r',label='Reconstructed Map (per detectors)')
plot(ll,(mcldet[:,k]+scltrue[:,k])*ll*(ll+1)/(2*np.pi)/ratio[k],'r:')
plot(ll,(mcldet[:,k]-scltrue[:,k])*ll*(ll+1)/(2*np.pi)/ratio[k],'r:')
xlabel('Multipole')
ylabel('$C_\ell^{rec}$ (corrected for bias)')
legend()
savefig('clrec_nbmc'+str(nbmc)+'_ptg'+str(npointings)+'_dth'+str(dtheta)+'.png')

clf()
plot(ll,scltrue[:,k]*ll*(ll+1)/(2*np.pi),'k',label='Initial Map')
plot(ll,sclall[:,k]*ll*(ll+1)/(2*np.pi),'b',label='Reconstructed Map (All detectors)')
plot(ll,scldet[:,k]*ll*(ll+1)/(2*np.pi)/ratio[k],'r',label='Reconstructed Map (per detectors)')
xlabel('Multipole')
ylabel('$\Delta C_\ell^{rec}$ (corrected for bias)')
legend()
ylim(0,20)
savefig('deltaclrec_nbmc'+str(nbmc)+'_ptg'+str(npointings)+'_dth'+str(dtheta)+'.png')



op = (DegreesOperator() * Cartesian2SphericalOperator('azimuth,elevation') *
      CartesianEquatorial2GalacticOperator() *
      Spherical2CartesianOperator('azimuth,elevation') *
      RadiansOperator())

center = op([racenter, deccenter])

clf()
hp.gnomview(newmapi,rot=center,reso=15,min=-250,max=250,sub=(2,4,1),title='Input')
hp.gnomview(output_maps_all_new[:,0],rot=center,reso=15,min=-250,max=250,sub=(2,4,2),title='All Det')
hp.gnomview(output_maps_det_new[:,0],rot=center,reso=15,min=-250,max=250,sub=(2,4,3),title='Per Det')
hp.gnomview(output_maps_det_new[:,0]/np.sqrt(ratio[2]),rot=center,reso=15,min=-250,max=250,sub=(2,4,4),title='Per Det debiased')
hp.gnomview(newmapi-newmapi,rot=center,reso=15,min=-30,max=30,sub=(2,4,5),title='Residuals')
hp.gnomview(output_maps_all_new[:,0]-newmapi,rot=center,reso=15,min=-30,max=30,sub=(2,4,6),title='Residuals')
hp.gnomview(output_maps_det_new[:,0]-newmapi,rot=center,reso=15,min=-30,max=30,sub=(2,4,7),title='Residuals')
hp.gnomview(output_maps_det_new[:,0]/np.sqrt(ratio[2])-newmapi,rot=center,reso=15,min=-30,max=30,sub=(2,4,8),title='Residuals')
savefig('maps_nbmc'+str(nbmc)+'_ptg'+str(npointings)+'_dth'+str(dtheta)+'.png')

def statstr(vec):
    m=np.mean(vec)
    s=np.std(vec)
    return '{0:.2f} +/- {1:.2f}'.format(m,s)

badpix= newmapi==0
clf()
subplot(2,3,1)
plot(newmapi[~badpix],output_maps_all_new[~badpix,0],'k,')
plot(linspace(-300,300,100),linspace(-300,300,100),'r--')
ylabel('All Det')
xlabel('Input')

subplot(2,3,2)
plot(newmapi[~badpix],output_maps_det_new[~badpix,0],'k,')
plot(linspace(-300,300,100),linspace(-300,300,100),'r--')
ylabel('Per Det')
xlabel('Input')

subplot(2,3,3)
plot(newmapi[~badpix],output_maps_det_new[~badpix,0]/np.sqrt(ratio[2]),'k,')
plot(linspace(-300,300,100),linspace(-300,300,100),'r--')
ylabel('Per Det Debiased')
xlabel('Input')

subplot(2,1,2)
aa=output_maps_all_new[~badpix,0]-newmapi[~badpix]
bb=output_maps_det_new[~badpix,0]-newmapi[~badpix]
cc=output_maps_det_new[~badpix,0]/np.sqrt(ratio[2])-newmapi[~badpix]
hist(aa,bins=100,range=[-50,50],label='All det: '+statstr(aa),alpha=0.3)
hist(bb,bins=100,range=[-50,50],label='Per det: '+statstr(bb),alpha=0.3)
hist(cc,bins=100,range=[-50,50],label='Per det debiased: '+statstr(cc),alpha=0.3)
legend()
savefig('histopix_nbmc'+str(nbmc)+'_ptg'+str(npointings)+'_dth'+str(dtheta)+'.png')




##### Same but with only k=2 and varying redundancy in order to check if the bias on the Cl is just related to redundancy (basically the explanation would be that with all pixel simultaneously, one has 1000 times more redundancy than with a single pixel).
nbmc=50
allnpointings=[5000, 20000]
dtheta=15.
kmax=2
allcl_det=np.zeros((3*nside,nbmc,len(allnpointings)))
allcl_true=np.zeros((3*nside,nbmc,len(allnpointings)))
for m in np.arange(nbmc):
    for k in np.arange(len(allnpointings)):
        print('')
        print('')
        print('=========================')
        print('k='+str(k)+'  m='+str(m))
        print('=========================')
        ## create initial maps
        mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=np.radians(38.93/60),pixwin=True,new=True)
        maps=np.transpose(np.array([mapi,mapq,mapu]))
        ## Random pointings
        rnd_pointing=create_random_pointings([racenter, deccenter],allnpointings[k],dtheta)
        tod_all=mm.map2tod(maps,rnd_pointing,qubic,kmax=2)
        ## per det
        output_maps_det,coverage_det=mm.tod2map_perdet(tod_all,rnd_pointing,qubic,disp=False,kmax=2)
        output_maps_det_new=output_maps_det.copy()
        mask = coverage_det < 500
        output_maps_det_new[mask,:]=0
        newmapconvi=maps[:,0].copy()
        newmapconvi[mask]=0
        cl=hp.anafast(newmapconvi)
        newmapi=maps[:,0].copy()
        newmapi[mask]=0
        cl_det=hp.anafast(output_maps_det_new[:,0])
        allcl_det[:,m,k]=cl_det
        allcl_true[:,m,k]=cl

clratio_det=allcl_det/allcl_true

ratio_det=np.zeros((3*nside,len(allnpointings)))
dratio_det=np.zeros((3*nside,len(allnpointings)))
mcldet=np.zeros((3*nside,len(allnpointings)))
scldet=np.zeros((3*nside,len(allnpointings)))
mcltrue=np.zeros((3*nside,len(allnpointings)))
scltrue=np.zeros((3*nside,len(allnpointings)))
for k in np.arange(len(allnpointings)):
    for i in np.arange(3*nside):
        ratio_det[i,k]=np.mean(clratio_det[i,:,k])
        dratio_det[i,k]=np.std(clratio_det[i,:,k])
        mcldet[i,k]=np.mean(allcl_det[i,:,k])
        scldet[i,k]=np.std(allcl_det[i,:,k])
        mcltrue[i,k]=np.mean(allcl_true[i,:,k])
        scltrue[i,k]=np.std(allcl_true[i,:,k])
        

clf()
ll=np.arange(3*nside)
col=['green','red']
ratio=np.zeros(len(allnpointings))
for k in np.arange(len(allnpointings)):
    mask = ll > 50
    bla=np.mean(ratio_det[mask,k])
    ratio[k]=bla
    blo=np.std(ratio_det[mask,k])
    plot(ll,ratio_det[:,k],label='$C_\ell^{rec} / C_\ell^{true}$ Per detector - $N_{ptg} = $'+str(allnpointings[k])+' - Mean = {0:.2f} +/- {1:.2f}'.format(bla,blo),color=col[k],lw=3)
    plot(ll,ll*0+bla,'--',color=col[k],lw=2)
ylim(0,2)
plot(ll,ll*0+1,'b--',lw=2)
legend(loc='upper right',frameon=False)
xlabel('Multipole')
ylabel('$C_\ell^{rec} / C_\ell^{true}$')
title('T Only - Average over '+str(nbmc)+' realizations - '+str(npointings)+' pointings over '+str(dtheta)+'deg radius')
savefig('clratioVSptg_nbmc'+str(nbmc)+'_kmax2_dth'+str(dtheta)+'.png')

clf()
for k in np.arange(len(allnpointings)):
    subplot(2,1,k+1)
    plot(ll,mcltrue[:,k]*ll*(ll+1)/(2*np.pi),'k',label='Initial Map')
    plot(ll,(mcltrue[:,k]+scltrue[:,k])*ll*(ll+1)/(2*np.pi),'k:')
    plot(ll,(mcltrue[:,k]-scltrue[:,k])*ll*(ll+1)/(2*np.pi),'k:')
    plot(ll,mcldet[:,k]*ll*(ll+1)/(2*np.pi)/ratio[k],color=col[k],label='Rec. Map (per det.) - $N_{ptg} = $'+str(allnpointings[k]))
    plot(ll,(mcldet[:,k]+scltrue[:,k])*ll*(ll+1)/(2*np.pi)/ratio[k],':',color=col[k])
    plot(ll,(mcldet[:,k]-scltrue[:,k])*ll*(ll+1)/(2*np.pi)/ratio[k],':',color=col[k])
    xlabel('Multipole')
    ylabel('$C_\ell^{rec}$ (corrected for bias)')
    legend()
savefig('clrecVSptg_nbmc'+str(nbmc)+'_kmax2_dth'+str(dtheta)+'.png')

clf()
for k in np.arange(len(allnpointings)):
    aa=scldet[:,k]/ratio[k]/scltrue[:,k]
    lok=(ll > 50) & (ll < 250)
    plot(ll,ll*0+1,'k:')
    plot([50,50],[0,2],'k:')
    plot([50,50],[0,2],'k:')
    plot(ll,ll*0+np.mean(aa[lok]),':',color=col[k])
    plot(ll,aa,color=col[k],label=('Rec. Map (per det.) - $N_p = {0}$: '+statstr(aa[lok])).format(allnpointings[k]))
xlabel('Multipole')
ylabel('$\Delta C_\ell^{rec} / \Delta C_\ell^{true}$ (corrected for bias)')
legend(loc='lower right')
ylim(0,2)
savefig('deltaclrecVSptg_nbmc'+str(nbmc)+'_kmax2_dth'+str(dtheta)+'.png')




######### It would be interesting to check that this is really an issue related to redundancy by redoing the same but with a low number of pointings and all pixels. In order to check that a similar behaviour is observed.

nbmc=2000
allnpointings=[10,100]
dtheta=15.
kmax=2
allcl=np.zeros((3*nside,nbmc,len(allnpointings)))
allcl_true=np.zeros((3*nside,nbmc,len(allnpointings)))
for m in np.arange(nbmc):
    for k in np.arange(len(allnpointings)):
        print('')
        print('')
        print('=========================')
        print('k='+str(k)+'  m='+str(m))
        print('=========================')
        ## create initial maps
        mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=np.radians(38.93/60),pixwin=True,new=True)
        maps=np.transpose(np.array([mapi,mapq,mapu]))
        ## Random pointings
        rnd_pointing=create_random_pointings([racenter, deccenter],allnpointings[k],dtheta)
        tod_all=mm.map2tod(maps,rnd_pointing,qubic,kmax=2)
        output_maps,coverage=mm.tod2map(tod_all,rnd_pointing,qubic,disp=False,kmax=2)
        output_maps_new=output_maps.copy()
        mask = coverage < 1
        output_maps_new[mask,:]=0
        newmapconvi=maps[:,0].copy()
        newmapconvi[mask]=0
        cl=hp.anafast(newmapconvi)
        newmapi=maps[:,0].copy()
        newmapi[mask]=0
        thecl=hp.anafast(output_maps_new[:,0])
        allcl[:,m,k]=thecl
        allcl_true[:,m,k]=cl

clratio=allcl/allcl_true

ratio=np.zeros((3*nside,len(allnpointings)))
dratio=np.zeros((3*nside,len(allnpointings)))
mcl=np.zeros((3*nside,len(allnpointings)))
scl=np.zeros((3*nside,len(allnpointings)))
mcltrue=np.zeros((3*nside,len(allnpointings)))
scltrue=np.zeros((3*nside,len(allnpointings)))
for k in np.arange(len(allnpointings)):
    for i in np.arange(3*nside):
        ratio[i,k]=np.mean(clratio[i,:,k])
        dratio[i,k]=np.std(clratio[i,:,k])
        mcl[i,k]=np.mean(allcl[i,:,k])
        scl[i,k]=np.std(allcl[i,:,k])
        mcltrue[i,k]=np.mean(allcl_true[i,:,k])
        scltrue[i,k]=np.std(allcl_true[i,:,k])
        

clf()
ll=np.arange(3*nside)
col=['blue','green','red']
theratio=np.zeros(len(allnpointings))
for k in np.arange(len(allnpointings)):
    mask =(ll > 50) & (ll<250)
    bla=np.mean(ratio[mask,k])
    theratio[k]=bla
    blo=np.std(ratio[mask,k])
    plot(ll,ratio[:,k],label='$C_\ell^{rec} / C_\ell^{true}$ All pixels - $N_{ptg} = $'+str(allnpointings[k])+' - Mean = {0:.2f} +/- {1:.2f}'.format(bla,blo),color=col[k],lw=3)
    plot(ll,ll*0+bla,'--',color=col[k],lw=2)
ylim(0,2)
plot(ll,ll*0+1,'k--',lw=2)
legend(loc='upper right',frameon=False)
xlabel('Multipole')
ylabel('$C_\ell^{rec} / C_\ell^{true}$')
title('T Only - Average over '+str(nbmc)+' realizations - pointings over '+str(dtheta)+'deg radius')
savefig('clratioVSptg_ALLPIX_nbmc'+str(nbmc)+'_kmax2_dth'+str(dtheta)+'.png')

























############### Test with 10 Hz - 24h individual bolos ################
#### Sweeping ####
racenter = 0.0
deccenter = -57.0
angspeed = 0.1    # deg/sec
delta_az = 25.
angspeed_psi = 0
maxpsi = 45.
nsweeps_el = 300
duration = 24   # hours
ts = 0.1         # seconds
pointing = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)
pointing.angle_hwp = np.random.random_integers(0, 7, pointing.size) * 22.5
ntimes = len(pointing)

## Loop on detectors
maprec_i=np.zeros(12*nside**2)
maprec_q=np.zeros(12*nside**2)
maprec_u=np.zeros(12*nside**2)
covrec=np.zeros(12*nside**2)

for i in np.arange(len(detectors)):
    print(i)
    detok=[i]
    thetod=mm.map2tod(maps,pointing,qubic,detector_list=detok)
    output_maps_i,coverage_i=mm.tod2map(thetod,pointing,qubic,detector_list=detok,disp=False)
    covrec += np.nan_to_num(coverage_i)
    maprec_i += np.nan_to_num(output_maps_i[:,0]*coverage_i)
    maprec_q += np.nan_to_num(output_maps_i[:,1]*coverage_i)
    maprec_u += np.nan_to_num(output_maps_i[:,2]*coverage_i)

mask = covrec < 20000
maprec_i_new=maprec_i/covrec
maprec_q_new=maprec_q/covrec
maprec_u_new=maprec_u/covrec
maprec_i_new[mask]=np.nan
maprec_q_new[mask]=np.nan
maprec_u_new[mask]=np.nan
newmapi=mapi.copy()
newmapq=mapq.copy()
newmapu=mapu.copy()
newmapi[mask]=np.nan
newmapq[mask]=np.nan
newmapu[mask]=np.nan



clf()
subplot(3,1,1)
rng=200
hist(maprec_i_new-newmapi,bins=100,range=[-rng,rng],label='Opt Mean: {0:.2f}'.format(np.std(maprec_i_new[~mask]-newmapi[~mask])),alpha=0.5)
legend()
title('I')

subplot(3,1,2)
rng=3
hist(maprec_q_new-newmapq,bins=100,range=[-rng,rng],label='Opt Mean: {0:.2f}'.format(np.std(maprec_q_new[~mask]-newmapq[~mask])),alpha=0.5)
legend()
title('Q')

subplot(3,1,3)
hist(maprec_u_new-newmapu,bins=100,range=[-rng,rng],label='Opt Mean: {0:.2f}'.format(np.std(maprec_u_new[~mask]-newmapu[~mask])),alpha=0.5)
legend()
title('U')

e2g = SphericalEquatorial2GalacticOperator(degrees=True)
center = e2g([racenter, deccenter])
res=25.

clf()
rng=200
hp.gnomview(newmapi,rot=center,reso=res,min=-rng,max=rng,title='Input',sub=(2,3,1))
hp.gnomview(maprec_i_new,rot=center,reso=res,min=-rng,max=rng,title='Opt Mean',sub=(2,3,2))
hp.gnomview(newmapi-maprec_i_new,rot=center,reso=res,min=-rng,max=rng,title='Res Opt Mean',sub=(2,3,3))
subplot(2,1,2)
hist(newmapi-maprec_i_new,bins=100,range=[-rng,rng],label='Opt Mean: {0:.2f}'.format(np.std(maprec_i_new[~mask]-newmapi[~mask])),alpha=0.5)
legend()


clf()
rng=3
hp.gnomview(newmapq,rot=center,reso=res,min=-rng,max=rng,title='Input',sub=(2,3,1))
hp.gnomview(maprec_q_new,rot=center,reso=res,min=-rng,max=rng,title='Opt Mean',sub=(2,3,2))
hp.gnomview(newmapq-maprec_q_new,rot=center,reso=res,min=-rng,max=rng,title='Res Opt Mean',sub=(2,3,3))
subplot(2,1,2)
hist(newmapq-maprec_q_new,bins=100,range=[-rng,rng],label='Opt Mean: {0:.2f}'.format(np.std(maprec_q_new[~mask]-newmapq[~mask])),alpha=0.5)
legend()

clf()
rng=3
hp.gnomview(newmapu,rot=center,reso=res,min=-rng,max=rng,title='Input',sub=(2,3,1))
hp.gnomview(maprec_u_new,rot=center,reso=res,min=-rng,max=rng,title='Opt Mean',sub=(2,3,2))
hp.gnomview(newmapu-maprec_u_new,rot=center,reso=res,min=-rng,max=rng,title='Res Opt Mean',sub=(2,3,3))
subplot(2,1,2)
hist(newmapu-maprec_u_new,bins=100,range=[-rng,rng],label='Opt Mean: {0:.2f}'.format(np.std(maprec_u_new[~mask]-newmapu[~mask])),alpha=0.5)
legend()


