from __future__ import division

import matplotlib.pyplot as mp
import numexpr as ne
import numpy as np
from scipy.constants import c, pi

from pyoperators import MaskOperator
from pysimulators import create_fitsheader, DiscreteSurface
from qubic import QubicInstrument
from qubic.beams import GaussianBeam
from SelfCalib import selfcal
import mayavi.mlab 

qubic = QubicInstrument('monochromatic,nopol')
input_power=2e-11 #W reasonnable for QUBIC see optimisation QUBIC.xls
xdet=qubic.detector.center[:,:,0]
ydet=qubic.detector.center[:,:,1]



#first get an image with no saturation to have the number of bolometers
signal,noise=selfcal.get_fpimage(qubic,SOURCE_POWER=5e-13,SOURCE_THETA=np.radians(0),SOURCE_PHI=np.radians(45),npts=256,display=False,background=True,saturation=False)
notbols = (noise == 0)
bols=(noise != 0)
nfake=int(notbols.sum())
nbolstot=int(bols.sum())
clf()
imshow(np.log10(signal),interpolation='nearest')
clf()
imshow(signal*1e12,interpolation='nearest',vmin=0,vmax=25)
colorbar()

#### Demonstration de la methode
horns=[[10,10],[11,11]]
Sout,Nout,all=selfcal.get_fringe(qubic,horns,SOURCE_THETA=np.radians(2.),SOURCE_POWER=5e-14,npts=128,display=False)

clf()
subplot(2,3,1)
imshow(all[0]*1e12,interpolation='nearest',vmin=0,vmax=25)
colorbar(label='pW')
title('All open: $S$')
subplot(2,3,2)
imshow(all[1]*1e12,interpolation='nearest',vmin=0,vmax=25)
colorbar(label='pW')
title('i closed: $C_i$')
subplot(2,3,5)
imshow(all[2]*1e12,interpolation='nearest',vmin=0,vmax=25)
colorbar(label='pW')
title('j closed: $C_j$')
subplot(2,3,4)
imshow(all[3]*1e12,interpolation='nearest',vmin=0,vmax=25)
colorbar(label='pW')
title('i and j closed: $C_{ij}$')
subplot(2,3,3)
imshow(Sout*1e12,interpolation='nearest',vmin=-0.2,vmax=0.2)
title('$S-C_i-C_j+S_{ij}$')
colorbar(label='pW')
subplot(2,3,6)
imshow(Nout*1e15,interpolation='nearest',vmin=0,vmax=0.5)
title('Noise for $S-C_i-C_j+S_{ij}$')
colorbar(label='fW')

mayavi.mlab.barchart(xdet.ravel()/0.003,ydet.ravel()/0.003,Sout.ravel()/np.max(Sout)*10)
mayavi.mlab.title('Resulting Fringe')

mayavi.mlab.barchart(xdet.ravel()/0.003,ydet.ravel()/0.003,Nout.ravel()*5e16)
mayavi.mlab.title('Resulting Fringe Noise')

mayavi.mlab.barchart(xdet.ravel()/0.003,ydet.ravel()/0.003,all[0].ravel()*1e14)
mayavi.mlab.title('All Horns Open')


# Signal to noise
S_N=float(np.sum(np.abs(Sout))/np.sqrt(np.sum(Nout**2)))
nzero = int((Nout == 0).sum())
fracsat=(nzero-nfake)*1./(nbolstot)*100

angles=[0,1,2,5,10,15]
horns=[[10,10],[11,11]]
nn=30
powers=np.logspace(-12,-7,nn)
S_N=np.zeros((nn,len(angles)))
fracsat=np.zeros((nn,len(angles)))
for j in np.arange(len(angles)):
    for i in np.arange(nn):
        print(i)
        Sout,Nout,all=selfcal.get_fringe(qubic,horns,SOURCE_THETA=np.radians(angles[j]),SOURCE_POWER=powers[i],display=False,npts=128)
        S_N[i,j]=float(np.sum(np.abs(Sout))/np.sqrt(np.sum(Nout**2)))
        if float(np.sqrt(np.sum(Nout**2))) == 0: S_N[i,j]=0
        nzero = int((Nout == 0).sum())
        fracsat[i,j]=(nzero-nfake)*1./(nbolstot)*100

clf()
subplot(2,1,1)
for a in np.arange(len(angles)):
    plot(powers,S_N[:,a],label=str(angles[a])+' deg.')
xlabel('Power entering one horn [W]')
ylabel('S/N on fringe')
xscale('log')
legend(loc='upper left')
title('Assuming DeltaNu/Nu=0.25')

subplot(2,1,2)
for a in np.arange(len(angles)):
    plot(powers,fracsat[:,a])
xlabel('Power entering one horn [W]')
ylabel('Fraction saturated')
xscale('log')



################## Now doing several positions in theta and phi in order to sample more bolometers
signal,noise=selfcal.get_fpimage(qubic,SOURCE_POWER=5e-12,SOURCE_THETA=np.radians(0),SOURCE_PHI=np.radians(45),npts=256,display=False,background=True,saturation=False)
notbols = (noise == 0)
bols=(noise != 0)
nfake=int(notbols.sum())
nbolstot=int(bols.sum())
clf()
imshow(np.log10(signal),interpolation='nearest')
clf()
imshow(signal*1e12,interpolation='nearest',vmin=0,vmax=25)
colorbar()

nbpos=1000
thmin=0
thmax=20
thetavals=np.degrees(np.arccos(np.random.uniform(np.cos(np.radians(thmax)),np.cos(np.radians(thmin)),size=nbpos)))
phivals=np.random.uniform(0,360,size=nbpos)

horns=[[10,10],[11,11]]
nn=30
powers=np.logspace(-12,-5,nn)
S_N_pix=np.zeros((34,34,nn))
S_N_all=np.zeros(nn)
fracsat_all=np.zeros(nn)
for i in np.arange(nn):
    Stot=np.zeros((34,34))
    Ntot=np.zeros((34,34))
    for j in np.arange(nbpos):
        print(i,j)
        Sout,Nout,all=selfcal.get_fringe(qubic,horns,SOURCE_THETA=np.radians(thetavals[j]),SOURCE_PHI=np.radians(phivals[j]),SOURCE_POWER=powers[i],display=False,npts=128)
        Stot=Stot+np.abs(Sout)
        Ntot=np.sqrt(Ntot**2+Nout**2)
        clf()
        subplot(1,3,1)
        imshow(Stot,interpolation='nearest')
        colorbar()
        subplot(1,3,2)
        imshow(Ntot,interpolation='nearest')
        colorbar()
        subplot(1,3,3)
        imshow(np.nan_to_num(Stot/Ntot),interpolation='nearest')
        colorbar()
        draw()
    S_N_pix[:,:,i]=np.nan_to_num(Stot/Ntot)
    S_N_all[i]=np.nan_to_num(float(Stot.sum())/np.sqrt(float(np.sum(Ntot**2))))
    nzero = int((Ntot == 0).sum())
    fracsat_all[i]=(nzero-nfake)*1./(nbolstot)*100

clf()
imshow(S_N_pix[:,:,20],interpolation='nearest')
colorbar()

clf()
subplot(2,1,1)
plot(powers,S_N_all)
title('Averaging over 1000 directions')
xlabel('Power entering one horn [W]')
ylabel('S/N on fringe')
xscale('log')
subplot(2,1,2,)
plot(powers,fracsat_all)
ylabel('Fraction saturated')
xlabel('Power entering one horn [W]')
xscale('log')
savefig('sn_average_directions.png')

mayavi.mlab.barchart(xdet.ravel()/0.003,ydet.ravel()/0.003,S_N_pix[:,:,10].ravel()/5e2)

