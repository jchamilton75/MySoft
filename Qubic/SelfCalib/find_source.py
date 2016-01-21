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


#minimum power to have entering one horn
min_pow = [3e-10, 1e-9, 3e-9, 1e-8]

#Horns
horndiam=0.0115
hornarea=np.pi * horndiam**2/4

#Geometrical quantities
hsource = 40.
dmast = 40.
distsource = np.sqrt(hsource**2 + dmast**2)
fwhmsource = np.radians(14.)
solid_angle_source_emitted = 2 * np.pi * (fwhmsource / 2.35)**2
solid_angle_horn_from_source = hornarea / distsource**2
dilution_factor = solid_angle_horn_from_source / solid_angle_source_emitted

0.01*dilution_factor*1e9  ## nW in a horn for 10 mW at the source


fwhm_max = 2.35 * np.degrees(1)    #### corresponds to solid angle of 2pi

nb = 100
log10source_power = np.linspace(-6, 0, nb)
source_power = 10**log10source_power
fwhmsource = np.linspace(0, 20, nb)

xx = np.zeros((nb, nb))
yy = np.zeros((nb, nb))
power_horn = np.zeros((nb, nb))
for i in np.arange(nb):
    for j in np.arange(nb):
        solid_angle_source_emitted = 2 * np.pi * (np.radians(fwhmsource[i]) / 2.35)**2
        solid_angle_horn_from_source = hornarea / distsource**2
        dilution_factor = solid_angle_horn_from_source / solid_angle_source_emitted
        power_horn[j,i] = dilution_factor * source_power[j]
        yy[i,j] = fwhmsource[i]
        xx[i,j] = source_power[j]



clf()
imshow(np.log10(power_horn),interpolation='nearest',extent=(np.min(fwhmsource),np.max(fwhmsource),np.min(log10source_power),np.max(log10source_power)),origin='lower',aspect='auto')
colorbar()
fmt = '%r W'
cs=contour(fwhmsource, log10source_power, power_horn,levels=min_pow)
clabel(cs,min_pow,inline=True,fmt=fmt,fontsize=10)
xlabel('FWHM Source [deg]')
ylabel('Log(Source Power [W])')
title('Log(Power entering a horn [W])')
plot(fwhmsource,fwhmsource*0-2,'r--',lw=3)
plot(log10source_power*0+14,log10source_power,'r--',lw=3)
savefig('source_power.png')

##### mouvements de la tour
deltax = [0.1,0.2,0.3,0.4,0.5] #meter
delta_theta = np.degrees(np.arctan(deltax / distsource))

delta_flux=np.zeros((len(fwhmsource),len(deltax)))
for i in np.arange(len(deltax)):
    delta_flux[:,i] = exp(-0.5*(delta_theta[i])**2/(fwhmsource/2.35)**2)

clf()
for i in np.arange(len(deltax)): plot(fwhmsource,delta_flux[:,i],label=str(deltax[i])+' m',lw=3)
legend(loc='lower right')
xlabel('FWHM Source [deg]')
ylabel('Horn Power variation')
title('Transverse shift of the source')
ylim(0.99,1)
plot(fwhmsource,fwhmsource*0+0.999,'k--',lw=3)
plot(log10source_power*0+14,linspace(0,1,nb),'k--',lw=3)
savefig('transverse_shift.png')


