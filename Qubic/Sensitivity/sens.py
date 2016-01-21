from __future__ import division
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np


#### at 150 GHz
### Old numbers from Michel, corresponding to summer
#NETpol=314*sqrt(2)*sqrt(2) ### one for polarization and one to be on sqrt(seconds)
### New numbers from Michel corresponding to winter
#NETpol=220*sqrt(2)*sqrt(2) ### one for polarization and one to be on sqrt(seconds)



#### at 220 GHz
### Old numbers from Michel, corresponding to summer
#NETpol=906*sqrt(2)*sqrt(2) ### one for polarization and one to be on sqrt(seconds)
### New numbers from Michel corresponding to winter
#NETpol=520*sqrt(2)*sqrt(2) ### one for polarization and one to be on sqrt(seconds)




### New numbers: 12/02/2015
### For QUBIC from email by M. Piat on feb 9th 2015
# DC: 
# 	150GHz: 291 / 369 uK.sqrt(s)
# 	220GHz: 547 / 840 uK.sqrt(s)
# Atacama (avec 230K / 266K et 5% / 10%):
# 	150GHz: 369 / 516 uK.sqrt(s)
# 	220GHz: 840 / 1356 uK.sqrt(s)
### need to apply sqrt(2) in order to account for polarization. But no other sqrt(2) as the required value needs to be in sqrt(s) not sqrt(Hz). One Hz corresponds to 1/2 seconds of integration.
### Concordia Average between summer and winter
### OLD numbers net150_concordia = 550.
### OLD numbers net220_concordia = 1450.
NETpol150_concordia_winter = 291.*np.sqrt(2)
NETpol150_concordia_summer = 369.*np.sqrt(2)
NETpol220_concordia_winter = 547.*np.sqrt(2)
NETpol220_concordia_summer = 840.*np.sqrt(2)
### Atacama Average between summer and winter
NETpol150_atacama_winter = 369.*np.sqrt(2)
NETpol150_atacama_summer = 516.*np.sqrt(2)
NETpol220_atacama_winter = 840.*np.sqrt(2)
NETpol220_atacama_summer = 1356.*np.sqrt(2)



def give_NET(NETpol, fsky, days, nbols):
	### Nuber of pixels of 1 arcmin**2
	npix = fsky*((180./np.pi)**2*4*np.pi)*(60**2)
	### number of seconds
	nsec = days*24.*3600
	### noise in uK.arcmin
	noise = sqrt(NETpol**2*npix/nsec/nbols)
	return noise


#### Latest numbers:
### Concordia Average between summer and winter
NETpol150_concordia_winter = 220.*np.sqrt(2)*np.sqrt(2)
NETpol150_concordia_summer = 314.*np.sqrt(2)*np.sqrt(2)
NETpol220_concordia_winter = 520.*np.sqrt(2)*np.sqrt(2)
NETpol220_concordia_summer = 906.*np.sqrt(2)*np.sqrt(2)
### Atacama Average between summer and winter
NETpol150_atacama_winter = 369.*np.sqrt(2)*np.sqrt(2)
NETpol150_atacama_summer = 516.*np.sqrt(2)*np.sqrt(2)
NETpol220_atacama_winter = 840.*np.sqrt(2)*np.sqrt(2)
NETpol220_atacama_summer = 1356.*np.sqrt(2)*np.sqrt(2)



#### QUBIC
fsky = 0.01
duration = 365*2
nbols = 1984/2
print('Concordia QUBIC fsky={0:3.1}%, duration = {1} days'.format(fsky, duration))
print('150 GHz Winter : NET_Pol_TOD = {0:5.2} muK.sqrt(s) => NET_pol_map = {1:5.2} muK.arcmin'.format(NETpol150_concordia_winter, give_NET(NETpol150_concordia_winter, fsky, duration, nbols)))
print('150 GHz Summer : NET_Pol_TOD = {0:5.2} muK.sqrt(s) => NET_pol_map = {1:5.2} muK.arcmin'.format(NETpol150_concordia_summer, give_NET(NETpol150_concordia_summer, fsky, duration, nbols)))
print('150 GHz Average : NET_Pol_TOD = {0:5.2} muK.sqrt(s) => NET_pol_map = {1:5.2} muK.arcmin'.format(0.5*(NETpol150_concordia_winter+NETpol150_concordia_summer), give_NET(0.5*(NETpol150_concordia_winter+NETpol150_concordia_summer), fsky, duration, nbols)))
print('220 GHz Winter : NET_Pol_TOD = {0:5.2} muK.sqrt(s) => NET_pol_map = {1:5.2} muK.arcmin'.format(NETpol220_concordia_winter, give_NET(NETpol220_concordia_winter, fsky, duration, nbols)))
print('220 GHz Summer : NET_Pol_TOD = {0:5.2} muK.sqrt(s) => NET_pol_map = {1:5.2} muK.arcmin'.format(NETpol220_concordia_summer, give_NET(NETpol220_concordia_summer, fsky, duration, nbols)))
print('220 GHz Average : NET_Pol_TOD = {0:5.2} muK.sqrt(s) => NET_pol_map = {1:5.2} muK.arcmin'.format(0.5*(NETpol220_concordia_winter+NETpol220_concordia_summer), give_NET(0.5*(NETpol220_concordia_winter+NETpol220_concordia_summer), fsky, duration, nbols)))


#### QUBIC
fsky = 0.01
duration = 365*2
nbols = 1984/2
print('atacama QUBIC fsky={0:3.1}%, duration = {1} days'.format(fsky, duration))
print('150 GHz Winter : NET_Pol_TOD = {0:5.2} muK.sqrt(s) => NET_pol_map = {1:5.2} muK.arcmin'.format(NETpol150_atacama_winter, give_NET(NETpol150_atacama_winter, fsky, duration, nbols)))
print('150 GHz Summer : NET_Pol_TOD = {0:5.2} muK.sqrt(s) => NET_pol_map = {1:5.2} muK.arcmin'.format(NETpol150_atacama_summer, give_NET(NETpol150_atacama_summer, fsky, duration, nbols)))
print('150 GHz Average : NET_Pol_TOD = {0:5.2} muK.sqrt(s) => NET_pol_map = {1:5.2} muK.arcmin'.format(0.5*(NETpol150_atacama_winter+NETpol150_atacama_summer), give_NET(0.5*(NETpol150_atacama_winter+NETpol150_atacama_summer), fsky, duration, nbols)))
print('220 GHz Winter : NET_Pol_TOD = {0:5.2} muK.sqrt(s) => NET_pol_map = {1:5.2} muK.arcmin'.format(NETpol220_atacama_winter, give_NET(NETpol220_atacama_winter, fsky, duration, nbols)))
print('220 GHz Summer : NET_Pol_TOD = {0:5.2} muK.sqrt(s) => NET_pol_map = {1:5.2} muK.arcmin'.format(NETpol220_atacama_summer, give_NET(NETpol220_atacama_summer, fsky, duration, nbols)))
print('220 GHz Average : NET_Pol_TOD = {0:5.2} muK.sqrt(s) => NET_pol_map = {1:5.2} muK.arcmin'.format(0.5*(NETpol220_atacama_winter+NETpol220_atacama_summer), give_NET(0.5*(NETpol220_atacama_winter+NETpol220_atacama_summer), fsky, duration, nbols)))




#### QUBIC 1%
### Old summer numbers
#give_NET(314*sqrt(2)*sqrt(2), 0.01, 365, 1984/2)   # = 4.3 muK.arcmin  at 150 GHz
#give_NET(906*sqrt(2)*sqrt(2), 0.01, 365, 1984/2)   # = 12.5 muK.arcmin  at 220 GHz
give_NET(220*sqrt(2)*sqrt(2), 0.01, 365, 1984/2)   # = 3.0 muK.arcmin  at 150 GHz WINTER
give_NET(314*sqrt(2)*sqrt(2), 0.01, 365, 1984/2)   # = 4.3 muK.arcmin  at 150 GHz SUMMER
give_NET(520*sqrt(2)*sqrt(2), 0.01, 365, 1984/2)   # = 7.2 muK.arcmin  at 220 GHz WINTER
give_NET(906*sqrt(2)*sqrt(2), 0.01, 365, 1984/2)   # = 12.5 muK.arcmin  at 220 GHz SUMMER

give_NET((220+314)/2*sqrt(2)*sqrt(2), 0.01, 365*2, 1984/2)   # = 3.7 muK.arcmin  at 150 GHz AVERAGE
give_NET((520+906)/2*sqrt(2)*sqrt(2), 0.01, 365*2, 1984/2)   # = 9.8 muK.arcmin  at 220 GHz AVERAGE


#### Qubic wide 4%
### Old summer numbers
#give_NET(314*sqrt(2)*sqrt(2), 0.04, 365, 1984/2)   # = 8.6 muK.arcmin  at 150 GHz
#give_NET(906*sqrt(2)*sqrt(2), 0.04, 365, 1984/2)   # = 25.0 muK.arcmin  at 220 GGHz
give_NET(220*sqrt(2)*sqrt(2), 0.04, 365, 1984/2)   # = 6.0 muK.arcmin  at 150 GHz WINTER
give_NET(314*sqrt(2)*sqrt(2), 0.04, 365, 1984/2)   # = 8.6 muK.arcmin  at 150 GHz SUMMER
give_NET(520*sqrt(2)*sqrt(2), 0.04, 365, 1984/2)   # = 14.3 muK.arcmin  at 220 GHz WINTER
give_NET(906*sqrt(2)*sqrt(2), 0.04, 365, 1984/2)   # = 25.0 muK.arcmin  at 220 GHz SUMMER

#### Planck Aniello
give_NET(17.4*sqrt(2), 1, 806, 1)   # = 35.9 muK.arcmin  at 143 GHz
give_NET(23.8*sqrt(2), 1, 806, 1)   # = 49.2 muK.arcmin  at 217 GGHz

#### Matt from Planck variance maps
40.16*sqrt(3)  # = 69.6 muK.arcmin  at 143 GHz
60.83*sqrt(3)  # = 105.4 muK.arcmin  at 217 GGHz

#### Matt 
## muK par sample
#143GHz : 1633 muK
#217GHz : 2535 muK
## hits moyen par pixel de 2048
#143 : 2012
#217 : 2113
pixsize2048 = ((180./np.pi)**2*4*np.pi)/(12*2048**2)*3600

1633/sqrt(2012)*sqrt(pixsize2048)   #62.5
2535/sqrt(2113)*sqrt(pixsize2048)   #94.7



