from pylab import *
import numpy as np
from Sites import loading

#### From Emilianno Rasztocky August 2015
month = np.arange(12) + 1
opacity210_25 = np.array([0.26, 0.37, 0.11, 0.10, 0.08, 0.07, 0.06, 0.065, 0.08, 0.075, 0.10, 0.12])
opacity210_50 = np.array([0.39, 0.54, 0.17, 0.14, 0.115, 0.105, 0.08, 0.085, 0.11, 0.095, 0.15, 0.24])
opacity210_75 = np.array([0.55, 0.75, 0.27, 0.18, 0.17, 0.15, 0.12, 0.11, 0.15, 0.13, 0.24, 0.43]) 

clf()
xlim(0,13)
plot(month, opacity210_25, 'ro-')
plot(month, opacity210_50, 'go-')
plot(month, opacity210_75, 'bo-')

Tatm_sac = 265
elevation_obs = 55

em210_25 = 1-np.exp(-opacity210_25/np.cos(np.radians(90-elevation_obs)))
em210_50 = 1-np.exp(-opacity210_50/np.cos(np.radians(90-elevation_obs)))
em210_75 = 1-np.exp(-opacity210_75/np.cos(np.radians(90-elevation_obs)))

clf()
xlim(0,13)
plot(month, em210_25, 'ro-')
plot(month, em210_50, 'go-')
plot(month, em210_75, 'bo-')

Teq210_25 = Tatm_sac * em210_25
Teq210_50 = Tatm_sac * em210_50
Teq210_75 = Tatm_sac * em210_75

clf()
xlim(0,13)
plot(month, Teq210_25, 'ro-')
plot(month, Teq210_50, 'go-')
plot(month, Teq210_75, 'bo-')


#### Concordia
atm_DomeC_summer = {'Name':'Atm Dome C  ', 'T':200,    'e':0.025, 'trans':1}
atm_DomeC_winter = {'Name':'Atm Dome C  ', 'T':260,    'e':0.06,  'trans':1}

val_DomeC_summer = loading.give_NET(atm=atm_DomeC_summer, verbose=True)
net_150_DomeC_summer = val_DomeC_summer[3][0]
net_220_DomeC_summer = val_DomeC_summer[3][1]

val_DomeC_winter = loading.give_NET(atm=atm_DomeC_winter, verbose=True)
net_150_DomeC_winter = val_DomeC_winter[3][0]
net_220_DomeC_winter = val_DomeC_winter[3][1]

#### Chorillos
net_150_chorillos_25 = np.zeros(len(month))
net_150_chorillos_50 = np.zeros(len(month))
net_150_chorillos_75 = np.zeros(len(month))
net_220_chorillos_25 = np.zeros(len(month))
net_220_chorillos_50 = np.zeros(len(month))
net_220_chorillos_75 = np.zeros(len(month))

Tchorillos = 273.
for i in np.arange(len(month)):
        theatm = {'Name':'Atm Chorillos  ', 'T':Tchorillos,    'e':em210_25[i], 'trans':1}
        vals = loading.give_NET(atm=theatm, verbose=False)
        net_150_chorillos_25[i] = vals[3][0]
        net_220_chorillos_25[i] = vals[3][1]
        
        theatm = {'Name':'Atm Chorillos  ', 'T':Tchorillos,    'e':em210_50[i], 'trans':1}
        vals = loading.give_NET(atm=theatm, verbose=False)
        net_150_chorillos_50[i] = vals[3][0]
        net_220_chorillos_50[i] = vals[3][1]
        
        theatm = {'Name':'Atm Chorillos  ', 'T':Tchorillos,    'e':em210_75[i], 'trans':1}
        vals = loading.give_NET(atm=theatm, verbose=False)
        net_150_chorillos_75[i] = vals[3][0]
        net_220_chorillos_75[i] = vals[3][1]
        



clf()
subplot(2,1,1)
xlim(0,13)
ylim(0,700)
xlabel('Month')
ylabel(r'Polarized NET [$\mu K.s^{-1/2}$]')
title('150 GHz')
fill_between(month, net_150_chorillos_25, y2=net_150_chorillos_75, color='r', alpha=0.3)
plot(month, net_150_chorillos_50, 'r', lw=2)
plot(month, net_150_DomeC_winter+month*0, 'k')
plot(month, net_150_DomeC_summer+month*0, 'k')
plot(month, 220.*np.sqrt(2)+month*0, 'b')
plot(month, 314.*np.sqrt(2)+month*0, 'b')
subplot(2,1,2)
xlim(0,13)
ylim(0,1600)
title('220 GHz')
xlabel('Month')
ylabel(r'Polarized NET [$\mu K.s^{-1/2}$]')
fill_between(month, net_220_chorillos_25, y2=net_220_chorillos_75, color='r', alpha=0.3)
plot(month, net_220_chorillos_50, 'r', lw=2)
plot(month, net_220_DomeC_winter+month*0, 'k')
plot(month, net_220_DomeC_summer+month*0, 'k')
plot(month, 520.*np.sqrt(2)+month*0, 'b')
plot(month, 906.*np.sqrt(2)+month*0, 'b')








