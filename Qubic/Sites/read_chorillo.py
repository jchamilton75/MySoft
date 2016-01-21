from pylab import *
import numpy as np



def profile(x,y,range=None,nbins=10,fmt=None,plot=True, dispersion=True, median=False):
  if range == None:
    mini = np.min(x)
    maxi = np.max(x)
  else:
    mini = range[0]
    maxi = range[1]
  dx = (maxi - mini) / nbins
  xmin = np.linspace(mini,maxi-dx,nbins)
  xmax = xmin + dx
  xc = xmin + dx / 2
  yval = np.zeros(nbins)
  dy = np.zeros(nbins)
  dx = np.zeros(nbins) + dx / 2
  for i in np.arange(nbins):
    ok = (x > xmin[i]) & (x < xmax[i])
    yval[i] = np.mean(y[ok])
    if median: yval[i]=np.median(y[ok])
    if dispersion: 
      fact = 1
    else:
      fact = np.sqrt(len(y[ok]))
    dy[i] = np.std(y[ok])/fact
  if plot: errorbar(xc, yval, xerr=dx, yerr=dy, fmt=fmt)
  return xc, yval, dx, dy




#Le format est 
#Time Tau Delta_Tau 
#avec Time from 31/12/1991 at 0 hs
#Ils m’ont aussi donné une valeur pour la conversion : PWV = 14.425 * Tau
#mais apparemment c’est fait avec un modèle d’atmosphère pas très sur…

import numpy as np
data = np.loadtxt('chorrillo_2009.dat')
tau = data[:,1]
dtau = data[:,2]

import jdcal as jdcal
init=np.sum(jdcal.gcal2jd(1991,12,31))
jd = init+data[:,0]
jdcal.jd2gcal(0,jd[0])


jdstart = np.sum(jdcal.gcal2jd(2009,1,1))
datestart = jdcal.jd2gcal(0,jdstart)
day = jd - jdstart

clf()
plot(day/365*12,tau)
xlim(0,12)

pwv = 14.425 * tau
clf()
plot(day/365*12,pwv)
xlim(0,12)

clf()
subplot(2,1,1)
xc,yc, dx, dy = profile(day,tau, nbins=365, range=[0,365], plot=False)
xc,med, a,b = profile(day,tau, nbins=365, range=[0,365], median=True, plot=False)
fill_between(xc/365*12,yc-dy,y2=yc+dy, color='red',alpha=0.1)
plot(xc/365*12,yc,'red',label='Mean +/- 1$\sigma$')
plot(xc/365*12,med,'k',label='Median')
legend(loc='upper left')
xlim(0,12)
ylim(0,0.3)
xlabel('Month')
ylabel('Tau at 210 GHz')
subplot(2,1,2)
xc,yc, dx, dy = profile(day,pwv, nbins=365, range=[0,365], plot=False)
xc,med, a,b = profile(day,pwv, nbins=365, range=[0,365], median=True, plot=False)
fill_between(xc/365*12,yc-dy,y2=yc+dy, color='red',alpha=0.1)
plot(xc/365*12,yc,'red',label='Mean +/- 1$\sigma$')
plot(xc/365*12,med,'k',label='Median')
plot([0,12],[1,1],'k:')
ylim(0,3)
xlim(0,12)
xlabel('Month')
ylabel('PWV')
legend(loc='upper left')
savefig('chorrillo.png')



################ Read Data from Beatriz Garcia 06/09/2015
#Longitude: 66d 28' 30" W
#Latitude: -24 d 11' 24"
#Altitude: 4813 asl
#WS_ms_S_WVT: Is a redundant variable. It is the mean wind speed.
#WinDir_D1_WVT: Mean azimuth wind direction (in units of degree).
#WinDir_SD1_WVT: Standard Deviation of previous item.

dir = '/Users/hamilton/Qubic/Sites/FromBeatriz/'
ts0, rec0, WS_ms_Avg, WS_ms_Std, WS_ms_Max, WS_ms_S_WVT, WindDir_D1_WVT, WindDir_SD1_WVT = np.loadtxt(dir+'chorrillo-t1m-6619.dat').T
ts1, rec1, BattV_Min, AirTC_Avg, AirTC_Std, RH_Avg, RH_Std, SlrkW_Avg, SlrkW_Std, ETos, Rso = np.loadtxt(dir+'chorrillo-t10m-a6d4.dat').T
ts2, rec2, BP_61302V = np.loadtxt(dir+'chorrillo-t60m-e00f.dat').T

#dir = '/Users/hamilton/Qubic/Sites/FromMarcelo/data_h/'
#ts0, rec0, WS_ms_Avg, WS_ms_Std, WS_ms_Max, WS_ms_S_WVT, WindDir_D1_WVT, WindDir_SD1_WVT = np.loadtxt(dir+'chorrillo-t1m_cal.csv').T
#ts1, rec1, BattV_Min, AirTC_Avg, AirTC_Std, RH_Avg, RH_Std, SlrkW_Avg, SlrkW_Std, ETos, Rso = np.loadtxt(dir+'chorrillo-t10m.csv').T
#ts2, rec2, BP_61302V = np.loadtxt(dir+'chorrillo-t60m.csv').T



import jdcal as jdcal
init = np.sum(jdcal.gcal2jd(1970,1,1))
jd0 = ts0/3600/24
jdcal.jd2gcal(init, jd0[0])
jdcal.jd2gcal(init, jd0[-1])
jdstartyear = np.sum(jdcal.gcal2jd(2011,1,1))

jdsince0 = init + ts0/3600/24 - jdstartyear
jdsince1 = init + ts1/3600/24 - jdstartyear
jdsince2 = init + ts2/3600/24 - jdstartyear

clf()
plot(jdsince2, BP_61302V)
xlabel('JD since 2011')
ylabel('Pressure $[hPa]$')

clf()
plot(jdsince1, AirTC_Avg)
plot(jdsince1, AirTC_Std)
xlabel('JD since 2011')
ylabel('Air Temperature $[C]$')

clf()
plot(jdsince1, RH_Avg)
plot(jdsince1, RH_Std)
xlabel('JD since 2011')
ylabel('Relative Humidity $[%]$')

clf()
plot(jdsince1, SlrkW_Avg)
plot(jdsince1, SlrkW_Std)
xlabel('JD since 2011')
ylabel('Solar Radiance $[kW/m^2]$')

clf()
plot(jdsince0, WS_ms_Max, 'r,')
plot(jdsince0, WS_ms_Avg)
plot(jdsince0, WS_ms_Std)
xlabel('JD since 2011')
ylabel('Wind Speed $[m/s]$')

ftws = np.fft.fft(WS_ms_Avg)
freq = np.fft.fftfreq(len(WS_ms_Avg), d = (jdsince0[1]-jdsince0[0])*24*3600 )

import scipy.ndimage.filters as filt
clf()
xscale('log')
yscale('log')
xlim(1e-4, 1e-2)
ylim(1e5, 1e8)
fpos = freq > 0
plot(freq[fpos], filt.gaussian_filter1d(np.abs(ftws[fpos])**2, 1000))
xlabel('Frequency $[Hz]$')
ylabel('Wind Speed Power Spectrum')




