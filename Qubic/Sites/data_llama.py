from pylab import *
import numpy as np
import glob



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



def statstr(vec):
    m=np.mean(vec)
    s=np.std(vec)
    return '{0:.4f} +/- {1:.4f}'.format(m,s)

def statstr2(thedata, percentiles=[0.25, 0.5, 0.75]):
    sd = sort(thedata)
    truc = ''
    vals = []
    for perc in percentiles:
        vals.append(sd[perc*len(sd)])
        truc = truc + '{0:.1f} ({1:.0f}%) - '.format(sd[perc*len(sd)], perc*100)
    return truc[0:-3], vals




### Wind data
dir = '/Users/hamilton/Qubic/Sites/FromMarcelo/data_h/'
ts0, rec0, WS_ms_Avg, WS_ms_Std, WS_ms_Max, WS_ms_S_WVT, WindDir_D1_WVT, WindDir_SD1_WVT = np.loadtxt(dir+'chorrillo-t1m_cal.csv', usecols = [2,3,4,5,6,7,8,9]).T
import jdcal as jdcal
init = np.sum(jdcal.gcal2jd(1970,1,1))
jd0 = ts0/3600/24
jdcal.jd2gcal(init, jd0[0])
jdcal.jd2gcal(init, jd0[-1])
jdstartyear = np.sum(jdcal.gcal2jd(2011,1,1))
jdsince0 = init + ts0/3600/24 - jdstartyear


clf()
plot(jdsince0, WS_ms_Max, 'r,')
plot(jdsince0, WS_ms_Avg)
plot(jdsince0, WS_ms_Std)
xlabel('JD since 2011')
ylabel('Wind Speed $[m/s]$')

#### comparer avec https://science.nrao.edu/facilities/alma/aboutALMA/Technology/ALMA_Memo_Series/alma497/memo497.pdf
#### Figure 3.1
bla=hist(WS_ms_Avg, range=[0,30], bins=300, cumulative=True,normed=1, alpha=0.1)
clf()
hist(WS_ms_Avg, range=[0,30], bins=50, normed=False, weights=WS_ms_Avg*0+1e-5, color='red')
plot(bla[1][0:-1], bla[0],'k', lw=3, label = statstr2(WS_ms_Avg)[0])
legend()
xlim(0,30)
grid()
xlabel('Wind Speed [m/s]')
title('Average over 1 minute')
#savefig('wind-chorillos_avg1min_new.pdf')

bla=hist(WS_ms_Max, range=[0,30], bins=300, cumulative=True,normed=1, alpha=0.1)
clf()
hist(WS_ms_Max, range=[0,30], bins=50, normed=False, weights=WS_ms_Max*0+1.2e-5, color='red')
plot(bla[1][0:-1], bla[0],'k', lw=3, label = statstr2(WS_ms_Max)[0])
legend()
xlabel('Wind Speed [m/s]')
title('Maximum measured in 1 minute')
grid()
#savefig('wind-chorillos_max_new.pdf')


x = WS_ms_Avg * cos(-pi/2-np.radians(WindDir_D1_WVT))
y = WS_ms_Avg * sin(-pi/2-np.radians(WindDir_D1_WVT))
mask = (x==0) & (y==0)
clf()
bla = hist2d(x[~mask],y[~mask], bins=300, range=[[-15,15],[-5,5]],normed=True)
xlabel('m/sec ')
ylabel('m/sec ')

import scipy.ndimage
blanew = scipy.ndimage.gaussian_filter(bla[0], 1.5)

clf()
imshow(blanew.T, origin='lower',extent=[-15,15,-15,15])
xlabel('m/sec ')
ylabel('m/sec ')







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

plot(freq[fpos], 0.5/freq[fpos]**2)
plot(freq[fpos], 2500/freq[fpos]**1)





#       Within the file  "data_radiometer.tgz" you'll find the
#tipper data (one record per day) under the format YYY_MM_DD.csv
#In this file, the time stamp is provided in days (and fraction of the
#day) since 01/01/1992.
#
#	The tipper is located at:
#
#Longitude: 66d 28' 30" W
#Latitude: -24 d 11' 24"
#Altitude: 4813 asl


dir = '/Users/hamilton/Qubic/Sites/FromMarcelo/data_radiometer/'
dates = glob.glob(dir+'*.csv')

allyears = [0]
allmonths = [0]
alldays = [0]
allts = [0]
alltau = [0]
alltau_s = [0]

for fn in dates:
    f0 = ((fn.split('/')[-1]).split('.')[0]).split('_')
    print(f0)
    year = int(f0[0])
    month = int(f0[1])
    day = int(f0[2])
    ts, tau, tau_s = np.genfromtxt(fn,invalid_raise=False, skip_header=1, usecols=[0,1,2]).T
    yy = np.zeros(len(ts))+year
    mm = np.zeros(len(ts))+month
    dd = np.zeros(len(ts))+day
    allyears = np.append(allyears, yy)
    allmonths = np.append(allmonths, mm)
    alldays = np.append(alldays, dd)
    allts = np.append(allts, ts)
    alltau = np.append(alltau, tau)
    alltau_s = np.append(alltau_s, tau_s)

data = np.array([allyears, allmonths, alldays, allts, alltau, alltau_s])
data = data[:, 1:]

clf()
plot(data[3,:], data[4,:],',')

clf()
allvals = np.zeros((3, 12))
alleqtau = np.zeros(12)
for i in np.arange(12)+1:
    subplot(4,3,i)
    thedata = data[4,(data[1,:]==i) & (data[4,:] > 0)]
    err = np.sqrt(thedata)
    alleqerr = np.sqrt(1./np.sum(1./err**2))*np.sqrt(len(err))
    alleqtau[i-1] =  alleqerr**2 
    str, vals = statstr2(thedata)
    allvals[:,i-1] = np.array(vals)
    hist(thedata, bins=50, range=[0,1], normed=True, label='Month = {0:} : '.format(i)+ str)
    xlabel('Tau at 210 GHz')
    legend(fontsize=8)

clf()
xlim(0,13)
month=np.arange(12)+1
plot(month, allvals[0,:], 'ro-')
plot(month, allvals[1,:], 'go-')
plot(month, allvals[2,:], 'bo-')

#### From Emilianno Rasztocky August 2015 in a poster from LLAMA
opacity210_25 = np.array([0.26, 0.37, 0.11, 0.10, 0.08, 0.07, 0.06, 0.065, 0.08, 0.075, 0.10, 0.12])
opacity210_50 = np.array([0.39, 0.54, 0.17, 0.14, 0.115, 0.105, 0.08, 0.085, 0.11, 0.095, 0.15, 0.24])
opacity210_75 = np.array([0.55, 0.75, 0.27, 0.18, 0.17, 0.15, 0.12, 0.11, 0.15, 0.13, 0.24, 0.43]) 

clf()
xlim(0,13)
title('Chorillos')
xlabel('Month')
ylabel(r'$\tau(210\,GHz)$ (Data)')
plot(month, opacity210_25, 'r:')
plot(month, opacity210_50, 'g:', label= 'From LLAMA Poster')
plot(month, opacity210_75, 'b:')
plot(month, allvals[0,:], 'ro-')
plot(month, allvals[1,:], 'go-', label='From LLAMA Files')
plot(month, allvals[2,:], 'bo-')
plot(month, alleqtau, 'k', lw=3)
legend()
#savefig('tau_chorillos.png')

for i in xrange(len(month)):
    print('{0:} {1:5.3f} {2:5.3f} {3:5.3f} {4:5.3f}'.format(month[i], allvals[0,i], allvals[1,i], allvals[2,i], alleqtau[i]))


### Now atmospheric model from AM done by Andrea Tartari 
fghz, tau, tb = np.loadtxt('/Users/hamilton/Qubic/Sites/FromAndrea/chaj.out').T
clf()
yscale('log')
plot(fghz, tau)
xlabel('Freq [GHz]')
ylabel('Opacity')
title('From AM with Chajnantor configuration')
plot([150,150], [1e-5, 1e5], 'k:')
plot([210,210], [1e-5, 1e5], 'k:')
plot([220,220], [1e-5, 1e5], 'k:')
ylim(1e-3, 1e2)
#savefig('opacity_atm.png')

conv_210_150 = np.interp(150, fghz, tau) / np.interp(210, fghz, tau)
conv_210_220 = np.interp(220, fghz, tau) / np.interp(210, fghz, tau)



clf()
xlim(0,13)
title('Alto Chorillos - 25, 50 and 75% percentiles')
xlabel('Month')
ylabel('Atmospheric Opacity (from 210 GHz data)')
fill_between(month, allvals[0,:] * conv_210_220, y2=allvals[2,:] * conv_210_220, color='b', alpha=0.2)
plot(month, allvals[1,:] * conv_210_220, 'b', label ='220 GHz',lw=3)
#plot(month, alleqtau * conv_210_220, 'b', lw=3, label ='220 GHz')
fill_between(month, allvals[0,:] * conv_210_150, y2=allvals[2,:] * conv_210_150, color='r', alpha=0.2)
plot(month, allvals[1,:] * conv_210_150, 'r', label = '150 GHz',lw=3)
#plot(month, alleqtau * conv_210_150, 'r', lw=3, label ='220 GHz')
ylim(0,0.6)
xlim(1,12)
legend()
grid()
#savefig('tau_qubic_freqs.pdf')


clf()
xlim(0,13)
title(r'Chorillos - Equivalent $\tau$')
xlabel('Month')
ylabel(r'$\tau(\nu)$ (extrapolated from 210 GHz)')
plot(month, alleqtau * conv_210_220, 'b', lw=3, label ='220 GHz')
plot(month, alleqtau * conv_210_150, 'r', lw=3, label ='150 GHz')
ylim(0,0.4)
legend()
#savefig('tau_chorillos_freqs.png')

### extracted with eye from roberto Puddu's plot for concordia
pwv_dc = np.array([0.8, 0.7, 0.4, 0.27, 0.25, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.8])

clf()
xlim(0,13)
title(r'Dome C')
xlabel('Month')
ylabel(r'PWV from $\tau$ [mm]')
plot(month, pwv_dc,'r--', lw=3, label ='150 GHz - Dome C')
ylim(0,3)
legend()
#savefig('pwv_concordia.png')


### From Battistelli et al. 2012 Paper, done with Andrea Tartari : eq. 5
# tau = (Q + M*PWV)/Tatm at 150 GHz
M = (5.4+6.6)/2
Q = 4.2
### From Aristidi et al. 2005 
Tdomec=np.array([-30., -30., -41., -55., -58., -58., -59., -60., -59., -58., -50., -37.])+273.15
#
tau_dc_150 = (Q + M*pwv_dc)/Tdomec
## now because we have nothing better yet, we use the Chajnantor extrapolation to 220 GHz
tau_dc_220 = tau_dc_150 / conv_210_150 * conv_210_220

clf()
xlim(0,13)
xlabel('Month')
ylabel(r'Atmospheric Opacity $\tau$')
plot(month, alleqtau * conv_210_220, 'b', lw=3, label ='Chorillos - 220 GHz')
plot(month, alleqtau * conv_210_150, 'r', lw=3, label ='Chorillos - 150 GHz')
plot(month, tau_dc_220, 'b--', lw=3, label ='Concordia - 220 GHz')
plot(month, tau_dc_150, 'r--', lw=3, label ='Concordia - 150 GHz')
ylim(0,0.3)
legend()
#savefig('compare_tau.png')

### atmospheric temperatures at CHorillo
dir = '/Users/hamilton/Qubic/Sites/FromBeatriz/'
ts1, rec1, BattV_Min, AirTC_Avg, AirTC_Std, RH_Avg, RH_Std, SlrkW_Avg, SlrkW_Std, ETos, Rso = np.loadtxt(dir+'chorrillo-t10m-a6d4.dat').T
import jdcal as jdcal
init = np.sum(jdcal.gcal2jd(1970,1,1))
jd1 = ts1/3600/24
jdcal.jd2gcal(init, jd1[0])
jdcal.jd2gcal(init, jd1[-1])
jdstartyear = np.sum(jdcal.gcal2jd(2011,1,1))
jdsince1 = init + ts1/3600/24 - jdstartyear
Tchorillo = AirTC_Avg + 273.15
clf()
plot(jdsince1, Tchorillo,'r,',alpha=0.1)
xlabel('Julian Days (start 01/01/2011)')
ylabel('Atmosphere Temperature in Kelvins')

mm = np.zeros(len(jd1))
ff = np.zeros(len(jd1))
i=0
for jj in jd1:
    bla = jdcal.jd2gcal(init, jj)
    mm[i] = bla[1]
    ff[i] = bla[3]
    i+=1

TAvchorillo = np.zeros(12)
STAvchorillo = np.zeros(12)
for i in np.arange(12):
    TAvchorillo[i] = np.median(Tchorillo[mm==(i+1)])
    STAvchorillo[i] = np.std(Tchorillo[mm==(i+1)])

TAvchorillo_day = np.zeros(24)
STAvchorillo_day = np.zeros(24)
for i in np.arange(24):
    TAvchorillo_day[i] = np.median(Tchorillo[np.floor(ff*24)==i])
    STAvchorillo_day[i] = np.std(Tchorillo[np.floor(ff*24)==i])

clf()
plot(TAvchorillo_day)



clf()
plot(jdsince1, Tchorillo-273.15,'r.', alpha=0.05)
xlabel('Julian Days (start 01/01/2011)')
ylabel('Atmosphere Temperature in Kelvins')
legend(numpoints=1)
#savefig('temperature_sadc_new.pdf')

clf()
#errorbar(np.arange(12)+1,TAvchorillo-273.15, yerr=STAvchorillo, fmt='bo',xerr=0.5, label='Median over 2 years',lw=3)
fill_between(np.arange(12)+1,TAvchorillo-273.15+STAvchorillo, y2=TAvchorillo-273.15-STAvchorillo, alpha=0.3)
plot(np.arange(12)+1,TAvchorillo-273.15, 'b', label='Median over 2 years',lw=3)
grid()
xlim(1.,12.)
ylim(-15,15)
xlabel('Month')
ylabel('Atmospheric temperature $[K]$')
legend(numpoints=1)
#savefig('temperature_month_sadc_new.pdf')

clf()
#errorbar(np.arange(24),TAvchorillo_day-273.15, yerr=STAvchorillo_day, fmt='bo',xerr=0.5, label='Median over 2 years', lw=3)
fill_between(np.arange(24),TAvchorillo_day-273.15+STAvchorillo_day, y2=TAvchorillo_day-273.15-STAvchorillo_day, alpha=0.3)
plot(np.arange(24),TAvchorillo_day-273.15, 'b', label='Median over 2 years',lw=3)
grid()
xlim(0,23)
xlabel('Hour')
ylabel('Atmospheric Temperature $[K]$')
legend(numpoints=1)
#savefig('temperature_hour_sadc_new.pdf')


### Now one can calculate NETs
elevation_obs = 50.



### data for Chajnantor at 350microns = 857 GHz : https://www.cfa.harvard.edu/~aas/oldtenmeter/opacity.htm
tau_857_chaj = np.array([3.8, 2.7, 2.6, 2.9, 2.65, 2.45, 2.0, 1.5, 1.8, 1.9, np.nan, np.nan])
#### conversion to 150 and 220 from AM Chajnantor by A. Tartari
conv_857_150 = 2.9e-2 / 1.39
conv_857_220 = 5.1e-2 / 1.39
conv_857_210 = 5.1e-2 / 1.39
plot(month, tau_857_chaj * conv_857_150, 'r')
plot(month, tau_857_chaj * conv_857_220, 'b')



### tau plot again
clf()
xlim(0,13)
xlabel('Month')
ylabel(r'Atmospheric Opacity $\tau$')
plot(month, alleqtau * conv_210_220, 'b', lw=3, label ='Chorillos - 220 GHz')
plot(month, alleqtau * conv_210_150, 'r', lw=3, label ='Chorillos - 150 GHz')
plot(month, tau_dc_220, 'b--', lw=3, label ='Concordia - 220 GHz')
plot(month, tau_dc_150, 'r--', lw=3, label ='Concordia - 150 GHz')
ylim(0,0.3)
legend()


### First get emissivities from tau
em150_dc = 1-np.exp(-tau_dc_150/np.cos(np.radians(90-elevation_obs)))
em220_dc = 1-np.exp(-tau_dc_220/np.cos(np.radians(90-elevation_obs)))
em150_ch = 1-np.exp(-alleqtau * conv_210_150/np.cos(np.radians(90-elevation_obs)))
em220_ch = 1-np.exp(-alleqtau * conv_210_220/np.cos(np.radians(90-elevation_obs)))
clf()
xlim(0,13)
xlabel('Month')
ylabel(r'Atmospheric Emissivity')
plot(month, em220_ch, 'b', lw=3, label ='Chorillos - 220 GHz')
plot(month, em150_ch, 'r', lw=3, label ='Chorillos - 150 GHz')
plot(month, em220_dc, 'b--', lw=3, label ='Concordia - 220 GHz')
plot(month, em150_dc, 'r--', lw=3, label ='Concordia - 150 GHz')
ylim(0,0.3)
legend()
#savefig('atm_emissivity.png')


em210_ch = 1-np.exp(-alleqtau/np.cos(np.radians(90-elevation_obs)))
trans150_dc = 1.-em150_dc
trans220_dc = 1.-em220_dc
trans150_ch = 1.-em150_ch
trans220_ch = 1.-em150_ch




clf()
xlim(0,13)
xlabel('Month')
ylabel(r'Atmospheric Emissivity')
plot(month, em220_ch, 'b', lw=3, label ='Chorillos - 220 GHz')
plot(month, em150_ch, 'r', lw=3, label ='Chorillos - 150 GHz')
ylim(0,0.3)
legend()
grid()
#savefig('atm_emissivity-cho.png')

# clf()
# xlim(0,13)
# xlabel('Month')
# ylabel(r'Atmospheric Emissivity')
# plot(month, em220_ch, 'b', lw=3, label ='Chorillos - 220 GHz')
# plot(month, em150_ch, 'r', lw=3, label ='Chorillos - 150 GHz')
# plot(month, em220_chaj, 'b--', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 220 GHz')
# plot(month, em150_chaj, 'r--', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 150 GHz')
# ylim(0,0.3)
# legend()
# grid()
#savefig('atm_emissivity-cho-chaj.png')

clf()
xlim(0,13)
xlabel('Month')
ylabel(r'Atmospheric Brightness Temperature (K)')
plot(month, em220_ch * TAvchorillo, 'b', lw=3, label ='Chorillos - 220 GHz')
plot(month, em150_ch * TAvchorillo, 'r', lw=3, label ='Chorillos - 150 GHz')
ylim(0,110)
legend()
grid()
#savefig('atm_brightness-cho.png')

clf()
xlim(0,13)
xlabel('Month')
ylabel(r'Atmospheric Brightness Temperature (K)')
plot(month, em220_ch * TAvchorillo, 'b', lw=3, label ='Chorillos - 220 GHz')
plot(month, em150_ch * TAvchorillo, 'r', lw=3, label ='Chorillos - 150 GHz')
# plot(month, em220_chaj * TAvchorillo, 'b--', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 220 GHz')
# plot(month, em150_chaj * TAvchorillo, 'r--', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 150 GHz')
ylim(0,110)
legend()
grid()
#savefig('atm_brightnesscho-chaj.png')



from Sites import loading
theatm = {'Name':'Dome C  ', 'T':Tdomec[0],    'e':em150_dc[0], 'trans':1}
vals = loading.give_NET(atm=theatm, verbose=True, bol_efficiency=0.8, freqGHz=150.)
theatm = {'Name':'Dome C  ', 'T':Tdomec[0],    'e':em220_dc[0], 'trans':1}
vals = loading.give_NET(atm=theatm, verbose=True, bol_efficiency=0.8, freqGHz=220.)

fraction = 0.76




net150_dc = np.zeros(12)
net220_dc = np.zeros(12)
net150_ch = np.zeros(12)
net220_ch = np.zeros(12)
margin = 0.8
for i in np.arange(12):
    theatm = {'Name':'Atm Chorillos  ', 'T':TAvchorillo[i],    'e':em150_ch[i], 'trans':1}
    vals = loading.give_NET(atm=theatm, verbose=False, bol_efficiency=margin, freqGHz=150.)
    net150_ch[i] = vals[3]

    theatm = {'Name':'Atm Chorillos  ', 'T':TAvchorillo[i],    'e':em220_ch[i], 'trans':1}
    vals = loading.give_NET(atm=theatm, verbose=False, bol_efficiency=margin, freqGHz=220.)
    net220_ch[i] = vals[3]

    theatm = {'Name':'Dome C  ', 'T':Tdomec[i],    'e':em150_dc[i], 'trans':1}
    vals = loading.give_NET(atm=theatm, verbose=False, bol_efficiency=margin, freqGHz=150.)
    net150_dc[i] = vals[3]

    theatm = {'Name':'Dome C  ', 'T':Tdomec[i],    'e':em220_dc[i], 'trans':1}
    vals = loading.give_NET(atm=theatm, verbose=False, bol_efficiency=margin, freqGHz=220.)
    net220_dc[i] = vals[3]



clf()
xlim(0,13)
xlabel('Month')
ylabel(r'NET [$\mu K.s^{1/2}$]')
plot(month, net220_ch, 'b--', lw=3, label ='Chorillos - 220 GHz')
plot(month, net150_ch, 'r--', lw=3, label ='Chorillos - 150 GHz')
plot(month, net220_dc, 'b', lw=3, label ='Concordia - 220 GHz')
plot(month, net150_dc, 'r', lw=3, label ='Concordia - 150 GHz')
legend()
ylim(0,1000)
grid()
#savefig('NET_results.png')


clf()
xlim(0,13)
xlabel('Month')
ylabel(r'NET$^2$ ratio Chorillos/Concordia')
plot(month, (net220_ch/net220_dc)**2, 'b', lw=3, label ='220 GHz')
plot(month, (net150_ch/net150_dc)**2, 'r', lw=3, label ='150 GHz')
legend()
grid()
#savefig('NET2_ratio.png')



#### Simulations will be performed with the following numbers:
frac_time = 0.4
monthsok = np.arange(12)[2:11]

print(np.mean(net150_dc))
print(np.mean(net220_dc))
print(np.mean(net150_ch[monthsok]))
print(np.mean(net220_ch[monthsok]))


final_net150_dc = np.mean(net150_dc)
final_net220_dc = np.mean(net220_dc)
final_net150_ch = np.mean(net150_ch[monthsok])/np.sqrt(frac_time*len(monthsok)*1./12)
final_net220_ch = np.mean(net220_ch[monthsok])/np.sqrt(frac_time*len(monthsok)*1./12)

print(final_net150_dc, final_net220_dc)
print(final_net150_ch, final_net220_ch)
print(final_net150_ch/final_net150_dc, final_net220_ch/final_net220_dc)
print((final_net150_ch/final_net150_dc)**2, (final_net220_ch/final_net220_dc)**2)




#### Checking things for Beatriz

names = ['BattV_Min', 'AirTC_Avg', 'AirTC_Std', 'RH_Avg', 'RH_Std', 'SlrkW_Avg', 'SlrkW_Std', 'ETos', 'Rso', 
                'WS_ms_Avg', 'WS_ms_Std', 'WS_ms_Max', 'WS_ms_S_WVT', 'WindDir_D1_WVT', 'WindDir_SD1_WVT']
data = [BattV_Min, AirTC_Avg, AirTC_Std, RH_Avg, RH_Std, SlrkW_Avg, SlrkW_Std, ETos, Rso, 
                WS_ms_Avg, WS_ms_Std, WS_ms_Max, WS_ms_S_WVT, WindDir_D1_WVT, WindDir_SD1_WVT]


for i in xrange(len(names)):
    nn = names[i]
    dd = data[i]
    dd = dd[isfinite(dd)]
    aa = statstr2(dd)[1]
    print('{0:15} | {1:6.2f} | {2:6.2f} | {3:6.2f} | {4:6.2f} | {5:6.2f} | {6:6.2f} | {7:6.2f} '.format(nn, np.mean(dd),
            np.std(dd), np.min(dd), aa[0], aa[1], aa[2], np.max(dd)))






################## Much simpler but looks very optimistic
h = 6.626070040e-34
c = 299792458
k = 1.3806488e-23

def Bnu(nuGHz, T):
    nu = nuGHz * 1e9
    val = 2 * h * nu**3 / c**2 / (np.exp(h * nu / k / T) -1)
    return val
    
def dBnudT(nuGHz, T):
    bnu = Bnu(nuGHz, T)
    nu = nuGHz * 1e9
    val = c**2 / 2 / nu**2 * bnu * bnu * np.exp(h * nu / k / T) / k / T**2
    return val
    
def nep2net(nep, nuGHz, eff, deltanuGHz, T):
    deltanu = deltanuGHz * 1e9
    nu = nuGHz * 1e9
    lambd = c / nu
    net = nep / (lambd**2 * eff * deltanu) / dBnudT(nuGHz, T) * 1e6
    return net/sqrt(2)*sqrt(2)   ### so that we are in muK.sqrt(s) for polarized if NEP is for intensity in muK/sqrt(Hx)
 
import qubic
def netpol_qubic(f, relative_bw, eff):
    scene = qubic.QubicScene(256)
    inst = qubic.QubicInstrument(filter_nu=f*1e9, filter_relative_bandwidth=relative_bw)
    nep_photons = np.mean(inst._get_noise_photon_nep(scene)*sqrt(2))   #sqrt(2) in order to have Polarized NEP
    nep_detector = inst.detector.nep
    nep_tot = np.sqrt(nep_photons**2 + nep_detector**2)
    net_photons = nep2net(nep_photons, f, eff, relative_bw*f, 2.728)
    net_detector = nep2net(nep_detector, f, eff, relative_bw*f, 2.728)
    net_tot = nep2net(nep_tot, f, eff, relative_bw*f, 2.728)
    return net_tot

   

print(netpol_qubic(150., 0.25, 0.3))
print(netpol_qubic(220., 0.25, 0.3))

















