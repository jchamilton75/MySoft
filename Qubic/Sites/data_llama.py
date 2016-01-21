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
        truc = truc + '{0:.3f} ({1:.0f}%) - '.format(sd[perc*len(sd)], perc*100)
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
hist(WS_ms_Avg, range=[0,30], bins=100, normed=True)
plot(bla[1][0:-1], bla[0]*0.12,'r', label = statstr2(WS_ms_Avg)[0])
legend()
xlabel('1 min Averaged Wind Speed [m/s]')
savefig('wind-chorillos_avg1min.png')

bla=hist(WS_ms_Max, range=[0,30], bins=300, cumulative=True,normed=1, alpha=0.1)
clf()
hist(WS_ms_Max, range=[0,30], bins=30, normed=True)
plot(bla[1][0:-1], bla[0]*0.12,'r', label = statstr2(WS_ms_Max)[0])
legend()
xlabel('Max Wind Speed [m/s]')
savefig('wind-chorillos_max.png')



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
    ts, tau, tau_s = np.genfromtxt(fn,invalid_raise=False, skiprows=1, usecols=[0,1,2]).T
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

### Now atmospheric model from ATM done by Andrea Tartari 
fghz, tau, tb = np.loadtxt('/Users/hamilton/Qubic/Sites/FromAndrea/chaj.out').T
clf()
yscale('log')
plot(fghz, tau)
xlabel('Freq [GHz]')
ylabel('Opacity')
title('From ATM with Chajnantor configuration')
plot([150,150], [1e-5, 1e5], 'k:')
plot([210,210], [1e-5, 1e5], 'k:')
plot([220,220], [1e-5, 1e5], 'k:')
ylim(1e-3, 1e2)
#savefig('opacity_atm.png')

conv_210_150 = np.interp(150, fghz, tau) / np.interp(210, fghz, tau)
conv_210_220 = np.interp(220, fghz, tau) / np.interp(210, fghz, tau)

clf()
xlim(0,13)
title('Chorillos - 25, 50 and 75% percentiles')
xlabel('Month')
ylabel(r'$\tau(\nu)$ (extrapolated from 210 GHz)')
fill_between(month, allvals[0,:] * conv_210_220, y2=allvals[2,:] * conv_210_220, color='b', alpha=0.2)
plot(month, allvals[1,:] * conv_210_220, 'b', label ='220 GHz')
#plot(month, alleqtau * conv_210_220, 'b', lw=3, label ='220 GHz')
fill_between(month, allvals[0,:] * conv_210_150, y2=allvals[2,:] * conv_210_150, color='r', alpha=0.2)
plot(month, allvals[1,:] * conv_210_150, 'r', label = '150 GHz')
#plot(month, alleqtau * conv_210_150, 'r', lw=3, label ='220 GHz')
ylim(0,0.3)
legend()


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
plot(jdsince1, Tchorillo)
xlabel('JD since 2011')
ylabel('Air Temperature $[K]$')

mm = np.zeros(len(jd1))
i=0
for jj in jd1:
    mm[i] = jdcal.jd2gcal(init, jj)[1]
    i+=1

TAvchorillo = np.zeros(12)
for i in np.arange(12):
    TAvchorillo[i] = np.mean(Tchorillo[mm==(i+1)])

clf()
plot(TAvchorillo)




### Now one can calculate NETs
elevation_obs = 50.

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

np.mean(net150_dc)
np.mean(net220_dc)
np.mean(net150_ch[monthsok])
np.mean(net220_ch[monthsok])


final_net150_dc = np.mean(net150_dc)
final_net220_dc = np.mean(net220_dc)
final_net150_ch = np.mean(net150_ch[monthsok])/np.sqrt(frac_time*len(monthsok)*1./12)
final_net220_ch = np.mean(net220_ch[monthsok])/np.sqrt(frac_time*len(monthsok)*1./12)

print(final_net150_dc, final_net220_dc)
print(final_net150_ch, final_net220_ch)
print(final_net150_ch/final_net150_dc, final_net220_ch/final_net220_dc)
print((final_net150_ch/final_net150_dc)**2, (final_net220_ch/final_net220_dc)**2)



#### now read simulations made with these numbers
####### Chains for ANR 2015 simulations
def upperlimit(chain,key,level=0.95):
	sorteddata = np.sort(chain[key])
	return sorteddata[level*len(sorteddata)]

from McMc import mcmc
from Sensitivity import data4mcmc
rep = '/Users/hamilton/Qubic/Sites/SimsComparisonSites/'

all_ul_B = []
all_ul_D = []
all_ul_nofg = []

for localisation in ['atac', 'conc']:
    #for eff in ['03', '05', '1']:
    for eff in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '1']:
        site = localisation + '_' + eff + '_' 
        config = ''
        #site = 'atac_05_'
        chain_B_r_dl_b = data4mcmc.readchains(rep+site+'instrumentB_r_dl_b.db')
        chain_D_r_dl_b = data4mcmc.readchains(rep+site+'instrumentD_r_dl_b.db')
        chain_nofg_r = data4mcmc.readchains(rep+site+'instrumentNofg_r.db')

        truer = 0.
        truebeta = 1.59
        truedl = 13.4 * 0.45
        truealpha = -2.42
        trueT = 19.6
        level =0.95
        cl = int(level*100)

        ########### r dl and beta
        sm=4
        histn=4
        alpha =0.5

        nbins=100
        from scipy.ndimage import gaussian_filter1d
        bla = np.histogram(chain_nofg_r['r'],bins=nbins,normed=True)
        xhist=(bla[1][0:nbins]+bla[1][1:nbins+1])/2
        ss=np.std(chain_nofg_r['r'])
        yhist=gaussian_filter1d(bla[0],ss/histn/(xhist[1]-xhist[0]), mode='nearest')
        plot(xhist,yhist/max(yhist))

        thelimits = [[truebeta*0.85, truebeta*1.15],[0,0.08]]

        bla=mcmc.matrixplot(chain_B_r_dl_b,['betadust','r'], 'green', sm, 
            limits=thelimits, alpha=alpha,histn=histn, truevals = [truebeta, truer])

        ### Au final
        clf()
        #c=mcmc.matrixplot(chain_C_r_dl_b,['betadust','r'], 'black', sm, limits=[[truebeta*0.95, truebeta*1.05],[0,0.05]], alpha=alpha,histn=histn, truevals = [truebeta, truer])
        b=mcmc.matrixplot(chain_B_r_dl_b,['betadust','r'], 'blue', sm, limits=thelimits, 
            alpha=alpha,histn=histn, truevals = [truebeta, truer])
        d=mcmc.matrixplot(chain_D_r_dl_b,['betadust','r'], 'red', sm, limits=thelimits, 
            alpha=alpha,histn=histn, truevals = [truebeta, truer])
        subplot(2,2,4)
        noFG = plot(xhist,yhist/max(yhist), color='green', label='toto')
        subplot(2,2,2)
        #legC = '150x2+353 : r < {0:5.2f} (95% CL)'.format(upperlimit(chain_C_r_dl_b,'r'))
        ul_B = upperlimit(chain_B_r_dl_b,'r')
        legB = '150+220 : r < {0:5.3f} (95% CL)'.format(ul_B)
        ul_D = upperlimit(chain_D_r_dl_b,'r')
        legD = '150+220+353: r < {0:5.3f} (95% CL)'.format(ul_D)
        ul_nofg = upperlimit(chain_nofg_r,'r')
        legnoFG = 'No Foregrounds: r < {0:5.3f} (95% CL)'.format(ul_nofg)
        legend([b, d, bla],[legB, legD, legnoFG], frameon=False, title='QUBIC 2 years '+config+site)
        savefig(site+'.png')
        
        all_ul_B. append(ul_B)
        all_ul_D. append(ul_D)
        all_ul_nofg. append(ul_nofg)


all_ul_B = np.reshape(all_ul_B, (2,len(all_ul_B)/2))
all_ul_D = np.reshape(all_ul_D, (2,len(all_ul_D)/2))
all_ul_nofg = np.reshape(all_ul_nofg, (2,len(all_ul_nofg)/2))

#eff = np.array([0.3,0.5, 1.])
eff = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
alleff=linspace(0.1, 1, 100)
clf()
grid()
xlim(0.1,1.)
ylim(0,0.05)
xlabel('Observation Efficiency (%)')
ylabel('95% Upper-limit on $r$ (2 years)')
title('QUBIC 150 + 220 GHz + Planck 353 GHz')
plot(eff, eff*1000, 'r', lw=3)
plot(eff, eff*1000, 'r-', lw=3, label = 'Concordia')
plot(eff, eff*1000, 'r--', lw=3, label = 'Chorillos')
plot(eff, all_ul_D[0,:], 'r--', lw=3)
plot(eff, all_ul_D[1,:], 'r-', lw=3)
legend()
#savefig('sensitivity_efficiency.png')

#### Concordia
badmonths = 2.
hours_below = 0.
cycle_he7 = 4.
hours_selfcalib = np.array([0., 6., 12.])
badhours = np.max([hours_below, cycle_he7])
badhours_remain = badhours-cycle_he7
hours_selfcalib_cost = np.clip(hours_selfcalib - badhours_remain, 0, 24)
hours_obsfield = 24 - badhours - hours_selfcalib_cost
hours_ratio = hours_obsfield/24
fracmax_concordia = (12-badmonths)*1./12 * hours_ratio

#### Chorillos
badmonths = 3.
hours_below = 0.6*24
cycle_he7 = 4.
hours_selfcalib = np.array([0., 6., 12.])
badhours = np.max([hours_below, cycle_he7])
badhours_remain = badhours-cycle_he7
hours_selfcalib_cost = np.clip(hours_selfcalib - badhours_remain, 0, 24)
hours_obsfield = 24 - badhours - hours_selfcalib_cost
hours_ratio = hours_obsfield/24
fracmax_chorillos = (12-badmonths)*1./12 * hours_ratio

#eff = np.array([0.3,0.5, 1.])
eff = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
alleff=linspace(0.1, 1, 100)
clf()
xlabel('Observation Efficiency (%)')
ylabel('95% Upper-limit on $r$ (2 years)')
title('QUBIC 150 + 220 GHz + Planck 353 GHz')
xlim(0.1,1.)
ylim(0,0.05)
plot(eff, all_ul_D[0,:], 'b-', lw=3, label = 'Chorillos')
plot(eff, all_ul_D[1,:], 'r-', lw=3, label = 'Concordia')

fill_between([fracmax_chorillos[0],1.], [0,0], y2=[0.1,0.1], alpha=0.2, color='blue', hatch='/', label='Excluded')
fill_between([fracmax_concordia[0],1.], [0,0], y2=[0.1,0.1], alpha=0.2, color='red',hatch='//', label='Excluded')

ls = [':','--','-.']
for i in xrange(len(ls)): plot([100,100],[100,100],ls=ls[i], label = 'Self-Calib = {0:.0f}h'.format(hours_selfcalib[i]), color='k')

for i in xrange(len(fracmax_concordia)):
    f = fracmax_concordia[i]
    plot([f, f], [0,1], 'r', ls=ls[i])
    plot([0,1], np.array([0,0])+np.interp(f, eff, all_ul_D[1,:]), 'r', ls=ls[i])
    plot(f, np.interp(f, eff, all_ul_D[1,:]),'ro',ms=10)
    #annotate('Self-Calib = {0:.0f}h'.format(np.int(hours_selfcalib[i])), xy=(0.11, np.interp(f, eff, all_ul_D[1,:])+0.0003))

for i in xrange(len(fracmax_chorillos)):
    f = fracmax_chorillos[i]
    plot([f, f], [0,1], 'b', ls=ls[i])
    plot([0,1], np.array([0,0])+np.interp(f, eff, all_ul_D[0,:]), 'b',ls=ls[i])
    plot(f, np.interp(f, eff, all_ul_D[0,:]),'bo',ms=10)
    #annotate('Self-Calib = {0:.0f}h'.format(np.int(hours_selfcalib[i])), xy=(0.11, np.interp(f, eff, all_ul_D[0,:])+0.0003))
legend()

#savefig('sensitivity_efficiency.png')



