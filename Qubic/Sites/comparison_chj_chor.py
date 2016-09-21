#### from file data_llama.py

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




############################# This one leads to absurd results ###################################
### values for Chajnantor: 
# pwv taken from plot 9 in https://almascience.nrao.edu/about-alma/weather
# converted to zenithal transmission 210 GHz using calculator from https://almascience.nrao.edu/about-alma/atmosphere-model
pwv_chajnantor = np.array([4.5, 5.6, 3.2, 1.5, 1., 1.2, 0.6, 0.5, 0.6, 0.7, 1., 1.5])
clf()
ylim(0,4)
plot(month, pwv_dc,label='Dome C')
plot(month, pwv_chajnantor, label='Chajnantor')
ylabel('PWV')
xlabel('Month')
legend()
grid()


clf()
xlim(0,13)
xlabel('Month')
ylabel(r'PWV [mm]')
plot(month, pwv_dc, 'k--', lw=3, label ='Concordia')
plot(month, pwv_chajnantor, 'k:', lw=3, label ='Chajnantor')
plot(month, 14.425 * alleqtau, 'k', lw=3, label ='Chorrillos')
ylim(0,5)
legend()
grid()



#### seems absurd: better than DC !
transm210_chajnantor = np.array([0.83, 0.78, 0.86, 0.94, 0.95, 0.94, 0.97, 0.97, 0.97, 0.96, 0.95, 0.93])
tau210_chajnantor = -np.log(transm210_chajnantor)
tau220_chaj = tau210_chajnantor * conv_210_220
tau150_chaj = tau210_chajnantor * conv_210_150



#tau150_chaj = (Q + M*pwv_chajnantor)/TAvchorillo
#tau220_chaj = tau150_chaj / conv_210_150 * conv_210_220
#tau210_chaj = tau150_chaj / conv_210_150

clf()
plot(month, alleqtau,label='Chorillos')
plot(month, tau210_chajnantor, label='Chajnantor')
ylabel('Tau 210 GHz')
xlabel('Month')
legend()
grid()

clf()
#plot(month, alleqtau * conv_210_220, 'b', lw=3, label ='Chorillos - 220 GHz')
#plot(month, alleqtau * conv_210_150, 'r', lw=3, label ='Chorillos - 150 GHz')
plot(month, tau220_chaj, 'b:', lw=3, label ='Chajnantor - 220 GHz')
plot(month, tau150_chaj, 'r:', lw=3, label ='Chajnantor - 150 GHz')
plot(month, tau_dc_220, 'b--', lw=3, label ='Concordia - 220 GHz')
plot(month, tau_dc_150, 'r--', lw=3, label ='Concordia - 150 GHz')
ylabel('Tau 210 GHz')
xlabel('Month')
legend()
grid()
savefig('weird.png')
### Chajnantor serait aussi bon que Concordia pour 5 mois de l'année !!! Difficile à croire...
####################################################################################################






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
plot(month, tau_857_chaj * conv_857_150, 'r:', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 150 GHz')
plot(month, tau_857_chaj * conv_857_220, 'b:', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 220 GHz')
plot(month, tau_dc_220, 'b--', lw=3, label ='Concordia - 220 GHz')
plot(month, tau_dc_150, 'r--', lw=3, label ='Concordia - 150 GHz')
ylim(0,0.3)
legend(fontsize=10)
#savefig('compare_tau_chaj_cho_dc_from857.png')

### tau plot again
clf()
xlim(0,13)
xlabel('Month')
ylabel(r'Atmospheric Opacity $\tau$')
plot(month, alleqtau * conv_210_220, 'b', lw=3, label ='Chorillos - 220 GHz')
fill_between(month, alleqtau*(opacity210_75/opacity210_50) * conv_210_220, y2=alleqtau*(opacity210_25/opacity210_50) * conv_210_220, color='b', alpha=0.1)
plot(month, alleqtau * conv_210_150, 'r', lw=3, label ='Chorillos - 150 GHz')
fill_between(month, alleqtau*(opacity210_75/opacity210_50) * conv_210_150, y2=alleqtau*(opacity210_25/opacity210_50) * conv_210_150, color='r', alpha=0.1)
plot(month, tau_857_chaj * conv_857_150, 'r--', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 150 GHz')
plot(month, tau_857_chaj * conv_857_220, 'b--', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 220 GHz')
ylim(0,0.3)
legend(fontsize=10)
#savefig('compare_tau_chaj_cho.png')

### First get emissivities from tau
em150_chaj = 1-np.exp(-tau_857_chaj * conv_857_150/np.cos(np.radians(90-elevation_obs)))
em220_chaj = 1-np.exp(-tau_857_chaj * conv_857_220/np.cos(np.radians(90-elevation_obs)))
clf()
xlim(0,13)
xlabel('Month')
ylabel(r'Atmospheric Emissivity')
plot(month, em220_ch, 'b', lw=3, label ='Chorillos - 220 GHz')
plot(month, em150_ch, 'r', lw=3, label ='Chorillos - 150 GHz')
plot(month, em220_dc, 'b--', lw=3, label ='Concordia - 220 GHz')
plot(month, em150_dc, 'r--', lw=3, label ='Concordia - 150 GHz')
plot(month, em150_chaj, 'r:', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 150 GHz')
plot(month, em220_chaj, 'b:', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 220 GHz')
ylim(0,0.3)
legend(frameon=False)
#savefig('atm_emissivity.png')


### First get emissivities from tau
clf()
xlim(0,13)
xlabel('Month')
ylabel(r'Brightness Temperature')
plot(month, em220_ch * TAvchorillo, 'b', lw=3, label ='Chorillos - 220 GHz')
plot(month, em150_ch * TAvchorillo, 'r', lw=3, label ='Chorillos - 150 GHz')
plot(month, em220_dc * Tdomec, 'b--', lw=3, label ='Concordia - 220 GHz')
plot(month, em150_dc * Tdomec, 'r--', lw=3, label ='Concordia - 150 GHz')
plot(month, em150_chaj * TAvchorillo, 'r:', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 150 GHz')
plot(month, em220_chaj * TAvchorillo, 'b:', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 220 GHz')
ylim(0,0.3)
legend(frameon=False)
#savefig('atm_emissivity.png')

### not good in the bolivian summer... too optimistic as coming from 1998 data - before it became agressive
####################################################################################################






####################################################################################################
### data from alma memo 512 data at 225 GHz
def func(a):
    return(np.array([np.median(a), np.std(a)]))
    
jan = func(np.array([0.08, 0.14, 0.18, 0.24, 0.26, 0.45]))
feb = func(np.array([0.1, 0.105, 0.12, 0.13, 0.27, 0.29, 0.33]))
mar = func(np.array([0.08, 0.09, 0.095, 0.11, 0.13, 0.21, 0.28, 0.33]))
apr = func(np.array([0.04, 0.05, 0.065, 0.09, 0.1, 0.15]))
may = func(np.array([0.04, 0.05, 0.06, 0.08, 0.1]))
jun = func(np.array([0.035, 0.04, 0.05, 0.06, 0.08,0.09]))
jul = func(np.array([0.03, 0.04, 0.05, 0.06, 0.08, 0.085]))
aug = func(np.array([0.04, 0.045, 0.05, 0.6, 0.11]))
sep = func(np.array([0.03, 0.045, 0.055, 0.06, 0.12]))
octo = func(np.array([0.03, 0.045, 0.055, 0.06, 0.06]))
nov = func(np.array([0.04, 0.05, 0.06, 0.065, 0.09]))
dec = func(np.array([0.04, 0.06, 0.075, 0.08, 0.085, 0.095]))

data = np.array([jan, feb, mar, apr, may, jun, jul, aug, sep, octo, nov, dec])

conv_225_150 = np.interp(150, fghz, tau) / np.interp(225, fghz, tau)
conv_225_220 = np.interp(220, fghz, tau) / np.interp(225, fghz, tau)

### tau plot again
clf()
xlim(0,13)
xlabel('Month')
ylabel(r'Atmospheric Opacity $\tau$')
plot(month, alleqtau * conv_210_220, 'b', lw=3, label ='Chorillos - 220 GHz')
plot(month, alleqtau * conv_210_150, 'r', lw=3, label ='Chorillos - 150 GHz')
plot(month, data[:,0] * conv_225_150, 'r:', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 150 GHz')
plot(month, data[:,0] * conv_225_220, 'b:', lw=3, label ='Chajnantor (1998 ext from 857 GHz) - 220 GHz')
plot(month, tau_dc_220, 'b--', lw=3, label ='Concordia - 220 GHz')
plot(month, tau_dc_150, 'r--', lw=3, label ='Concordia - 150 GHz')
ylim(0,0.3)
legend(fontsize=10)
savefig('compare_tau_chaj_cho_dc_fromAlma512.png')


####################################################################################################





em150_chaj = 1-np.exp(-tau_857_chaj * conv_857_150/np.cos(np.radians(90-elevation_obs)))
em220_chaj = 1-np.exp(-tau_857_chaj * conv_857_220/np.cos(np.radians(90-elevation_obs)))
clf()
xlim(0,13)
xlabel('Month')
ylabel(r'Atmospheric Emissivity')
plot(month, em220_ch, 'b', lw=3, label ='Chorillos - 220 GHz')
plot(month, em150_ch, 'r', lw=3, label ='Chorillos - 150 GHz')
plot(month, em220_dc, 'b--', lw=3, label ='Concordia - 220 GHz')
plot(month, em150_dc, 'r--', lw=3, label ='Concordia - 150 GHz')
plot(month, em220_chaj, 'b:', lw=3, label ='Chajnantor - 220 GHz')
plot(month, em150_chaj, 'r:', lw=3, label ='Chajnantor - 150 GHz')
ylim(0,0.3)
legend()
#savefig('atm_emissivity.png')


