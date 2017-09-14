
#### now read simulations made with these numbers
####### Chains for ANR 2015 simulations
def upperlimit(chain,key,level=0.95):
	sorteddata = np.sort(chain[key])
	return sorteddata[level*len(sorteddata)]

from McMc import mcmc
from Sensitivity import data4mcmc
# Simulaitons for QUBIC with sensitivities calculated by JCH
#rep = '/Users/hamilton/Qubic/Sites/SimsComparisonSites/'
# Simulaitons for QUBIC with sensitivities calculated by Elia
rep = '/Users/hamilton/Qubic/Sites/SimsComparisonSites_Elia/'


all_ul_B = []
all_ul_D = []
all_ul_nofg = []

#for localisation in ['atac', 'conc']:
for localisation in ['atac']:
    for eff in ['03']:
    #for eff in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '1']:
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
        histn=100
        alpha =0.5

        nbins=100
        from scipy.ndimage import gaussian_filter1d
        bla = np.histogram(chain_nofg_r['r'],bins=nbins,normed=True)
        xhist=(bla[1][0:nbins]+bla[1][1:nbins+1])/2
        ss=np.std(chain_nofg_r['r'])
        yhist=gaussian_filter1d(bla[0],20*ss/histn/(xhist[1]-xhist[0]), mode='nearest')
        plot(xhist,yhist/max(yhist))

        thelimits = [[truebeta*0.98, truebeta*1.05],[0,0.03]]

        bla=mcmc.matrixplot(chain_B_r_dl_b,['betadust','r'], 'green', sm, 
            limits=thelimits, alpha=alpha,nbins=histn)#, truevals = [truebeta, truer])

        ### Au final
        clf()
        #c=mcmc.matrixplot(chain_C_r_dl_b,['betadust','r'], 'black', sm, limits=[[truebeta*0.95, truebeta*1.05],[0,0.05]], alpha=alpha,histn=histn, truevals = [truebeta, truer])
        b=mcmc.matrixplot(chain_B_r_dl_b,['betadust','r'], 'blue', sm, limits=thelimits, 
            alpha=alpha,nbins=histn)#, truevals = [truebeta, truer])
        d=mcmc.matrixplot(chain_D_r_dl_b,['betadust','r'], 'red', sm, limits=thelimits, 
            alpha=alpha,nbins=histn)#, truevals = [truebeta, truer])
        subplot(2,2,4)
        noFG = plot(xhist,yhist/max(yhist), color='green', label='toto')
        #subplot(2,2,2)
        #legC = '150x2+353 : r < {0:5.2f} (95% CL)'.format(upperlimit(chain_C_r_dl_b,'r'))
        ul_B = upperlimit(chain_B_r_dl_b,'r', level =0.68)
        legB = '150+220 : $\sigma_r$ = {0:5.3f}'.format(ul_B)
        ul_D = upperlimit(chain_D_r_dl_b,'r', level =0.68)
        legD = '150+220+353: $\sigma_r$ = {0:5.3f}'.format(ul_D)
        ul_nofg = upperlimit(chain_nofg_r,'r', level =0.68)
        legnoFG = 'No Foregrounds: $\sigma_r$ = {0:5.3f}'.format(ul_nofg)
        legend([b, d, bla],[legB, legD, legnoFG], frameon=False, title='QUBIC 2 years '+config+site, fontsize=12)
        #legend([b, d, bla],[legB, legD, legnoFG], frameon=False, title='QUBIC 2 years - Concordia - $\epsilon$=0.3')
        #savefig(site+'.pdf')
        draw()
        
        all_ul_B. append(ul_B)
        all_ul_D. append(ul_D)
        all_ul_nofg. append(ul_nofg)







#for localisation in ['atac', 'conc']:
for localisation in ['atac']:
    for eff in ['03']:
    #for eff in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '1']:
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
        histn=100
        alpha =0.5

        nbins=100
        from scipy.ndimage import gaussian_filter1d
        bla = np.histogram(chain_nofg_r['r'],bins=nbins,normed=True)
        xhist=(bla[1][0:nbins]+bla[1][1:nbins+1])/2
        ss=np.std(chain_nofg_r['r'])
        yhist=gaussian_filter1d(bla[0],20*ss/histn/(xhist[1]-xhist[0]), mode='nearest')
        plot(xhist,yhist/max(yhist))

        thelimits = [[truebeta*0.98, truebeta*1.05],[0,0.03]]

        bla=mcmc.matrixplot(chain_B_r_dl_b,['betadust','r'], 'green', sm, 
            limits=thelimits, alpha=alpha,nbins=histn, leg=False)#, truevals = [truebeta, truer])

        ### Au final
        clf()
        #c=mcmc.matrixplot(chain_C_r_dl_b,['betadust','r'], 'black', sm, limits=[[truebeta*0.95, truebeta*1.05],[0,0.05]], alpha=alpha,histn=histn, truevals = [truebeta, truer])
        #b=mcmc.matrixplot(chain_B_r_dl_b,['betadust','r'], 'blue', sm, limits=thelimits, 
        #    alpha=alpha,nbins=histn, linestyle=':')#, truevals = [truebeta, truer])
        d=mcmc.matrixplot(chain_D_r_dl_b,['betadust','r'], 'red', sm, limits=thelimits, 
            alpha=alpha,nbins=histn, linestyle='-', leg=False)#, truevals = [truebeta, truer])
        subplot(2,2,4)
        #subplot(2,2,2)
        #legC = '150x2+353 : r < {0:5.2f} (95% CL)'.format(upperlimit(chain_C_r_dl_b,'r'))
        #ul_B = upperlimit(chain_B_r_dl_b,'r', level =0.68)
        #legB = '150+220 : $\sigma_r$ = {0:5.3f}'.format(ul_B)
        ul_D = upperlimit(chain_D_r_dl_b,'r', level =0.68)
        legD = '150+220+353: $\sigma_r$ = {0:5.3f}'.format(ul_D)
        ul_nofg = upperlimit(chain_nofg_r,'r', level =0.68)
        legnoFG = 'No Foregrounds: $\sigma_r$ = {0:5.3f}'.format(ul_nofg)
        dd=plot([-1,-1],[-1,-1], color='red', label=legD, lw=2)
        ee=plot(xhist,yhist/max(yhist), color='green', label=legnoFG, linestyle='--', lw=2)
        legend(frameon=False, title='QUBIC 2 years \n Argentina, 30% efficiency\n ', fontsize=10)
        #legend([d, noFG],[legD, legnoFG], frameon=False, title='QUBIC 2 years \n Argentina, 30% efficiency\n ', fontsize=12)
        #legend([b, d, bla],[legB, legD, legnoFG], frameon=False, title='QUBIC 2 years '+config+site, fontsize=12)
        #legend([b, d, bla],[legB, legD, legnoFG], frameon=False, title='QUBIC 2 years - Concordia - $\epsilon$=0.3')
        #savefig(site+'.pdf')
        draw()

savefig('newplot_argentina.pdf')


















all_ul_B = np.reshape(all_ul_B, (2,len(all_ul_B)/2))
all_ul_D = np.reshape(all_ul_D, (2,len(all_ul_D)/2))
all_ul_nofg = np.reshape(all_ul_nofg, (2,len(all_ul_nofg)/2))

#eff = np.array([0.3,0.5, 1.])
eff = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
alleff=linspace(0.1, 1, 100)
clf()
grid()
xlim(0.1,1.)
ylim(0,0.03)
xlabel('Observation Efficiency (%)')
ylabel('$\sigma_r$')
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
ylabel('$\sigma_r$')
title('QUBIC 150 + 220 GHz (2 Years) + Planck 353 GHz')
xlim(0.1,1.)
ylim(0,0.03)
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

savefig('sensitivity_efficiency_final.pdf',format='pdf')











#eff = np.array([0.3,0.5, 1.])
eff = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
alleff=linspace(0.1, 1, 100)
clf()
grid()
xlim(0.1,1.)
ylim(0.004,0.011)
xlabel('Observation Efficiency (%)')
ylabel('$1\sigma$ Upper-limit on $r$ (2 years)')
title('QUBIC 150 + 220 GHz + Planck 353 GHz')
plot(eff, eff*1000, 'r', lw=3)
plot(eff, eff*1000, 'b-', lw=3, label = 'Concordia')
plot(eff, eff*1000, 'r-', lw=3, label = 'Chorillos')
plot(eff, all_ul_D[0,:], 'r', lw=3)
plot(eff, all_ul_D[1,:], 'b', lw=3)
legend()
#savefig('sensitivity_efficiency.png')

