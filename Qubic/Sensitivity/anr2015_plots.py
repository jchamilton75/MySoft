from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import pi
import pycamb
from qubic import QubicInstrument
from scipy.constants import c
from Sensitivity import qubic_sensitivity
from Homogeneity import SplineFitting
from pysimulators import FitsArray
from Cosmo import interpol_camb as ic
from Sensitivity import dualband_lib as db
from McMc import mcmc
import pymc
from Sensitivity import data4mcmc



def upperlimit(chain,key,level=0.95):
	sorteddata = np.sort(chain[key])
	return sorteddata[level*len(sorteddata)]

truer = 0.
truebeta = 1.59
truedl = 13.4 * 0.45
truealpha = -2.42
trueT = 19.6
level =0.95
cl = int(level*100)



####### Chains for ANR 2015 simulations
rep = '/Users/hamilton/CMB/Interfero/DualBand/SimsANR2015/'
site = 'Concordia'
config = ' $(\epsilon=1)$ '
chain_A_r_dl_b = data4mcmc.readchains(rep+'instrumentA_r_dl_b.db')
chain_B_r_dl_b = data4mcmc.readchains(rep+'instrumentB_r_dl_b.db')
chain_C_r_dl_b = data4mcmc.readchains(rep+'instrumentC_r_dl_b.db')
chain_D_r_dl_b = data4mcmc.readchains(rep+'instrumentD_r_dl_b.db')
chain_nofg_r = data4mcmc.readchains(rep+'instrumentNofg_r.db')

########### r dl and beta
figure(0)
sm=3
histn=3
alpha =0.5

nbins=100
from scipy.ndimage import gaussian_filter1d
bla = np.histogram(chain_nofg_r['r'],bins=nbins,normed=True)
xhist=(bla[1][0:nbins]+bla[1][1:nbins+1])/2
ss=np.std(chain_nofg_r['r'])
yhist=gaussian_filter1d(bla[0],ss/histn/(xhist[1]-xhist[0]), mode='nearest')
plot(xhist,yhist/max(yhist))

bla=mcmc.matrixplot(chain_B_r_dl_b,['betadust','r'], 'green', sm, limits=[[truebeta*0.95, truebeta*1.05],[0,0.05]], alpha=alpha,histn=histn, truevals = [truebeta, truer])

### Au final
clf()
#c=mcmc.matrixplot(chain_C_r_dl_b,['betadust','r'], 'black', sm, limits=[[truebeta*0.95, truebeta*1.05],[0,0.05]], alpha=alpha,histn=histn, truevals = [truebeta, truer])
b=mcmc.matrixplot(chain_B_r_dl_b,['betadust','r'], 'blue', sm, limits=[[truebeta*0.95, truebeta*1.1],[0,0.07]], alpha=alpha,histn=histn, truevals = [truebeta, truer])
d=mcmc.matrixplot(chain_D_r_dl_b,['betadust','r'], 'red', sm, limits=[[truebeta*0.95, truebeta*1.1],[0,0.07]], alpha=alpha,histn=histn, truevals = [truebeta, truer])
subplot(2,2,4)
noFG = plot(xhist,yhist/max(yhist), color='green', label='toto')
subplot(2,2,2)
#legC = '150x2+353 : r < {0:5.2f} (95% CL)'.format(upperlimit(chain_C_r_dl_b,'r'))
legB = '150+220 : r < {0:5.2f} (95% CL)'.format(upperlimit(chain_B_r_dl_b,'r'))
legD = '150+220+353: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_D_r_dl_b,'r'))
legnoFG = 'No Foregrounds: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_nofg_r,'r'))
legend([b, d, bla],[legB, legD, legnoFG], frameon=False, title='QUBIC 2 years '+config+site)

savefig('limits_anr2015.png', transparent=True)



figure(1)

rep = '/Users/hamilton/CMB/Interfero/DualBand/SimsANR2015/Sims_eps0.3/'
site = 'Concordia'
config = ' $(\epsilon=0.3)$ '
chain_A_r_dl_b = data4mcmc.readchains(rep+'instrumentA_r_dl_b.db')
chain_B_r_dl_b = data4mcmc.readchains(rep+'instrumentB_r_dl_b.db')
chain_C_r_dl_b = data4mcmc.readchains(rep+'instrumentC_r_dl_b.db')
chain_D_r_dl_b = data4mcmc.readchains(rep+'instrumentD_r_dl_b.db')
chain_nofg_r = data4mcmc.readchains(rep+'instrumentNofg_r.db')


########### r dl and beta

sm=3
histn=3
alpha =0.5

nbins=100
from scipy.ndimage import gaussian_filter1d
bla = np.histogram(chain_nofg_r['r'],bins=nbins,normed=True)
xhist=(bla[1][0:nbins]+bla[1][1:nbins+1])/2
ss=np.std(chain_nofg_r['r'])
yhist=gaussian_filter1d(bla[0],ss/histn/(xhist[1]-xhist[0]), mode='nearest')
plot(xhist,yhist/max(yhist))

bla=mcmc.matrixplot(chain_B_r_dl_b,['betadust','r'], 'green', sm, limits=[[truebeta*0.95, truebeta*1.05],[0,0.05]], alpha=alpha,histn=histn, truevals = [truebeta, truer])

### Au final
clf()
#c=mcmc.matrixplot(chain_C_r_dl_b,['betadust','r'], 'black', sm, limits=[[truebeta*0.95, truebeta*1.05],[0,0.05]], alpha=alpha,histn=histn, truevals = [truebeta, truer])
b=mcmc.matrixplot(chain_B_r_dl_b,['betadust','r'], 'blue', sm, limits=[[truebeta*0.95, truebeta*1.1],[0,0.07]], alpha=alpha,histn=histn, truevals = [truebeta, truer])
d=mcmc.matrixplot(chain_D_r_dl_b,['betadust','r'], 'red', sm, limits=[[truebeta*0.95, truebeta*1.1],[0,0.07]], alpha=alpha,histn=histn, truevals = [truebeta, truer])
subplot(2,2,4)
noFG = plot(xhist,yhist/max(yhist), color='green', label='toto')
subplot(2,2,2)
#legC = '150x2+353 : r < {0:5.2f} (95% CL)'.format(upperlimit(chain_C_r_dl_b,'r'))
legB = '150+220 : r < {0:5.2f} (95% CL)'.format(upperlimit(chain_B_r_dl_b,'r'))
legD = '150+220+353: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_D_r_dl_b,'r'))
legnoFG = 'No Foregrounds: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_nofg_r,'r'))
legend([b, d, bla],[legB, legD, legnoFG], frameon=False, title='QUBIC 2 years '+config+site)
savefig('limits_anr2015_eps_0.3.png', transparent=True)






####### Other plot
rep = '/Users/hamilton/CMB/Interfero/DualBand/SimsANR2015/'
site = 'Concordia'
config = ' '
chain_A_r_dl_b = data4mcmc.readchains('/Users/hamilton/CMB/Interfero/DualBand/SimsANR2015/'+'instrumentA_r_dl_b.db')
chain_B_r_dl_b = data4mcmc.readchains('/Users/hamilton/CMB/Interfero/DualBand/SimsANR2015/'+'instrumentB_r_dl_b.db')
chain_B03_r_dl_b = data4mcmc.readchains('/Users/hamilton/CMB/Interfero/DualBand/SimsANR2015/Sims_eps0.3/'+'instrumentB_r_dl_b.db')
chain_nofg_r = data4mcmc.readchains('/Users/hamilton/CMB/Interfero/DualBand/SimsANR2015/'+'instrumentNofg_r.db')
chain_D03_r_dl_b = data4mcmc.readchains('/Users/hamilton/CMB/Interfero/DualBand/SimsANR2015/Sims_eps0.3/'+'instrumentD_r_dl_b.db')

sm=3
histn=3
alpha =0.5

nbins=100
from scipy.ndimage import gaussian_filter1d
bla = np.histogram(chain_nofg_r['r'],bins=nbins,normed=True)
xhist=(bla[1][0:nbins]+bla[1][1:nbins+1])/2
ss=np.std(chain_nofg_r['r'])
yhist=gaussian_filter1d(bla[0],ss/histn/(xhist[1]-xhist[0]), mode='nearest')
plot(xhist,yhist/max(yhist))

bla=mcmc.matrixplot(chain_B_r_dl_b,['betadust','r'], 'black', sm, limits=[[truebeta*0.95, truebeta*1.05],[0,0.05]], alpha=alpha,histn=histn, truevals = [truebeta, truer])

clf()
a=mcmc.matrixplot(chain_A_r_dl_b,['betadust','r'], 'brown', sm, limits=[[truebeta*0.95, truebeta*1.1],[0,0.07]], alpha=alpha,histn=histn, truevals = [truebeta, truer])
b03=mcmc.matrixplot(chain_B03_r_dl_b,['betadust','r'], 'blue', sm, limits=[[truebeta*0.95, truebeta*1.1],[0,0.07]], alpha=alpha,histn=histn, truevals = [truebeta, truer])
d03=mcmc.matrixplot(chain_D03_r_dl_b,['betadust','r'], 'red', sm, limits=[[truebeta*0.95, truebeta*1.1],[0,0.07]], alpha=alpha,histn=histn, truevals = [truebeta, truer])
#b=mcmc.matrixplot(chain_B_r_dl_b,['betadust','r'], 'green', sm, limits=[[truebeta*0.95, truebeta*1.1],[0,0.07]], alpha=alpha,histn=histn, truevals = [truebeta, truer])
subplot(2,2,4)
noFG = plot(xhist,yhist/max(yhist), color='black', label='toto')
subplot(2,2,2)

legA = '150 Ghz only, $\epsilon=1$, FG: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_A_r_dl_b,'r'))
#legB = '150/220 GHz, $\epsilon=1$, FG: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_B_r_dl_b,'r'))
legB03 = '150/220 GHz, $\epsilon=0.3$, FG: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_B03_r_dl_b,'r'))
legD03 = '150/220 GHz+Planck, $\epsilon=0.3$, FG: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_D03_r_dl_b,'r'))
legnoFG = '$\epsilon=1$, No FG: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_nofg_r,'r'))
legend([bla, a, b03, d03],[legnoFG, legA, legB03, legD03], frameon=False, title='QUBIC 2 years '+config+site,fontsize=13)
savefig('all_limits_qubic.png', transparent=False)





sm=3
histn=3
alpha =0.5

nbins=100
from scipy.ndimage import gaussian_filter1d
bla = np.histogram(chain_nofg_r['r'],bins=nbins,normed=True)
xhist=(bla[1][0:nbins]+bla[1][1:nbins+1])/2
ss=np.std(chain_nofg_r['r'])
yhist=gaussian_filter1d(bla[0],ss/histn/(xhist[1]-xhist[0]), mode='nearest')
plot(xhist,yhist/max(yhist))

bla=mcmc.matrixplot(chain_B_r_dl_b,['dldust_80_353','betadust','r'], 'black', sm, limits=[[truedl*0.95, truedl*1.1],[truebeta*0.95, truebeta*1.05],[0,0.05]], alpha=alpha,histn=histn, truevals = [truedl, truebeta, truer])

clf()
a=mcmc.matrixplot(chain_A_r_dl_b,['dldust_80_353','betadust','r'], 'brown', sm, limits=[[truedl*0.95, truedl*1.1],[truebeta*0.95, truebeta*1.1],[0,0.07]], alpha=alpha,histn=histn, truevals = [truedl, truebeta, truer])
b03=mcmc.matrixplot(chain_B03_r_dl_b,['dldust_80_353','betadust','r'], 'blue', sm, limits=[[truedl*0.95, truedl*1.1],[truebeta*0.95, truebeta*1.1],[0,0.07]], alpha=alpha,histn=histn, truevals = [truedl, truebeta, truer])
d03=mcmc.matrixplot(chain_D03_r_dl_b,['dldust_80_353','betadust','r'], 'red', sm, limits=[[truedl*0.95, truedl*1.1],[truebeta*0.95, truebeta*1.1],[0,0.07]], alpha=alpha,histn=histn, truevals = [truedl, truebeta, truer])
#b=mcmc.matrixplot(chain_B_r_dl_b,['dldust_80_353','betadust','r'], 'green', sm, limits=[[truedl*0.95, truedl*1.1],[truebeta*0.95, truebeta*1.1],[0,0.07]], alpha=alpha,histn=histn, truevals = [truedl, truebeta, truer])
subplot(3,3,9)
noFG = plot(xhist,yhist/max(yhist), color='black', label='toto')
subplot(3,3,3)

legA = '150 Ghz only, $\epsilon=1$, FG: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_A_r_dl_b,'r'))
#legB = '150/220 GHz, $\epsilon=1$, FG: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_B_r_dl_b,'r'))
legB03 = '150/220 GHz, $\epsilon=0.3$, FG: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_B03_r_dl_b,'r'))
legD03 = '150/220 GHz+Planck, $\epsilon=0.3$, FG: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_D03_r_dl_b,'r'))
legnoFG = '$\epsilon=1$, No FG: r < {0:5.2f} (95% CL)'.format(upperlimit(chain_nofg_r,'r'))
legend([bla, a, b03, d03],[legnoFG, legA, legB03, legD03], frameon=False, title='QUBIC 2 years '+config+site,fontsize=13)
savefig('all_limits_qubic_dl_beta_r.png', transparent=False)











############ Lensing floor ?
dl_lensing = ic.get_Dlbb_fromlib(lll, 0, camblib)
fsky=0.8
deltal=100
nsig=2
svar_lensing = np.sqrt(2./(2*lll+1)/fsky/deltal)*dl_lensing*nsig
svar_lensing_5percent = np.sqrt(2./(2*lll+1)/0.05/deltal)*dl_lensing*nsig
svar_lensing_1percent = np.sqrt(2./(2*lll+1)/0.01/deltal)*dl_lensing*nsig

dl_01 = ic.get_Dlbb_fromlib(lll, 0.01, camblib)-dl_lensing
dl_001 = ic.get_Dlbb_fromlib(lll, 0.001, camblib)-dl_lensing
dl_005 = ic.get_Dlbb_fromlib(lll, 0.005, camblib)-dl_lensing

clf()
xlim(0,150)
yscale('log')
ylim(1e-6,0.02)
plot(lll, dl_lensing, 'k',lw=2,label ='Lensing Dl')
plot(lll, svar_lensing, 'k:',lw=2,label ='2 sig sample variance on lensing deltal=100, fsky=0.8')
plot(lll, svar_lensing_5percent, 'k--',lw=2,label ='2 sig sample variance on lensing deltal=100, fsky=0.05')
plot(lll, svar_lensing_1percent, 'k-.',lw=2,label ='2 sig sample variance on lensing deltal=100, fsky=0.01')
plot(lll, dl_01, 'r',lw=2,label ='Primordial r=0.01')
plot(lll, dl_005, 'm',lw=2,label ='Primordial r=0.005')
plot(lll, dl_001, 'g',lw=2,label ='Primordial r=0.001')
legend(fontsize=10)








