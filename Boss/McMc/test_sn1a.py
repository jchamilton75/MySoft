import sys
import pymc
from pymc import Metropolis
import cosmolopy
from McMc import mcmc
from astropy.io import fits
from McMc import cosmo_utils
import scipy
import pickle

xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm planck &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm planck+BAO &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm planck+BAO+hRiess &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm planck+BAO+LyaDR11+hRiess &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm planckHp &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm planckHp+BAO &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm planckHp+BAO+hRiess &

############################ Flat w ###################################
model='flatwcdm'
rep='/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/'
ext='.db'
planck=mcmc.readchains(rep+model+'-'+'planck'+ext)
planck_bao=mcmc.readchains(rep+model+'-'+'planck+BAO'+ext)
planck_lya=mcmc.readchains(rep+model+'-'+'planck+LyaDR11'+ext)
planck_bao_lya=mcmc.readchains(rep+model+'-'+'planck+BAO+LyaDR11'+ext)
planck_bao_hriess=mcmc.readchains(rep+model+'-'+'planck+BAO+hRiess'+ext)
planck_bao_lya_hriess=mcmc.readchains(rep+model+'-'+'planck+BAO+LyaDR11+hRiess'+ext)

planckHp=mcmc.readchains(rep+model+'-'+'planckHp'+ext)
planckHp_bao=mcmc.readchains(rep+model+'-'+'planckHp+BAO'+ext)
planckHp_bao_hriess=mcmc.readchains(rep+model+'-'+'planckHp+BAO+hRiess'+ext)
###################################################

clf()
sm=6
xlim(0.,0.6)
ylim(-2.,-0.3)
alpha=0.5
a0=mcmc.cont(planck['omega_M_0'],planck['w'],color='green',nsmooth=sm,alpha=0.7)
a2=mcmc.cont(planck_bao['omega_M_0'],planck_bao['w'],color='blue',nsmooth=sm,alpha=0.7)
a3=mcmc.cont(planck_bao_hriess['omega_M_0'],planck_bao_hriess['w'],color='red',nsmooth=sm,alpha=alpha)
x=linspace(0,1,100)
plot(x,x*0-1,'k:')
xlabel('$\Omega_m$')
ylabel('$w$')
legend([a0,a2,a3],['Planck','Planck+BAO','Planck+BAO+H$_0$ (Riess)'],frameon=False,loc=2)
savefig('cmb_bao_h0.pdf')



clf()
sm=6
xlim(0.1,0.4)
ylim(-2.,-0.7)
alpha=1
a0=mcmc.cont(planck['omega_M_0'],planck['w'],color='green',nsmooth=sm,alpha=0.3)
a3=mcmc.cont(planck_bao_hriess['omega_M_0'],planck_bao_hriess['w'],color='red',nsmooth=sm,Fill=False,linewidths=2)
x=linspace(0,1,100)
plot(x,x*0-1,'k:')
xlabel('$\Omega_m$')
ylabel('$w$')
legend([a0,a3],['Planck','Planck+BAO+H$_0$ (Riess)'],frameon=False,loc=2)
grid()
savefig('cmb_bao_h0_zoom.pdf')

clf()
sm=6
xlim(0.1,0.4)
ylim(-2.,-0.7)
alpha=1
a0=mcmc.cont(planck['omega_M_0'],planck['w'],color='green',nsmooth=sm,alpha=0.3)
a2=mcmc.cont(planck_bao['omega_M_0'],planck_bao['w'],color='blue',nsmooth=sm,Fill=False,linewidths=2)
a3=mcmc.cont(planck_bao_hriess['omega_M_0'],planck_bao_hriess['w'],color='red',nsmooth=sm,Fill=False,linewidths=2)
x=linspace(0,1,100)
plot(x,x*0-1,'k:')
xlabel('$\Omega_m$')
ylabel('$w$')
legend([a0,a2,a3],['Planck','Planck+BAO','Planck+BAO+H$_0$ (Riess)'],frameon=False,loc=2)
grid()
savefig('cmb_bao_zoom.pdf')

def meansig(x):
    return str('%.3f'%np.mean(x))+' +/- '+str('%.3f'%np.std(x))

clf()
hist(planck_bao['h'],bins=30,alpha=0.5,label='Planck+BAO: '+meansig(planck_bao['h']))
a=np.random.randn(planck_bao['h'].size)*0.024+0.738
hist(a,bins=30,alpha=0.5,label='Riess: '+meansig(a),color='red')
xlabel('$h$')
legend(frameon=False,loc=2)
savefig('histos_h.pdf')

(mean(planck_bao['h'])-mean(a))/np.sqrt(std((planck_bao['h']))**2+std((a))**2)


clf()
sm=6
xlim(0.1,0.4)
ylim(-2.,-0.7)
alpha=1
a0=mcmc.cont(planck['omega_M_0'],planck['w'],color='green',nsmooth=sm,alpha=0.3)
#a1=mcmc.cont(planck_lya['omega_M_0'],planck_lya['w'],color='blue',nsmooth=sm,alpha=0.3)
a2=mcmc.cont(planck_bao['omega_M_0'],planck_bao['w'],color='blue',nsmooth=sm,Fill=False,linewidths=2)
a3=mcmc.cont(planck_bao_hriess['omega_M_0'],planck_bao_hriess['w'],color='red',nsmooth=sm,Fill=False,linewidths=2)
a4=mcmc.cont(planck_bao_lya['omega_M_0'],planck_bao_lya['w'],color='purple',nsmooth=sm,Fill=False,linewidths=2)
a5=mcmc.cont(planck_bao_lya_hriess['omega_M_0'],planck_bao_lya_hriess['w'],color='yellow',nsmooth=sm,Fill=False,linewidths=2)
x=linspace(0,1,100)
plot(x,x*0-1,'k:')
xlabel('$\Omega_m$')
ylabel('$w$')
grid()
legend([a0,a2,a3,a4,a5],['Planck','Planck+BAO','Planck+BAO+H$_0$ (Riess)',r'Planck+BAO+Lyman-$\alpha$',r'Planck+BAO+Lyman-$\alpha$+H$_0$ (Riess)'],frameon=False,loc=2)
savefig('cmb_bao_h0_zoom_lymanalpha.pdf')


######################
#with an additional prior on h planck+wp+lensing+high ell
clf()
sm=6
xlim(0.1,0.4)
ylim(-2.,-0.7)
alpha=1
a0=mcmc.cont(planckHp['omega_M_0'],planckHp['w'],color='green',nsmooth=sm,alpha=0.3)
a2=mcmc.cont(planckHp_bao['omega_M_0'],planckHp_bao['w'],color='blue',nsmooth=sm,Fill=False,linewidths=2)
a3=mcmc.cont(planckHp_bao_hriess['omega_M_0'],planckHp_bao_hriess['w'],color='red',nsmooth=sm,Fill=False,linewidths=2)
x=linspace(0,1,100)
plot(x,x*0-1,'k:')
xlabel('$\Omega_m$')
ylabel('$w$')
legend([a0,a2,a3],['PlanckHp','PlanckHp+BAO','PlanckHp+BAO+H$_0$ (Riess)'],frameon=False,loc=2)
grid()
savefig('cmb_bao_zoom_Hp.pdf')


clf()
hist(planckHp_bao['h'],bins=30,alpha=0.5,label='PlanckHp+BAO: '+meansig(planckHp_bao['h']))
a=np.random.randn(planckHp_bao['h'].size)*0.024+0.738
hist(a,bins=30,alpha=0.5,label='Riess: '+meansig(a),color='red')
xlabel('$h$')
legend(frameon=False,loc=2)
savefig('histos_h_Hp.pdf')

