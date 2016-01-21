from pylab import *
import sys
import pymc
from pymc import Metropolis
import cosmolopy
from McMc import mcmc
from astropy.io import fits
from McMc import cosmo_utils
import scipy
import pickle


#### run in a shell
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11_HPlanck1s_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11_HPlanck2s_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11_HRiess1s_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11_HRiess2s_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11_HRiessPlanck_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11_HPlanck1s_obh2Planck2s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11_HPlanck2s_obh2Planck2s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11_HRiess1s_obh2Planck2s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11_HRiess2s_obh2Planck2s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11_HRiessPlanck_obh2Planck2s &

xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11_HPlanck1s_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11_HPlanck2s_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11_HRiess1s_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11_HRiess2s_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11_HRiessPlanck_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11_HPlanck1s_obh2Planck2s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11_HPlanck2s_obh2Planck2s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11_HRiess1s_obh2Planck2s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11_HRiess2s_obh2Planck2s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11_HRiessPlanck_obh2Planck2s &

xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11_HPlanck1s_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11_HPlanck2s_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11_HRiess1s_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11_HRiess2s_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11_HRiessPlanck_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11_HPlanck1s_obh2Planck2s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11_HPlanck2s_obh2Planck2s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11_HRiess1s_obh2Planck2s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11_HRiess2s_obh2Planck2s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11_HRiessPlanck_obh2Planck2s &

#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11_BAO_HRiessPlanck_obh2Planck1s &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11_BAO_HRiessPlanck_obh2Planck1s &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11_BAO_HRiessPlanck_obh2Planck1s &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm BAO_HRiessPlanck_obh2Planck1s &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm BAO_HRiessPlanck_obh2Planck1s &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm BAO_HRiessPlanck_obh2Planck1s &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11+BAO &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11+BAO &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11+BAO &

xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm_fixed_h_obh2 LyaDR11 &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm_fixed_h_obh2 LyaDR11 &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm_fixed_h_obh2 LyaDR11 &


rep='/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/'
ext='.db'

######################## Ok ################################
model='olambdacdm'
lya_hp1_obh2p1=mcmc.readchains(rep+model+'-'+'LyaDR11_HPlanck1s_obh2Planck1s'+ext,add_extra=True)
lya_hr2_obh2p1=mcmc.readchains(rep+model+'-'+'LyaDR11_HRiess2s_obh2Planck1s'+ext,add_extra=True)
lya_hrp_obh2p1=mcmc.readchains(rep+model+'-'+'LyaDR11_HRiessPlanck_obh2Planck1s'+ext,add_extra=True)

bao_hrp_obh2p1=mcmc.readchains(rep+model+'-'+'BAO_HRiessPlanck_obh2Planck1s'+ext,add_extra=True)
lya_bao_hrp_obh2p1=mcmc.readchains(rep+model+'-'+'LyaDR11_BAO_HRiessPlanck_obh2Planck1s'+ext,add_extra=True)
lya_bao=mcmc.readchains(rep+model+'-'+'LyaDR11+BAO'+ext,add_extra=True)
###################################################

reload(mcmc)
clf()
limits=[[0,1],[0,1.5],[0.6,0.8],[0.021,0.023]]
vars=['omega_M_0','omega_lambda_0','h','obh2']
a2=mcmc.matrixplot(lya_hrp_obh2p1,vars,'red',8,limits=limits,alpha=0.5)
a0=mcmc.matrixplot(lya_hp1_obh2p1,vars,'blue',8,limits=limits,alpha=0.5)
a1=mcmc.matrixplot(lya_hr1_obh2p1,vars,'green',8,limits=limits,alpha=0.5)
subplot(2,2,2)
axis('off')
legend([a0,a1,a2],['BAO Lyman-alpha + h Planck (1s) + Obh2 Planck (1s)',
                   'BAO Lyman-alpha + h Riess (1s) + Obh2 Planck (1s)',
                   'BAO Lyman-alpha + h Riess/Planck + Obh2 Planck (1s)'],title=model)

reload(mcmc)
clf()
limits=[[0.,1],[0,1.5]]
vars=['omega_M_0','omega_lambda_0']
a2=mcmc.matrixplot(lya_hrp_obh2p1,vars,'blue',8,limits=limits,alpha=0.5)
subplot(len(vars),len(vars),len(vars))
axis('off')
legend([a2],['BAO Lyman-alpha + h Riess/Planck + Obh2 Planck (1s)'],title=model)


clf()
a2=mcmc.cont(lya_hrp_obh2p1['omega_M_0'],lya_hrp_obh2p1['omega_lambda_0'],nsmooth=3)
xx=linspace(0,1,100)
plot(xx,1.-xx,'k--')
plot(xx*0+omplanckwp,2*xx,'k:')
plot(xx,xx*0+1-omplanckwp,'k:')
a1=plot(omplanckwp,1-omplanckwp,'*',color='yellow',ms=20)
xlim([0,1])
ylim([0,2])
xlabel('$\Omega_M$')
ylabel('$\Omega_\Lambda$')
legend([a2],['BAO Lyman-alpha + h Riess/Planck + $\Omega_b h^2$ Planck'],frameon=False)
title('Open $\Lambda$CDM')
savefig('/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/Jan2014/olambdacdm_lya.png',bbox_inches="tight")


reload(mcmc)
clf()
limits=[[0,1],[0,1.5],[0.6,0.8],[0.021,0.023]]
vars=['omega_M_0','omega_lambda_0','h','obh2']
a0=mcmc.matrixplot(lya_hrp_obh2p1,vars,'blue',4,limits=limits,alpha=0.5)
a1=mcmc.matrixplot(bao_hrp_obh2p1,vars,'green',4,limits=limits,alpha=0.5)
a2=mcmc.matrixplot(lya_bao_hrp_obh2p1,vars,'red',4,limits=limits,alpha=0.5)
subplot(2,2,2)
axis('off')
legend([a0,a1,a2],['BAO Lyman-alpha + h + $\Omega_b h^2$',
                   'BAO LRG + h + $\Omega_b h^2$',
                   'BAO Lyman-alpha + LRG + h + $\Omega_b h^2$'],title='Open $\Lambda$CDM')
savefig('/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/Jan2014/olambdacdm_combined.png',bbox_inches="tight")


clf()
xx=linspace(0,1,100)
plot(xx,1.-xx,'k--')
plot(xx*0+omplanckwp,2*xx,'k:')
plot(xx,xx*0+1-omplanckwp,'k:')
a0=mcmc.cont(lya_hrp_obh2p1['omega_M_0'],lya_hrp_obh2p1['omega_lambda_0'],nsmooth=3,color='blue',alpha=0.5)
a1=mcmc.cont(bao_hrp_obh2p1['omega_M_0'],bao_hrp_obh2p1['omega_lambda_0'],nsmooth=3,color='green',alpha=0.5)
a2=mcmc.cont(lya_bao_hrp_obh2p1['omega_M_0'],lya_bao_hrp_obh2p1['omega_lambda_0'],nsmooth=3,color='red',alpha=0.5)
aa=plot(omplanckwp,1-omplanckwp,'*',color='yellow',ms=20)
xlim([0,1])
ylim([0,1.8])
xlabel('$\Omega_M$')
ylabel('$\Omega_\Lambda$')
legend([a0,a1,a2],['BAO Lyman-alpha + h + $\Omega_b h^2$',
                   'BAO LRG + h + $\Omega_b h^2$',
                   'BAO Lyman-alpha + LRG + h + $\Omega_b h^2$'],frameon=False)
title('Open $\Lambda$CDM')
savefig('/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/Jan2014/olambdacdm_OmOl_combined.png',bbox_inches="tight")

clf()
xx=linspace(0,1,100)
plot(xx,1.-xx,'k--')
plot(xx*0+omplanckwp,2*xx,'k:')
plot(xx,xx*0+1-omplanckwp,'k:')
a0=mcmc.cont(lya_hrp_obh2p1['omega_M_0'],lya_hrp_obh2p1['omega_lambda_0'],nsmooth=3,color='blue',alpha=0.5)
a1=mcmc.cont(bao_hrp_obh2p1['omega_M_0'],bao_hrp_obh2p1['omega_lambda_0'],nsmooth=3,color='green',alpha=0.5)
a3=mcmc.cont(lya_bao['omega_M_0'],lya_bao['omega_lambda_0'],nsmooth=3,color='brown',alpha=0.5)
a2=mcmc.cont(lya_bao_hrp_obh2p1['omega_M_0'],lya_bao_hrp_obh2p1['omega_lambda_0'],nsmooth=3,color='red',alpha=0.5)
aa=plot(omplanckwp,1-omplanckwp,'*',color='yellow',ms=20)
xlim([0,1])
ylim([0,1.8])
xlabel('$\Omega_M$')
ylabel('$\Omega_\Lambda$')
legend([a0,a1,a2,a3],['BAO Lyman-alpha + h + $\Omega_b h^2$',
                   'BAO LRG + h + $\Omega_b h^2$',
                   'BAO Lyman-alpha + LRG + h + $\Omega_b h^2$',
                   'BAO Lyman-alpha + LRG'],frameon=False)
title('Open $\Lambda$CDM')
savefig('/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/Jan2014/olambdacdm_OmOl_combined_checkpriors.png',bbox_inches="tight")


########################  w  ################################
model='flatwcdm'
lya_hp1_obh2p1=mcmc.readchains(rep+model+'-'+'LyaDR11_HPlanck1s_obh2Planck1s'+ext,add_extra=True)
lya_hr2_obh2p1=mcmc.readchains(rep+model+'-'+'LyaDR11_HRiess2s_obh2Planck1s'+ext,add_extra=True)
lya_hrp_obh2p1=mcmc.readchains(rep+model+'-'+'LyaDR11_HRiessPlanck_obh2Planck1s'+ext,add_extra=True)

bao_hrp_obh2p1=mcmc.readchains(rep+model+'-'+'BAO_HRiessPlanck_obh2Planck1s'+ext,add_extra=True)
lya_bao_hrp_obh2p1=mcmc.readchains(rep+model+'-'+'LyaDR11_BAO_HRiessPlanck_obh2Planck1s'+ext,add_extra=True)
lya_bao=mcmc.readchains(rep+model+'-'+'LyaDR11+BAO'+ext,add_extra=True)
###################################################


reload(mcmc)
clf()
limits=[[0,0.4],[-2.5,0],[0.6,0.8],[0.021,0.023]]
vars=['omega_M_0','w','h','obh2']
a2=mcmc.matrixplot(lya_hrp_obh2p1,vars,'red',8,limits=limits,alpha=0.5)
a0=mcmc.matrixplot(lya_hp1_obh2p1,vars,'blue',8,limits=limits,alpha=0.5)
a1=mcmc.matrixplot(lya_hr1_obh2p1,vars,'green',8,limits=limits,alpha=0.5)
subplot(2,2,2)
axis('off')
legend([a0,a1,a2],['BAO Lyman-alpha + h Planck (1s) + Obh2 Planck (1s)',
                   'BAO Lyman-alpha + h Riess (1s) + Obh2 Planck (1s)',
                   'BAO Lyman-alpha + h Riess/Planck + Obh2 Planck (1s)'],title=model)

reload(mcmc)
clf()
limits=[[0,0.4],[-2.5,0],[0.6,0.8],[0.021,0.023]]
vars=['omega_M_0','w','h','obh2']
a0=mcmc.matrixplot(lya_hrp_obh2p1,vars,'blue',3,limits=limits,alpha=0.5)
a1=mcmc.matrixplot(bao_hrp_obh2p1,vars,'green',3,limits=limits,alpha=0.5)
a2=mcmc.matrixplot(lya_bao_hrp_obh2p1,vars,'red',3,limits=limits,alpha=0.5)
subplot(2,2,2)
axis('off')
legend([a0,a1,a2],['BAO Lyman-alpha + h + $\Omega_b h^2$',
                   'BAO LRG + h + $\Omega_b h^2$',
                   'BAO Lyman-alpha + LRG + h + $\Omega_b h^2$'],title='Open $\Lambda$CDM')
savefig('/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/Jan2014/flatwcdm_combined.png',bbox_inches="tight")

reload(mcmc)
clf()
limits=[[0.,0.4],[-2.5,0]]
vars=['omega_M_0','w']
a2=mcmc.matrixplot(lya_hrp_obh2p1,vars,'blue',3,limits=limits,alpha=0.5)
subplot(len(vars),len(vars),len(vars))
axis('off')
legend([a2],['BAO Lyman-alpha + h Riess/Planck + Obh2 Planck (1s)'],title=model)

omplanckwp=0.3183
clf()
a2=mcmc.cont(lya_hrp_obh2p1['omega_M_0'],lya_hrp_obh2p1['w'],alpha=0.7,nsmooth=3)
xx=linspace(0,1,100)
plot(xx,xx*0-1,'k:')
plot(xx*0+omplanckwp,-2*xx,'k:')
a1=plot(omplanckwp,-1,'*',color='yellow',ms=20)
xlim([0,0.4])
ylim([-2,0])
xlabel('$\Omega_M$')
ylabel('$w$')
legend([a2],['BAO Lyman-alpha + h Riess/Planck + $\Omega_b h^2$ Planck'],frameon=False)
title('Flat $w$CDM')
savefig('/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/Jan2014/flatwcdm_lya.png',bbox_inches="tight")

omplanckwp=0.3183
clf()
plot(xx,xx*0-1,'k:')
plot(xx*0+omplanckwp,-2*xx,'k:')
a0=mcmc.cont(lya_hrp_obh2p1['omega_M_0'],lya_hrp_obh2p1['w'],alpha=0.5,color='blue',nsmooth=3)
a1=mcmc.cont(bao_hrp_obh2p1['omega_M_0'],bao_hrp_obh2p1['w'],alpha=0.5,color='green',nsmooth=3)
a2=mcmc.cont(lya_bao_hrp_obh2p1['omega_M_0'],lya_bao_hrp_obh2p1['w'],alpha=0.5,color='red',nsmooth=3)
xx=linspace(0,1,100)
aa=plot(omplanckwp,-1,'*',color='yellow',ms=20)
xlim([0,0.4])
ylim([-2,0])
xlabel('$\Omega_M$')
ylabel('$w$')
legend([a0,a1,a2],['BAO Lyman-alpha + h + $\Omega_b h^2$',
                   'BAO LRG + h + $\Omega_b h^2$',
                   'BAO Lyman-alpha + LRG + h + $\Omega_b h^2$'],frameon=False)
title('Flat $w$CDM')
savefig('/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/Jan2014/flatwcdm_Omw_lya.png',bbox_inches="tight")

omplanckwp=0.3183
clf()
plot(xx,xx*0-1,'k:')
plot(xx*0+omplanckwp,-2*xx,'k:')
a0=mcmc.cont(lya_hrp_obh2p1['omega_M_0'],lya_hrp_obh2p1['w'],alpha=0.5,color='blue',nsmooth=3)
a1=mcmc.cont(bao_hrp_obh2p1['omega_M_0'],bao_hrp_obh2p1['w'],alpha=0.5,color='green',nsmooth=3)
a3=mcmc.cont(lya_bao['omega_M_0'],lya_bao['w'],alpha=0.5,color='brown',nsmooth=3)
a2=mcmc.cont(lya_bao_hrp_obh2p1['omega_M_0'],lya_bao_hrp_obh2p1['w'],alpha=0.5,color='red',nsmooth=3)
xx=linspace(0,1,100)
aa=plot(omplanckwp,-1,'*',color='yellow',ms=20)
xlim([0,0.4])
ylim([-2,0])
xlabel('$\Omega_M$')
ylabel('$w$')
legend([a0,a1,a2,a3],['BAO Lyman-alpha + h + $\Omega_b h^2$',
                   'BAO LRG + h + $\Omega_b h^2$',
                   'BAO Lyman-alpha + LRG + h + $\Omega_b h^2$',
                   'BAO Lyman-alpha + LRG'],frameon=False)
title('Flat $w$CDM')
savefig('/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/Jan2014/flatwcdm_Omw_lya_checkpriors.png',bbox_inches="tight")


######################## Ok,w ################################
model='owcdm'
lya_hp1_obh2p1=mcmc.readchains(rep+model+'-'+'LyaDR11_HPlanck1s_obh2Planck1s'+ext,add_extra=True)
lya_hr2_obh2p1=mcmc.readchains(rep+model+'-'+'LyaDR11_HRiess2s_obh2Planck1s'+ext,add_extra=True)
lya_hrp_obh2p1=mcmc.readchains(rep+model+'-'+'LyaDR11_HRiessPlanck_obh2Planck1s'+ext,add_extra=True)

bao_hrp_obh2p1=mcmc.readchains(rep+model+'-'+'BAO_HRiessPlanck_obh2Planck1s'+ext,add_extra=True)
lya_bao_hrp_obh2p1=mcmc.readchains(rep+model+'-'+'LyaDR11_BAO_HRiessPlanck_obh2Planck1s'+ext,add_extra=True)
lya_bao=mcmc.readchains(rep+model+'-'+'LyaDR11+BAO'+ext,add_extra=True)
###################################################


reload(mcmc)
clf()
alpha=0.4
limits=[[0,1],[0,2],[-2.5,0],[0.6,0.8],[0.021,0.023]]
vars=['omega_M_0','omega_lambda_0','w','h','obh2']
a2=mcmc.matrixplot(lya_hrp_obh2p1,vars,'red',3,limits=limits,alpha=alpha)
a0=mcmc.matrixplot(lya_hp1_obh2p1,vars,'blue',3,limits=limits,alpha=alpha)
a1=mcmc.matrixplot(lya_hr1_obh2p1,vars,'green',3,limits=limits,alpha=alpha)
subplot(2,2,2)
axis('off')
legend([a0,a1,a2],['BAO Lyman-alpha + h Planck (1s) + Obh2 Planck (1s)',
                   'BAO Lyman-alpha + h Riess (1s) + Obh2 Planck (1s)',
                   'BAO Lyman-alpha + h Riess/Planck + Obh2 Planck (1s)'],title=model)

reload(mcmc)
clf()
alpha=0.4
limits=[[0,1],[0,2],[-2.5,0],[0.6,0.8],[0.021,0.023]]
vars=['omega_M_0','omega_lambda_0','w','h','obh2']
a0=mcmc.matrixplot(lya_hrp_obh2p1,vars,'blue',4,limits=limits,alpha=0.5)
a1=mcmc.matrixplot(bao_hrp_obh2p1,vars,'green',8,limits=limits,alpha=0.5)
a2=mcmc.matrixplot(lya_bao_hrp_obh2p1,vars,'red',4,limits=limits,alpha=0.5)
subplot(2,2,2)
axis('off')
legend([a0,a1,a2],['BAO Lyman-alpha + h + $\Omega_b h^2$',
                   'BAO LRG + h + $\Omega_b h^2$',
                   'BAO Lyman-alpha + LRG + h + $\Omega_b h^2$'],title='Open wCDM')

reload(mcmc)
clf()
alpha=0.7
limits=[[0.,1],[0,2],[-2.5,0]]
vars=['omega_M_0','omega_lambda_0','w']
a2=mcmc.matrixplot(lya_hrp_obh2p1,vars,'blue',8,limits=limits,alpha=alpha)
subplot(len(vars),len(vars),len(vars))
axis('off')
legend([a2],[ 'BAO Lyman-alpha + h Riess/Planck + Obh2 Planck (1s)'],title=model)
subplot(3,3,4)
xx=linspace(0,1,100)
plot(xx,xx*0+1-omplanckwp,'k:')
plot(xx*0+omplanckwp,xx*2,'k:')
plot(xx,1-xx,'k--')
a1=plot(omplanckwp,1.-omplanckwp,'*',color='yellow',ms=10)
subplot(3,3,7)
plot(xx,xx*0-1,'k:')
plot(xx*0+omplanckwp,xx*3-3,'k:')
a1=plot(omplanckwp,-1,'*',color='yellow',ms=10)
subplot(3,3,8)
plot(xx*2,xx*0-1,'k:')
plot(xx*0+1-omplanckwp,xx*3-3,'k:')
a1=plot(1.-omplanckwp,-1,'*',color='yellow',ms=10)
savefig('/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/Jan2014/owcdm_lya.png',bbox_inches="tight")

reload(mcmc)
clf()
alpha=0.5
limits=[[0.,1],[0,2],[-2.5,0]]
vars=['omega_M_0','omega_lambda_0','w']
a0=mcmc.matrixplot(lya_hrp_obh2p1,vars,'blue',4,limits=limits,alpha=alpha)
a1=mcmc.matrixplot(bao_hrp_obh2p1,vars,'green',8,limits=limits,alpha=alpha)
a2=mcmc.matrixplot(lya_bao_hrp_obh2p1,vars,'red',4,limits=limits,alpha=alpha)
subplot(len(vars),len(vars),len(vars))
axis('off')
legend([a0,a1,a2],['BAO Lyman-alpha + h + $\Omega_b h^2$',
                   'BAO LRG + h + $\Omega_b h^2$',
                   'BAO Lyman-alpha + LRG + h + $\Omega_b h^2$'],title='Open wCDM')
subplot(3,3,4)
xx=linspace(0,1,100)
plot(xx,xx*0+1-omplanckwp,'k:')
plot(xx*0+omplanckwp,xx*2,'k:')
plot(xx,1-xx,'k--')
a1=plot(omplanckwp,1.-omplanckwp,'*',color='yellow',ms=10)
subplot(3,3,7)
plot(xx,xx*0-1,'k:')
plot(xx*0+omplanckwp,xx*3-3,'k:')
a1=plot(omplanckwp,-1,'*',color='yellow',ms=10)
subplot(3,3,8)
plot(xx*2,xx*0-1,'k:')
plot(xx*0+1-omplanckwp,xx*3-3,'k:')
a1=plot(1.-omplanckwp,-1,'*',color='yellow',ms=10)
savefig('/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/Jan2014/owcdm_lya_combined.png',bbox_inches="tight")





###### fixing h and obh2
model='olambdacdm_fixed_h_obh2'
lya_fixed=mcmc.readchains(rep+model+'-'+'LyaDR11'+ext)
model='olambdacdm'
lya_priors=mcmc.readchains(rep+model+'-'+'LyaDR11_HRiessPlanck_obh2Planck1s'+ext,add_extra=True)


clf()
a2=mcmc.cont(lya_priors['omega_M_0'],lya_priors['omega_lambda_0'],nsmooth=3,color='blue',alpha=0.5)
a3=mcmc.cont(lya_fixed['omega_M_0'],lya_fixed['omega_lambda_0'],nsmooth=3,color='red',alpha=0.5)
xx=linspace(0,1,100)
plot(xx,1.-xx,'k--')
plot(xx*0+omplanckwp,2*xx,'k:')
plot(xx,xx*0+1-omplanckwp,'k:')
a1=plot(omplanckwp,1-omplanckwp,'*',color='yellow',ms=20)
xlim([0,1])
ylim([0,2])
xlabel('$\Omega_M$')
ylabel('$\Omega_\Lambda$')
legend([a2,a3],['BAO Lyman-alpha + $h$ Riess/Planck + $\Omega_b h^2$ Planck', 'BAO Lyman-alpha + $h=0.706$ and $\Omega_b h^2=0.02207$ fixed'],frameon=False)
title('Open $\Lambda$CDM')
savefig('/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/Jan2014/olambdacdm_lya_priors_or_fixed.png',bbox_inches="tight")


model='flatwcdm_fixed_h_obh2'
lya_fixed=mcmc.readchains(rep+model+'-'+'LyaDR11'+ext)
model='flatwcdm'
lya_priors=mcmc.readchains(rep+model+'-'+'LyaDR11_HRiessPlanck_obh2Planck1s'+ext,add_extra=True)

omplanckwp=0.3183
clf()
a2=mcmc.cont(lya_priors['omega_M_0'],lya_priors['w'],alpha=0.5,nsmooth=2,color='blue')
a3=mcmc.cont(lya_fixed['omega_M_0'],lya_fixed['w'],alpha=0.5,nsmooth=2,color='red')
xx=linspace(0,1,100)
plot(xx,xx*0-1,'k:')
plot(xx*0+omplanckwp,-2*xx,'k:')
a1=plot(omplanckwp,-1,'*',color='yellow',ms=20)
xlim([0,0.4])
ylim([-2,0])
xlabel('$\Omega_M$')
ylabel('$w$')
legend([a2,a3],['BAO Lyman-alpha + $h$ Riess/Planck + $\Omega_b h^2$ Planck', 'BAO Lyman-alpha + $h=0.706$ and $\Omega_b h^2=0.02207$ fixed'],frameon=False)
title('Flat $w$CDM')
savefig('/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/Jan2014/flatwcdm_lya_priors_or_fixed.png.png',bbox_inches="tight")

model='owcdm_fixed_h_obh2'
lya_fixed=mcmc.readchains(rep+model+'-'+'LyaDR11'+ext)
model='owcdm'
lya_priors=mcmc.readchains(rep+model+'-'+'LyaDR11_HRiessPlanck_obh2Planck1s'+ext,add_extra=True)


reload(mcmc)
clf()
alpha=0.5
limits=[[0.,1],[0,2],[-2.5,0]]
vars=['omega_M_0','omega_lambda_0','w']
a2=mcmc.matrixplot(lya_priors,vars,'blue',5,limits=limits,alpha=alpha)
a3=mcmc.matrixplot(lya_fixed,vars,'red',5,limits=limits,alpha=alpha)
subplot(len(vars),len(vars),len(vars))
axis('off')
legend([a2,a3],['BAO Lyman-alpha + $h$ Riess/Planck + $\Omega_b h^2$ Planck', 'BAO Lyman-alpha + $h=0.706$ and $\Omega_b h^2=0.02207$ fixed'],frameon=False,title='Open $w$CDM')
subplot(3,3,4)
xx=linspace(0,1,100)
plot(xx,xx*0+1-omplanckwp,'k:')
plot(xx*0+omplanckwp,xx*2,'k:')
plot(xx,1-xx,'k--')
a1=plot(omplanckwp,1.-omplanckwp,'*',color='yellow',ms=10)
subplot(3,3,7)
plot(xx,xx*0-1,'k:')
plot(xx*0+omplanckwp,xx*3-3,'k:')
a1=plot(omplanckwp,-1,'*',color='yellow',ms=10)
subplot(3,3,8)
plot(xx*2,xx*0-1,'k:')
plot(xx*0+1-omplanckwp,xx*3-3,'k:')
a1=plot(1.-omplanckwp,-1,'*',color='yellow',ms=10)
savefig('/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/Jan2014/owcdm_lya_priors_or_fixed.png',bbox_inches="tight")
