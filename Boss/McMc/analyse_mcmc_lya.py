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
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm BAO_Beutler &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm BAO_DR7 &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm BAO_Anderson &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm BAO &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11+BAO &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11 &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm planck &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm planck+BAO &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm planck+LyaDR11 &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm planck+BAO+LyaDR11 &

#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm BAO_Beutler &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm BAO_DR7 &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm BAO_Anderson &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm BAO &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11+BAO &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm LyaDR11 &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm planck &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm planck+BAO &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm planck+LyaDR11 &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py flatwcdm planck+BAO+LyaDR11 &

#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm BAO_Beutler &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm BAO_DR7 &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm BAO_Anderson &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm BAO &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11+BAO &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11 &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm planck &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm planck+BAO &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm planck+LyaDR11 &
#xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm planck+BAO+LyaDR11 &

rep='/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/'
ext='.db'

######################## Ok ################################
model='olambdacdm'
planck=mcmc.readchains(rep+model+'-'+'planck'+ext,add_extra=True)
planck_bao=mcmc.readchains(rep+model+'-'+'planck+BAO'+ext,add_extra=True)
planck_bao_lya=mcmc.readchains(rep+model+'-'+'planck+BAO+LyaDR11'+ext,add_extra=True)
planck_lya=mcmc.readchains(rep+model+'-'+'planck+LyaDR11'+ext,add_extra=True)
lya=mcmc.readchains(rep+model+'-'+'LyaDR11'+ext,add_extra=True)
beutler=mcmc.readchains(rep+model+'-'+'BAO_Beutler'+ext,add_extra=True)
dr7=mcmc.readchains(rep+model+'-'+'BAO_DR7'+ext,add_extra=True)
anderson=mcmc.readchains(rep+model+'-'+'BAO_Anderson'+ext,add_extra=True)
bao=mcmc.readchains(rep+model+'-'+'BAO'+ext,add_extra=True)
lya_bao=mcmc.readchains(rep+model+'-'+'LyaDR11+BAO'+ext,add_extra=True)
###################################################



clf()
plot(lya['om_ol'],lya['rssqrtomh2'],'k,')
ylim(0,100)

clf()
plot(bao['om_ol'],bao['rssqrtomh2'],'k,')
ylim(0,100)

reload(mcmc)
clf()
vars=['om_ol','rssqrtomh2']
limits=[[0.,3],[0,100]]
doit=[True,True]
a0=mcmc.matrixplot(bao,vars,'yellow',8,limits=limits,doit=doit)
a1=mcmc.matrixplot(lya,vars,'purple',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(lya_bao,vars,'blue',4,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a0,a1,a2],['BAO Gal','BAO Lya','All BAO (LRG+Lya)'],title=model)



reload(mcmc)
clf()
vars=['omega_M_0','omega_lambda_0','w','h']
limits=[[0.,0.5],[-0.5,2],[-2,0],[0.,1.5]]
doit=[True,True,False,True]
a0=mcmc.matrixplot(bao,vars,'yellow',8,limits=limits,doit=doit)
a1=mcmc.matrixplot(lya,vars,'purple',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(lya_bao,vars,'blue',4,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a0,a1,a2],['BAO Gal','BAO Lya','All BAO (LRG+Lya)'],title=model)
savefig(model+'_all_bao_gal_lya.pdf')




reload(mcmc)
clf()
vars=['omega_M_0','omega_lambda_0','w','h']
limits=[[0.,0.5],[0.3,1.4],[-2,0],[0.4,1.]]
doit=[True,True,False,True]
a0=mcmc.matrixplot(planck,vars,'green',8,limits=limits,doit=doit)
a1=mcmc.matrixplot(lya_bao,vars,'blue',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(planck_bao_lya,vars,'red',2,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a0,a1,a2],['Planck','BAO (LRG+Lya)','Planck + BAO (LRG+Lya)'],title=model)
savefig(model+'_all_bao_planck.pdf')


reload(mcmc)
clf()
vars=['omega_M_0','omega_k_0','w','h']
limits=[[0.,0.5],[-0.1,0.1],[-2.5,-0.5],[0.4,1.]]
doit=[True,True,False,True]
a1=mcmc.matrixplot(planck_lya,vars,'blue',8,limits=limits,doit=doit)
a0=mcmc.matrixplot(planck_bao,vars,'green',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(planck_bao_lya,vars,'red',4,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a1,a0,a2],['Planck + BAO (Lya)','Planck + BAO (LRG)','Planck + BAO (LRG+Lya)'],title=model)
savefig(model+'_all_planck_lya_lrg.pdf')


reload(mcmc)
clf()
vars=['omega_M_0','omega_k_0','w','h']
limits=[[0.2,0.4],[-0.025,0.025],[-1.8,-0.4],[0.6,0.8]]
doit=[True,True,False,True]
a0=mcmc.matrixplot(planck_bao,vars,'green',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(planck_bao_lya,vars,'red',4,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a0,a2],['Planck + BAO (LRG)','Planck + BAO (LRG+Lya)'],title=model)
savefig(model+'_all_planck_lya_improvement.pdf')



############################ Flat w ###################################
model='flatwcdm'
planck=mcmc.readchains(rep+model+'-'+'planck'+ext)
planck_bao=mcmc.readchains(rep+model+'-'+'planck+BAO'+ext)
planck_bao_lya=mcmc.readchains(rep+model+'-'+'planck+BAO+LyaDR11'+ext)
planck_lya=mcmc.readchains(rep+model+'-'+'planck+LyaDR11'+ext)
lya=mcmc.readchains(rep+model+'-'+'LyaDR11'+ext)
lya=mcmc.readchains(rep+model+'-'+'LyaDR11'+ext)
#beutler=mcmc.readchains(rep+model+'-'+'BAO_Beutler'+ext)
dr7=mcmc.readchains(rep+model+'-'+'BAO_DR7'+ext)
anderson=mcmc.readchains(rep+model+'-'+'BAO_Anderson'+ext)
bao=mcmc.readchains(rep+model+'-'+'BAO'+ext)
lya_bao=mcmc.readchains(rep+model+'-'+'LyaDR11+BAO'+ext)
###################################################



reload(mcmc)
clf()
vars=['omega_M_0','omega_lambda_0','w','h']
limits=[[0.,0.5],[-0.5,2],[-2,0],[0.,1.5]]
doit=[True,False,True,True]
a0=mcmc.matrixplot(bao,vars,'yellow',8,limits=limits,doit=doit)
a1=mcmc.matrixplot(lya,vars,'purple',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(lya_bao,vars,'blue',4,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a0,a1,a2],['BAO Gal','BAO Lya','All BAO (LRG+Lya)'],title=model)
savefig(model+'_all_bao_gal_lya.pdf')


reload(mcmc)
clf()
vars=['omega_M_0','omega_lambda_0','w','h']
limits=[[0.,0.5],[0.3,1.4],[-2,0],[0.4,1.]]
doit=[True,False,True,True]
a0=mcmc.matrixplot(planck,vars,'green',8,limits=limits,doit=doit)
a1=mcmc.matrixplot(lya_bao,vars,'blue',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(planck_bao_lya,vars,'red',2,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a0,a1,a2],['Planck','BAO (LRG+Lya)','Planck + BAO (LRG+Lya)'],title=model)
savefig(model+'_all_bao_planck.pdf')


reload(mcmc)
clf()
vars=['omega_M_0','omega_k_0','w','h']
limits=[[0.,0.5],[-0.1,0.1],[-2.5,-0.5],[0.4,1.]]
doit=[True,False,True,True]
a1=mcmc.matrixplot(planck_lya,vars,'blue',8,limits=limits,doit=doit)
a0=mcmc.matrixplot(planck_bao,vars,'green',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(planck_bao_lya,vars,'red',4,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a1,a0,a2],['Planck + BAO (Lya)','Planck + BAO (LRG)','Planck + BAO (LRG+Lya)'],title=model)
savefig(model+'_all_planck_lya_lrg.pdf')



reload(mcmc)
clf()
vars=['omega_M_0','omega_k_0','w','h']
limits=[[0.2,0.4],[-0.025,0.025],[-1.8,-0.4],[0.6,0.8]]
doit=[True,False,True,True]
a0=mcmc.matrixplot(planck_bao,vars,'green',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(planck_bao_lya,vars,'red',4,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a0,a2],['Planck + BAO (LRG)','Planck + BAO (LRG+Lya)'],title=model)
savefig(model+'_all_planck_lya_improvement.pdf')



########################################################################################################

############################ Open w ###################################
## les chaines open w de planck n'existent pas donc on a du bricoler
## on prend les chaines openlambdacdm et on ne garde que obh2, och2, theta
## qui sont fixes par la position du pic et l'amplitude des pics
## par contre on va relacher omegak et w
model='owcdm'
planck=mcmc.readchains(rep+model+'-'+'planck'+ext)
planck_bao=mcmc.readchains(rep+model+'-'+'planck+BAO'+ext)
planck_bao_lya=mcmc.readchains(rep+model+'-'+'planck+BAO+LyaDR11'+ext)
planck_lya=mcmc.readchains(rep+model+'-'+'planck+LyaDR11'+ext)
lya=mcmc.readchains(rep+model+'-'+'LyaDR11'+ext)
lya=mcmc.readchains(rep+model+'-'+'LyaDR11'+ext)
lya=mcmc.readchains(rep+model+'-'+'LyaDR11'+ext)
beutler=mcmc.readchains(rep+model+'-'+'BAO_Beutler'+ext)
dr7=mcmc.readchains(rep+model+'-'+'BAO_DR7'+ext)
anderson=mcmc.readchains(rep+model+'-'+'BAO_Anderson'+ext)
bao=mcmc.readchains(rep+model+'-'+'BAO'+ext)
lya_bao=mcmc.readchains(rep+model+'-'+'LyaDR11+BAO'+ext)
###################################################



reload(mcmc)
clf()
vars=['omega_M_0','omega_lambda_0','w','h']
limits=[[0.,0.5],[-0.5,2],[-2,0],[0.,1.5]]
doit=[True,True,True,True]
a0=mcmc.matrixplot(bao,vars,'yellow',8,limits=limits,doit=doit)
a1=mcmc.matrixplot(lya,vars,'purple',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(lya_bao,vars,'blue',4,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a0,a1,a2],['BAO Gal','BAO Lya','All BAO (LRG+Lya)'],title=model)
savefig(model+'_all_bao_gal_lya.pdf')



reload(mcmc)
clf()
vars=['omega_M_0','omega_lambda_0','w','h']
limits=[[0.,0.5],[0.3,1.4],[-2,0],[0.4,1.]]
doit=[True,True,True,True]
a0=mcmc.matrixplot(planck,vars,'green',8,limits=limits,doit=doit)
a1=mcmc.matrixplot(lya_bao,vars,'blue',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(planck_bao_lya,vars,'red',2,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a0,a1,a2],['Planck','BAO (LRG+Lya)','Planck + BAO (LRG+Lya)'],title=model)
savefig(model+'_all_bao_planck.pdf')



reload(mcmc)
clf()
vars=['omega_M_0','omega_k_0','w','h']
limits=[[0.,0.5],[-0.1,0.1],[-2.5,-0.5],[0.4,1.]]
doit=[True,True,True,True]
a1=mcmc.matrixplot(planck_lya,vars,'blue',8,limits=limits,doit=doit)
a0=mcmc.matrixplot(planck_bao,vars,'green',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(planck_bao_lya,vars,'red',4,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a1,a0,a2],['Planck + BAO (Lya)','Planck + BAO (LRG)','Planck + BAO (LRG+Lya)'],title=model)
savefig(model+'_all_planck_lya_lrg.pdf')



reload(mcmc)
clf()
vars=['omega_M_0','omega_k_0','w','h']
limits=[[0.2,0.4],[-0.025,0.025],[-1.8,-0.4],[0.6,0.8]]
doit=[True,True,True,True]
a0=mcmc.matrixplot(planck_bao,vars,'green',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(planck_bao_lya,vars,'red',4,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a0,a2],['Planck + BAO (LRG)','Planck + BAO (LRG+Lya)'],title=model)
savefig(model+'_all_planck_lya_improvement.pdf')


