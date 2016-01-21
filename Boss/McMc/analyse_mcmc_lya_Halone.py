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
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm BAO &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11_Halone+BAO &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py owcdm LyaDR11_Halone &

rep='/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/'
ext='.db'

######################## OwCDM ################################
model='owcdm'
lya=mcmc.readchains(rep+model+'-'+'LyaDR11_Halone'+ext,add_extra=True)
bao=mcmc.readchains(rep+model+'-'+'BAO'+ext,add_extra=True)
lya_bao=mcmc.readchains(rep+model+'-'+'LyaDR11_Halone+BAO'+ext,add_extra=True)
###################################################



reload(mcmc)
clf()
vars=['omega_M_0', 'omega_lambda_0','c_H0rs']
limits=[[0.,0.5],[0,1.2],[25,35]]
doit=[True,True,True]
a0=mcmc.matrixplot(bao,vars,'yellow',8,limits=limits,doit=doit)
a1=mcmc.matrixplot(lya,vars,'purple',4,limits=limits,doit=doit)
a2=mcmc.matrixplot(lya_bao,vars,'blue',4,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a0,a1,a2],['BAO Gal','BAO Lya','All BAO (LRG+Lya)'],title=model)







#### run in a shell
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm BAO &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11_Halone+BAO &
xterm -e python ~/Python/Boss/McMc/mcmc_launcher.py olambdacdm LyaDR11_Halone &


######################## OlambdaCDM ################################
model='olambdacdm'
lya=mcmc.readchains(rep+model+'-'+'LyaDR11_Halone'+ext,add_extra=True)
bao=mcmc.readchains(rep+model+'-'+'BAO'+ext,add_extra=True)
lya_bao=mcmc.readchains(rep+model+'-'+'LyaDR11_Halone+BAO'+ext,add_extra=True)
###################################################



reload(mcmc)
clf()
vars=['omega_M_0', 'omega_lambda_0','c_H0rs']
limits=[[0.,0.5],[0,1.2],[25,35]]
doit=[True,True,True]
a0=mcmc.matrixplot(bao,vars,'blue',8,limits=limits,doit=doit)
a2=mcmc.matrixplot(lya_bao,vars,'red',4,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a0,a2],['BAO Gal','All BAO (LRG+Lya)'],title=model)

######################## OlambdaCDM ################################
model='olambdacdm'
lya_bao_HDa=mcmc.readchains(rep+model+'-'+'LyaDR11+BAO'+ext,add_extra=True)
lya_bao_H=mcmc.readchains(rep+model+'-'+'LyaDR11_Halone+BAO'+ext,add_extra=True)
###################################################



reload(mcmc)
clf()
vars=['omega_M_0', 'omega_lambda_0','c_H0rs']
limits=[[0.,0.5],[0,1.2],[25,35]]
doit=[True,True,True]
a2=mcmc.matrixplot(lya_bao_H,vars,'blue',4,limits=limits,doit=doit)
a0=mcmc.matrixplot(lya_bao_HDa,vars,'red',8,limits=limits,doit=doit)
subplot(3,3,3)
axis('off')
legend([a0,a2],['BAO Gal + Lya(H and Da)','BAO Gal + Lya(H only)'],title=model)
savefig('with_or_without_da.pdf')

