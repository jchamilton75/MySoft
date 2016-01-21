import pymc
from pymc import Metropolis
import cosmolopy
from McMc import mcmc
from astropy.io import fits
from McMc import cosmo_utils
import scipy
mycosmo=cosmolopy.fidcosmo.copy()



### SN
#SN=pymc.MCMC(mcmc.generic_model([data_SNIa]))
#SN.use_step_method(pymc.AdaptiveMetropolis,SN.stochastics,delay=1000)
#SN.sample(iter=10000,burn=5000,thin=10)

### SN+Lya+h
#both=pymc.MCMC(mcmc.generic_model([data_SNIa,data_lyaDR11,data_hPlanck]))
#both.use_step_method(pymc.AdaptiveMetropolis,both.stochastics,delay=1000)
#both.sample(iter=10000,burn=5000,thin=10)





############################## LAMBDA CDM ######################################
from McMc import data_hPlanck
reload(data_hPlanck)
from McMc import data_lyaDR11
reload(data_lyaDR11)
from McMc import data_DR7
reload(data_DR7)
from McMc import data_Beutler
reload(data_Beutler)
from McMc import data_Anderson
reload(data_Anderson)
niter=3000000
nburn=10000
nthin=10
reload(mcmc)
library='jc'
model='LambdaCDM'
variables=['h','omega_M_0','omega_lambda_0']

### LYA+h
BAOh=pymc.MCMC(mcmc.generic_model([data_lyaDR11,data_hPlanck],variables=variables,library=library),db='pickle',dbname='BAOh_'+model+'_'+library+'.db')
BAOh.use_step_method(pymc.AdaptiveMetropolis,BAOh.stochastics,delay=1000)
BAOh.sample(iter=niter,burn=nburn,thin=nthin)
BAOh.db.close()

### DR7+h
DR7h=pymc.MCMC(mcmc.generic_model([data_DR7,data_hPlanck],variables=variables,library=library),db='pickle',dbname='BAOh_'+model+'_'+library+'.db')
DR7h.use_step_method(pymc.AdaptiveMetropolis,DR7h.stochastics,delay=1000)
DR7h.sample(iter=niter,burn=nburn,thin=nthin)
DR7h.db.close()

### Beutler+h
Beutlerh=pymc.MCMC(mcmc.generic_model([data_Beutler,data_hPlanck],variables=variables,library=library),db='pickle',dbname='BAOh_'+model+'_'+library+'.db')
Beutlerh.use_step_method(pymc.AdaptiveMetropolis,Beutlerh.stochastics,delay=1000)
Beutlerh.sample(iter=niter,burn=nburn,thin=nthin)
Beutlerh.db.close()

### Anderson+h
Andersonh=pymc.MCMC(mcmc.generic_model([data_Anderson,data_hPlanck],variables=variables,library=library),db='pickle',dbname='BAOh_'+model+'_'+library+'.db')
Andersonh.use_step_method(pymc.AdaptiveMetropolis,Andersonh.stochastics,delay=1000)
Andersonh.sample(iter=niter,burn=nburn,thin=nthin)
Andersonh.db.close()

### Anderson+Beutler+DR7+LyaDR11+h
AllBAOh=pymc.MCMC(mcmc.generic_model([data_Anderson,data_Beutler, data_DR7, data_lyaDR11, data_hPlanck],variables=variables,library=library),db='pickle',dbname='BAOh_'+model+'_'+library+'.db')
AllBAOh.use_step_method(pymc.AdaptiveMetropolis,AllBAOh.stochastics,delay=1000)
AllBAOh.sample(iter=niter,burn=nburn,thin=nthin)
AllBAOh.db.close()


reload(mcmc)
clf()
xlim(0,1)
ylim(0,1.5)
mcmc.cont_gkde(Beutlerh.trace('omega_M_0')[:],Beutlerh.trace('omega_lambda_0')[:],color='green')
mcmc.cont_gkde(DR7h.trace('omega_M_0')[:],DR7h.trace('omega_lambda_0')[:],color='orange')
mcmc.cont_gkde(Andersonh.trace('omega_M_0')[:],Andersonh.trace('omega_lambda_0')[:],color='pink')
mcmc.cont_gkde(BAOh.trace('omega_M_0')[:],BAOh.trace('omega_lambda_0')[:],color='blue')
mcmc.cont_gkde(AllBAOh.trace('omega_M_0')[:],AllBAOh.trace('omega_lambda_0')[:],color='red')
xx=np.linspace(0,1,1000)
plot(xx,1-xx,'k:')

range=[[-0.5,1.5],[-0.5,2.]]
reload(mcmc)
clf()
xlim(0,1)
ylim(0,1.5)
sm=2
mcmc.cont(Beutlerh.trace('omega_M_0')[:],Beutlerh.trace('omega_lambda_0')[:],color='green',nsmooth=sm)
mcmc.cont(DR7h.trace('omega_M_0')[:],DR7h.trace('omega_lambda_0')[:],color='orange',nsmooth=sm)
mcmc.cont(Andersonh.trace('omega_M_0')[:],Andersonh.trace('omega_lambda_0')[:],color='pink',nsmooth=sm)
mcmc.cont(BAOh.trace('omega_M_0')[:],BAOh.trace('omega_lambda_0')[:],color='blue',nsmooth=sm)
mcmc.cont(AllBAOh.trace('omega_M_0')[:],AllBAOh.trace('omega_lambda_0')[:],color='red',nsmooth=sm)
xx=np.linspace(0,1,1000)
plot(xx,1-xx,'k:')


############################## FLAT W CDM ######################################
from McMc import data_hPlanck
reload(data_hPlanck)
from McMc import data_lyaDR11
reload(data_lyaDR11)
from McMc import data_DR7
reload(data_DR7)
from McMc import data_Beutler
reload(data_Beutler)
from McMc import data_Anderson
reload(data_Anderson)

niter=3000000
nburn=10000
nthin=10
reload(mcmc)
reload(cosmo_utils)
model='wCDM'
variables=['h','omega_M_0','w']
#library='cosmolopy'
#library='astropy'
library='jc'

### LYA+h
BAOh=pymc.MCMC(mcmc.generic_model([data_lyaDR11,data_hPlanck],variables=variables,flat=True,library=library),db='pickle',dbname='BAOh_'+model+'_'+library+'.db')
BAOh.use_step_method(pymc.AdaptiveMetropolis,BAOh.stochastics,delay=1000)
BAOh.sample(iter=niter,burn=nburn,thin=nthin)
BAOh.db.close()

### DR7+h
DR7h=pymc.MCMC(mcmc.generic_model([data_DR7,data_hPlanck],variables=variables,flat=True,library=library),db='pickle',dbname='DR7h_'+model+'_'+library+'.db')
DR7h.use_step_method(pymc.AdaptiveMetropolis,DR7h.stochastics,delay=1000)
DR7h.sample(iter=niter,burn=nburn,thin=nthin)
DR7h.db.close()

### Beutler+h
Beutlerh=pymc.MCMC(mcmc.generic_model([data_Beutler,data_hPlanck],variables=variables,flat=True,library=library),db='pickle',dbname='Beutlerh_'+model+'_'+library+'.db')
Beutlerh.use_step_method(pymc.AdaptiveMetropolis,Beutlerh.stochastics,delay=1000)
Beutlerh.sample(iter=niter,burn=nburn,thin=nthin)
Beutlerh.db.close()

### Anderson+h
Andersonh=pymc.MCMC(mcmc.generic_model([data_Anderson,data_hPlanck],variables=variables,flat=True,library=library),db='pickle',dbname='Andersonh_'+model+'_'+library+'.db')
Andersonh.use_step_method(pymc.AdaptiveMetropolis,Andersonh.stochastics,delay=1000)
Andersonh.sample(iter=niter,burn=nburn,thin=nthin)
Andersonh.db.close()

### Anderson+Beutler+DR7+LyaDR11+h
AllBAOh=pymc.MCMC(mcmc.generic_model([data_Anderson,data_Beutler, data_DR7, data_lyaDR11, data_hPlanck],variables=variables,flat=True,library=library),db='pickle',dbname='AllBAOh_'+model+'_'+library+'.db')
AllBAOh.use_step_method(pymc.AdaptiveMetropolis,AllBAOh.stochastics,delay=1000)
AllBAOh.sample(iter=niter,burn=nburn,thin=nthin)
AllBAOh.db.close()


reload(mcmc)
clf()
xlim(0.0227/0.7**2,1)
ylim(-2,0)
mcmc.cont_gkde(Beutlerh.trace('omega_M_0')[:],Beutlerh.trace('w')[:],color='green')
mcmc.cont_gkde(DR7h.trace('omega_M_0')[:],DR7h.trace('w')[:],color='orange')
mcmc.cont_gkde(Andersonh.trace('omega_M_0')[:],Andersonh.trace('w')[:],color='pink')
mcmc.cont_gkde(BAOh.trace('omega_M_0')[:],BAOh.trace('w')[:],color='blue')
mcmc.cont_gkde(AllBAOh.trace('omega_M_0')[:],AllBAOh.trace('w')[:],color='red')
xx=np.linspace(0,1,1000)
plot(xx,xx*0-1,'k:')

reload(mcmc)
clf()
xlim(0.0227/0.7**2,1)
ylim(-2,0)
sm=3
mcmc.cont(Beutlerh.trace('omega_M_0')[:],Beutlerh.trace('w')[:],color='green',nsmooth=sm)
mcmc.cont(DR7h.trace('omega_M_0')[:],DR7h.trace('w')[:],color='orange',nsmooth=sm)
mcmc.cont(Andersonh.trace('omega_M_0')[:],Andersonh.trace('w')[:],color='pink',nsmooth=sm)
mcmc.cont(BAOh.trace('omega_M_0')[:],BAOh.trace('w')[:],color='blue',nsmooth=sm)
mcmc.cont(AllBAOh.trace('omega_M_0')[:],AllBAOh.trace('w')[:],color='red',nsmooth=sm)
xx=np.linspace(0,1,1000)
plot(xx,xx*0-1,'k:')








#### convergence checking

# Geweke: mean in segments compared with global mean
scores=pymc.geweke(BAOh,intervals=10)
pymc.Matplot.geweke_plot(scores)

# Raftery-Lewis
pymc.raftery_lewis(BAOh,q=0.68,r=0.01)

ft=scipy.fft(BAOh.trace('w')[:])
ps=abs(ft)**2
clf()
xscale('log')
yscale('log')
plot(ps)




