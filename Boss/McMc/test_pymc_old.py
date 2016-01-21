import cosmolopy
import pymc
from pymc import Metropolis
from McMc import mcmc
from astropy.io import fits
from McMc import cosmo_utils
import scipy

# post intéressant pour mettre son propre likelihood
#https://groups.google.com/forum/#!topic/pymc/u9v3XPOMWTY


################ SNIa #############################################
from McMc import model_sn1a
reload(model_sn1a)
S=pymc.MCMC(model_sn1a)
S.use_step_method(pymc.AdaptiveMetropolis,S.stochastics,delay=1000)
S.sample(iter=10000,burn=5000,thin=10)

clf()
xlim(0,1)
ylim(0,1.5)
mcmc.cont(S.trace('om')[:],S.trace('ol')[:],nsig=5,color='red')
xx=np.linspace(0,1,1000)
plot(xx,1-xx,'k:')
###################################################################



############# BAO Lyman-alpha DR11 ################################
from McMc import model_lyaDR11 as lya
reload(lya)

B=pymc.MCMC(lya)
B.use_step_method(pymc.AdaptiveMetropolis,B.stochastics,delay=1000)
B.sample(iter=50000,burn=10000,thin=10)

clf()
xlim(0,1)
ylim(0,1.5)
mcmc.cont_gkde(B.trace('om')[:],B.trace('ol')[:],nsig=5)
xx=np.linspace(0,1,1000)
plot(xx,1-xx,'k:')

reload(lya)
hvals=B.trace('h')[:]
omvals=B.trace('om')[:]
olvals=B.trace('ol')[:]
ll=np.zeros(hvals.size)
invhrs=np.zeros(hvals.size)
da_rs=np.zeros(hvals.size)
for i in np.arange(hvals.size):
    print(i,hvals.size)
    ll[i],invhrs[i],da_rs[i]=lya.theproba_ext(h=hvals[i], om=omvals[i],ol=olvals[i],ob=mycosmo['omega_b_0'])

mcmc.cont_gkde(invhrs,da_rs,fill=False,color='red',alpha=1)



###################################################################



######## Plot of both #############################################
clf()
xlim(0,1)
ylim(0,1.5)
mcmc.cont(B.trace('om')[:],B.trace('ol')[:],nsig=5,alpha=0.5)
mcmc.cont(S.trace('om')[:],S.trace('ol')[:],nsig=5,color='red',alpha=0.5)
xx=np.linspace(0,1,1000)
plot(xx,1-xx,'k:')
###################################################################



############# BAO Lyman-alpha DR11 Flat w ################################
from McMc import model_lyaDR11_flatw as lya
reload(lya)

B=pymc.MCMC(lya)
B.use_step_method(pymc.AdaptiveMetropolis,B.stochastics,delay=10000)
B.sample(iter=500000,burn=10000,thin=10)

clf()
xlim(0,1)
ylim(-2,0)
mcmc.cont_gkde(B.trace('om')[:],B.trace('w')[:],nsig=5)
xx=np.linspace(0,1,1000)
plot(xx,1-xx,'k:')

reload(lya)
hvals=B.trace('h')[:]
omvals=B.trace('om')[:]
olvals=1.-B.trace('om')[:]
wvals=B.trace('w')[:]
ll=np.zeros(hvals.size)
invhrs=np.zeros(hvals.size)
da_rs=np.zeros(hvals.size)
for i in np.arange(hvals.size):
    print(i,hvals.size)
    ll[i],invhrs[i],da_rs[i]=lya.thelogproba_ext(h=hvals[i], om=omvals[i],w=wvals[i],ob=0.0463)


reload(lya)
plot(invhrs,da_rs,'k,',alpha=0.2)
mcmc.cont_gkde(invhrs,da_rs,fill=False,color='red',alpha=1)
xlim(0.0022,0.0032)
ylim(6,16)

###################################################################





############# BAO Lyman-alpha DR11 Flat w NEW ################################
import pymc
from pymc import Metropolis
import cosmolopy
from McMc import mcmc
from astropy.io import fits
from McMc import cosmo_utils
import scipy
from McMc import model_lyaDR11_flatw_new as lya
reload(lya)

B=pymc.MCMC(lya)
B.use_step_method(pymc.AdaptiveMetropolis,B.stochastics,delay=10000)
B.sample(iter=100000,burn=10000,thin=10)

clf()
xlim(0,1)
ylim(-2,0)
mcmc.cont_gkde(B.trace('om')[:],B.trace('w')[:],nsig=5)
xx=np.linspace(0,1,1000)
plot(xx,1-xx,'k:')

h=B.trace('h')[:]
om=B.trace('om')[:]
w=B.trace('w')[:]
ob=B.trace('ob')[:]
invhrs=B.trace('invhrs')[:]
da_rs=B.trace('da_rs')[:]



reload(lya)
plot(invhrs,da_rs,'g,',alpha=0.6)
mcmc.cont_gkde(invhrs,da_rs,fill=False,color='green',alpha=1,nsig=5)
xlim(0.0022,0.0032)
ylim(6,16)




### compare with IDL drawn McMc (soft used for Busca et al. 2012)
import scipy.io
bla=scipy.io.readsav('chain_flatwcst_anislyaDR11.save')
chain=bla.chain
sh=shape(chain)
idl_da_rs=np.zeros(sh[0])
idl_invhrs=np.zeros(sh[0])
for i in np.arange(sh[0]):
    print(i,sh[0])
    idl_da_rs[i]=lya.my_da_rs(h=chain[i,4],om=chain[i,0],ob=0.0227/0.7**2,w=chain[i,2])
    idl_invhrs[i]=lya.my_invhrs(h=chain[i,4],om=chain[i,0],ob=0.0227/0.7**2,w=chain[i,2])

rndorder=argsort(np.random.random(sh[0]))
idl_invhrs_new=idl_invhrs[rndorder]
idl_da_rs_new=idl_da_rs[rndorder]
om_idl=chain[rndorder,0]
w_idl=chain[rndorder,2]

nmax=100000
clf()
xlim(0,1)
ylim(-2,0)
mcmc.cont_gkde(B.trace('om')[:],B.trace('w')[:],nsig=5)
mcmc.cont_gkde(om_idl[0:nmax],w_idl[0:nmax],nsig=5,color='red')
xx=np.linspace(0,1,1000)
plot(xx,1-xx,'k:')


nmax=100000
clf()
reload(lya)
#plot(idl_invhrs_new[0:nmax],idl_da_rs_new[0:nmax],'r,',alpha=0.2)
mcmc.cont_gkde(idl_invhrs_new[0:nmax],idl_da_rs_new[0:nmax],fill=False,color='green',alpha=1,nsig=5)
#plot(invhrs,da_rs,'k,',alpha=1)
mcmc.cont_gkde(invhrs,da_rs,fill=False,color='red',alpha=1,nsig=5)
xlim(0.0022,0.0032)
ylim(6,16)



clf()
xlim(0,1)
ylim(-2,0)
title(library)
mcmc.cont_gkde(om,w,color='blue')
xx=np.linspace(0,1,1000)
plot(xx,xx*0-1,'k:')

###################################################################


