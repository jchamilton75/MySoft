import pymc
from pymc import Metropolis
import cosmolopy
from McMc import mcmc
from astropy.io import fits
from McMc import cosmo_utils
import scipy
from McMc import data_lyaDR11
reload(data_lyaDR11)
from McMc import data_DR7
reload(data_DR7)
from McMc import data_Beutler
reload(data_Beutler)
from McMc import data_Anderson
reload(data_Anderson)

mycosmo=cosmolopy.fidcosmo.copy()



###### Planck base : on doit obtenir les memes plots que dans PlanckXVI fig2 page 10
from McMc import data_base_planck_lowl_lowLike as data_planck
reload(data_planck)

niter=30000
nburn=10000
nthin=10
reload(mcmc)
reload(cosmo_utils)
library='jc'
variables=['h','omega_M_0','omega_b_0']
flat=True

### Planck
Planck=pymc.MCMC(mcmc.generic_model([data_planck],variables=variables,library=library,flat=flat))
Planck.use_step_method(pymc.AdaptiveMetropolis,Planck.stochastics,delay=1000)
Planck.sample(iter=niter,burn=nburn,thin=nthin)


obh2=Planck.trace('omega_b_0')[:]*Planck.trace('h')[:]**2
och2=(Planck.trace('omega_M_0')[:]-Planck.trace('omega_b_0')[:])*Planck.trace('h')[:]**2
ol=Planck.trace('omega_lambda_0')[:]
h=Planck.trace('h')[:]
om=Planck.trace('omega_M_0')[:]
ok=Planck.trace('omega_k_0')

clf()
subplot(2,2,1)
xlim(0.02,0.025)
ylim(0.60,0.82)
scatter(obh2,ol,c=h,linewidth=0)
colorbar()
mcmc.cont_gkde(obh2,ol,color='red',fill=False,alpha=1)
xlabel('$\Omega_b h^2$')
ylabel('$\Omega_\Lambda$')

subplot(2,2,2)
xlim(0.098,0.132)
ylim(0.60,0.82)
scatter(och2,ol,c=h,linewidth=0)
colorbar()
mcmc.cont_gkde(och2,ol,color='red',fill=False)
xlabel('$\Omega_c h^2$')
ylabel('$\Omega_\Lambda$')

subplot(2,2,3)
xlim(0.02,0.025)
ylim(0.096,0.132)
scatter(obh2,och2,c=h,linewidth=0)
colorbar()
mcmc.cont_gkde(obh2,och2,color='red',alpha=1,fill=False)
xlabel('$\Omega_b h^2$')
ylabel('$\Omega_c h^2$')

### fig 3 page 12
subplot(2,2,4)
xlim(0.25,0.40)
ylim(62,72.5)
scatter(om,h*100,c=h,linewidth=0)
colorbar()
mcmc.cont_gkde(om,h*100,color='red',fill=False)
xlabel('$\Omega_m$')
ylabel('$H_0$')




