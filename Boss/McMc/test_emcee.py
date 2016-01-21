import numpy as np
import emcee


####### Simulate fake data following a model ###############################
## the model
def funct(x, pars):
    return (pars[0]+pars[1]*x)*np.cos(2*np.pi*pars[2]*x)

## the parameters
pars_true=[1.,3.,3.]
clf()
plot(linspace(0,1,100),funct(linspace(0,1,100),pars_true))

## the data
nbx=30
noiserms=0.3
noiseval=randn(nbx)*noiserms
x=linspace(0,1,nbx)
y=funct(x,pars_true)+noiseval
dy=zeros(nbx)+noiserms

## plot the dataset
clf()
errorbar(x,y,yerr=dy,fmt='ro',label='Data')
plot(linspace(0,1,100),funct(linspace(0,1,100),pars_true),label='Input model')
legend(loc='upper left')
#############################################################################



####### Now MCMC fit the model with emcee ###################################
## define the log of the posterior
def lnprob(thepars, xvalues, yvalues, errors):
    chi = (yvalues-funct(xvalues,thepars))/errors
    chi2=sum(chi**2)/2.0
    return -chi2

nwalkers=20
ndim=size(pars_true)
p0=np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))*5

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[x, y, dy],threads=4)

pos, prob, state = sampler.run_mcmc(p0, 1000)
sampler.reset()

sampler.run_mcmc(pos, 10000)
chains=sampler.chain
fractions=sampler.acceptance_fraction


best=where(fractions == max(fractions))

bestwalker=best[0]


clf()
for i in arange(ndim):
    subplot(2,2,i)
    a=hist(reshape(chains[bestwalker,:,i],size(chains[bestwalker,:,i])),50)
    plot([pars_true[i],pars_true[i]],[0,max(a[0])],linewidth=2,color='r')
    print(i,pars_true[i],mean(chains[bestwalker,:,i]),std(chains[bestwalker,:,i]))

clf()
subplot(2,2,1)
plot(chains[bestwalker,:,0],chains[bestwalker,:,1],'r.')
plot(pars_true[0],pars_true[1],'b+',ms=10,mew=3)
subplot(2,2,2)
plot(chains[bestwalker,:,0],chains[bestwalker,:,2],'r.')
plot(pars_true[0],pars_true[2],'b+',ms=10,mew=3)
subplot(2,2,3)
plot(chains[bestwalker,:,1],chains[bestwalker,:,2],'r.')
plot(pars_true[1],pars_true[2],'b+',ms=10,mew=3)


##############################################################################











