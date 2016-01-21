


##### test
from Homogeneity import fitting
import numpy as np
import numpy.random as npr
npr.seed(0)


def thefct(x, pars):
    return pars[0]*np.exp(-x/pars[1])
partrue = np.array([20.,2.])

    
xx = np.linspace(0,10,100)
clf()
plot(xx, thefct(xx, partrue))


#### simulated data
nbpts = 25
signoise = 3.
err = np.linspace(0.7, 1.3, nbpts)*signoise
#signoise = 1.
#err = np.zeros(nbpts)+signoise
x = np.linspace(0,10,nbpts)
y = thefct(x, partrue) + randn(nbpts)*signoise
clf()
errorbar(x,y,err,fmt='.');
xlim(-0.2,10.2)

############### Fitting
initpar = np.array([10.,1.])
#### Minuit
toto_minuit=fitting.dothefit(x,y,err,initpar,functname=thefct)
#### MPFit (Fast Levenberg-Markardt minimizer)
toto_mpfit=fitting.dothefit(x,y,err,initpar,functname=thefct, method='mpfit')
#### MCMC [slow !!]
toto_mcmc=fitting.dothefit(x,y,err,initpar,functname=thefct, method='mcmc')

clf()
xx = np.linspace(0,10,1000)
plot(xx,thefct(xx,toto_mcmc[1]), label='MCMC')
chains = toto_mcmc[0]
for i in xrange(len(chains)/100):
    plot(xx,thefct(xx,chains[i,:]), color='b', alpha=0.01,lw=2)
errorbar(x,y,err,fmt='.');
plot(xx,thefct(xx,toto_minuit[1]), label='Minuit', lw=2)
plot(xx,thefct(xx,toto_mpfit[1]), label='MPFit', lw=2)
legend()

#### Parameters
print(toto_minuit[1])
print(toto_mpfit[1])
print(toto_mcmc[1])

#### Error bars
print(toto_minuit[2])
print(toto_mpfit[2])
print(toto_mcmc[2])

#### Covariance Matrix
print(toto_minuit[3])
print(toto_mpfit[3])
print(toto_mcmc[3])


#### Avantage de MCMC
chains = toto_mcmc[0]
clf()
plot(chains[:,0], chains[:,1],',')





