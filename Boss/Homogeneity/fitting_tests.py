


##### test
from Homogeneity import minuit_jc
import numpy as np
import numpy.random as npr
npr.seed(0)

#### Input model
#def thefct(x, pars):
#    return pars[0]*np.exp(-x/pars[1])*np.sin(2*np.pi*x/pars[2]+pars[3])
#partrue = np.array([1.,2.,2.5,0.3])

def thefct(x, pars):
    return pars[0]*np.exp(-x/pars[1])
partrue = np.array([20.,2.])

    
xx = np.linspace(0,10,100)
clf()
plot(xx, thefct(xx, partrue))


#### simulated data
nbpts = 25
signoise = 1
err = np.linspace(0.7, 1.3, nbpts)*signoise
#signoise = 1.
#err = np.zeros(nbpts)+signoise
x = np.linspace(0,10,nbpts)
y = thefct(x, partrue) + randn(nbpts)*signoise
clf()
errorbar(x,y,err,fmt='.');
xlim(-0.2,10.2)

############### Fitting
initpar = np.array([1.,1.])
#### Minuit
toto_minuit=minuit_jc.do_minuit(x,y,err,initpar,functname=thefct)

clf()
xx = np.linspace(0,10,1000)
errorbar(x,y,err,fmt='.');
plot(xx,thefct(xx,toto_minuit[1]), label='Minuit')
legend()

#### Parameters
print(toto_minuit[1])

#### Error bars
print(toto_minuit[2])

#### Covariance Matrix
print(toto_minuit[3])


##### Now check error bars with a Monte-Carlo
nbmc = 1000
allpars = np.zeros((2, nbmc))
allerrs = np.zeros((2, nbmc))
allcovs = np.zeros((2,2,nbmc))

for i in xrange(nbmc):
    print(i)
    x = np.linspace(0,10,nbpts)
    y = thefct(x, partrue) + randn(nbpts)*signoise
    toto=minuit_jc.do_minuit(x,y,err,initpar,functname=thefct)
    allpars[:, i] = toto[1]
    allerrs[:, i] = toto[2]
    allcovs[:, :, i] = toto[3]

##### Parameters
print(partrue)
print(mean(allpars, axis=1))
print(toto_minuit[1])

##### Error bars
print(std(allpars, axis=1))
print(toto_minuit[2])

##### covariance
print(cov(allpars))
print(toto_minuit[3])




