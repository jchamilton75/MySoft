# A quadratic fit
import numpy, pymc

# create some test data
x = numpy.arange(100) * 0.3
f = 0.1 * x**2 - 2.6 * x - 1.5
numpy.random.seed(76523654)
noise = numpy.random.normal(size=100) * .1     # create some Gaussian noise
f = f + noise                                # add noise to the data

z = numpy.polyfit(x, f, 2)   # the traditional chi-square fit
print 'The chi-square result: ', z

#priors
sig = pymc.Uniform('sig', 0.0, 100.0, value=1.)

a = pymc.Uniform('a', -10.0, 10.0, value= 0.0)
b = pymc.Uniform('b', -10.0, 10.0, value= 0.0)
c = pymc.Uniform('c', -10.0, 10.0, value= 0.0)

#model
@pymc.deterministic(plot=False)
def mod_quadratic(x=x, a=a, b=b, c=c):
      return a*x**2 + b*x + c

#likelihood
y = pymc.Normal('y', mu=mod_quadratic, tau=1.0/sig**2, value=f, observed=True)


