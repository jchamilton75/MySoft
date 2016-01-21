import math 
import array 
import random 
import pdb 
from pymc import * 
from scipy import special 
from scipy.stats import poisson 
import numpy as np 
from pymc.Matplot import plot 


mass = [3.09, 3.47, 3.96, 3.99, 4.17, 4.25, 4.62, 4.73, 5.11, 5.69] 
Mmin = 2.99 




def _model(mass, nstar): 

        #Priors with initial guesses 
        alpha = Uniform('alpha', -6,6, value=3.0) 
        Mmax = Uniform('Mmax', max(mass), 120., value=max(mass)+1) 

        #compute the likelihood function, i.e., c*M**-alpha 
        #c**-1 = int_Mmin^Mmax M**-alpha dM 
        def lnP(M,alpha,Mmax): 
                c1 = (1-alpha)/(Mmax**(1-alpha)-Mmin**(1-alpha)) 
                Pval = c1*M**-alpha 
                return np.log(Pval) 

        #compute posterior probability, so I can track it's value 
        @deterministic 
        def likeP(alpha=alpha, Mmax=Mmax): 
                logp1 = np.sum(lnP(mass, alpha, Mmax)) 
                return logp1 

        #set up posterior for use with PyMC 
        @stochastic(trace=True, observed=True, plot=False) 
        def likelihood(value=likeP, mass=mass, alpha=alpha, Mmax=Mmax): 
                logp = np.sum(lnP(mass, alpha, Mmax)) 
                return logp 

        #not computing a 'model' explicitly, so pass on it 
        @deterministic 
        def modelled_y(M=mass, alpha=alpha, Mmax=Mmax): 
                pass 

        return locals() 

#execute pymc 
model = pymc.MCMC(_model(mass, len(mass)))
model.sample(50000, burn=20000)
plot(model)
plt.show()
