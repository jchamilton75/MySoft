import numpy as np
from scipy import integrate as scint
from pylab import *
import cosmolopy
import cosmolopy.distance as cd
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab

x=open('sn.txt','r')
y=x.readlines()
x.close()
a=len(y)

for i in range(a) :
   y[i]=y[i].split()

c=np.zeros((a,3))
z=np.zeros(a)
mu=np.zeros(a)
dmu=np.zeros(a)

for i in range(a):
   c[i][0] = y[i][1]
   c[i][1] = y[i][8]
   c[i][2] = y[i][9]
   z[i]=c[i][0]
   mu[i]=c[i][1]
   dmu[i]=c[i][2]



def Ez(z,Ol,Om):
   Ok=1.-Om-Ol
   dh=3.e5/70
   return (z>0.01)*np.sqrt( Om*(1.+z)**3 +Ok*(1.+z)**2 +Ol) + 1.*(z<=0.01)

def dc(z,Ol,Om):
   Ok=1.-Om-Ol
   dh=3.e5/70
   nbins=int(z*1000+100.)
   zt = linspace( 0., z,nbins)
   dt = dh/Ez(zt,Ol,Om)
   return scint.simps(dt, x = zt)





def dm(z,Ol,Om):
   Ok=1.-Om-Ol
   dh=3.e5/70
   if (Ok)>0 :
       return ( dh / np.sqrt(Ok)) * np.sinh( np.sqrt(Ok) * dc(z,Ol,Om)/dh)

   if (Ok)<0 :
       return ( dh / np.sqrt(-Ok)) * np.sin( np.sqrt(-Ok) * dc(z,Ol,Om)/dh)

   if (Ok)==0 :
       return dc(z,Ol,Om)

def dl(z,Ol,Om):
   return (1.+z)*dm(z,Ol,Om)



def mutheorique(z,Ol,Om):
   return 5.*np.log10(dl(z,Ol,Om)*1.e6)-5.




def tab( Olmin,Olmax,Ommin,Ommax,Oltaille,Omtaille ):
   Ol = linspace(Olmin,Olmax,Oltaille)
   Om = linspace(Ommin,Ommax,Omtaille)
   muth = np.zeros( len(z))
   chi2 = np.zeros( (Oltaille,Omtaille))
   for i in xrange(Oltaille):
       for j in xrange(Omtaille):
           print i,j
           for k in xrange( len(z)):
               muth[k]= mutheorique(z[k], Ol[i],Om[j])
               chi2[i,j] = sum( ( mu - muth )**2/dmu**2)
               #print chi2[i,j]
   return chi2


def thefunct(z,pars):
    res=np.zeros(z.size)
    for i in np.arange(np.size(z)):
        res[i]=mutheorique(z[i],pars[0],pars[1])
    return(res)

def lnprob(thepars, xvalues, yvalues, errors):
    chi = (yvalues-thefunct(xvalues,thepars))/errors
    chi2=sum(chi**2)/2.0
    return -chi2


nwalkers=4
ndim=2
p0=np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[z, mu, dmu],threads=4)

pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()

sampler.run_mcmc(pos, 1000)
chains=sampler.chain
fractions=sampler.acceptance_fraction


best=where(fractions == max(fractions))

bestwalker=best[0]



clf()
for i in arange(ndim):
    subplot(2,2,i)
    a=hist(reshape(chains[bestwalker,:,i],size(chains[bestwalker,:,i])),50)


clf()
xlim(0,1.5)
ylim(0,1.5)
plot(chains[bestwalker,:,1],chains[bestwalker,:,0],'r,')

contourf(



