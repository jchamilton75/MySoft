import numpy as np
import matplotlib.pyplot as plt
from Homogeneity import fitting

ndim = 10

means = np.random.rand(ndim)

cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov,cov)

x = np.arange(10)

data = x + means + 5

cho = np.linalg.cholesky(cov)

corr_data = np.dot(cho.T,data)

##### see if cholesky is correct:
print 'Accuracy of Cholesky Decomposition'
print np.dot(cho,cho.T.conj()) == cov

def model(x,pars):
    f=np.poly1d(pars)
    return(f(x))

guess1 = [np.mean(data),np.mean(data)]
guess2 = [np.mean(corr_data),np.mean(corr_data)]

res = fitting.dothefit(x,data,cov,guess1,functname=model,method='minuit')

corr_res = fitting.dothefit(x,corr_data,cov,guess2,functname=model,method='minuit')



plt.figure()
plt.errorbar(x,data,yerr=np.diag(cov),color='b',fmt='o-',label='data' )
plt.errorbar(x,corr_data,yerr=np.diag(cov),color='g',fmt='o-',label='corr data')

plt.plot(x,model(x,res[1]),'r-')
plt.plot(x,model(x,corr_res[1]),'r--')

plt.legend(loc=2,numpoints=1)
plt.show()


#### Building a covariance matrix
line = 1./(np.linspace(1,10,ndim))
cov = np.zeros((ndim,ndim))
for i in xrange(ndim):
    for j in xrange(ndim):
        cov[i,j] = line[np.abs(i-j)]

clf()
imshow(cov, interpolation='nearest')

#### Cholesky decomposition and Inverse
cho = np.linalg.cholesky(cov)
invcov = np.linalg.inv(cov)

#### model
def true_model(x, pars):
    f=np.poly1d(pars)
    return(f(x))
truepars = np.array([1.,5.])

def generate_covnoise(cho,nb):
    gnoise = np.random.randn(nb)
    return np.dot(cho,gnoise)

#### 1/ Check that noise is generated according to cov mat
nbmc = 100000
realisations = np.zeros((ndim,nbmc))
for i in xrange(nbmc):
    print(i)
    realisations[:,i] = generate_covnoise(cho,ndim)

#### calculate the MC covariance matrix    
covmc = np.cov(realisations)
     
clf()
subplot(1,2,1)
imshow(cov, interpolation = 'nearest')
title('Input')
colorbar()
subplot(1,2,2)
imshow(covmc, interpolation='nearest')
title('MC based')
colorbar()

### OK the covariance matrix of the simulated data matches the input one



### 2/ do a MC with fitting to look at chi2 histograms
guess = truepars
nbmc = 1000
doplot = False
chi2_uncorr = np.zeros(nbmc) 
chi2_corr = np.zeros(nbmc)

for i in xrange(nbmc):
    print(i)
    uncorr_noise = np.random.randn(ndim) * np.sqrt(np.diag(cov))
    corr_noise = generate_covnoise(cho,ndim)
    nonoise_data = true_model(x, truepars)
    uncorr_data = nonoise_data + uncorr_noise
    corr_data = nonoise_data + corr_noise
    res = fitting.dothefit(x,uncorr_data,np.diag(np.diag(cov)),guess,functname=true_model,method='minuit')
    thechi2 = np.sum((uncorr_data - true_model(x,res[1]))**2 / np.diag(cov))
    chi2_uncorr[i] = thechi2
    corr_res = fitting.dothefit(x,corr_data,cov,guess,functname=true_model,method='minuit')
    resid = corr_data - true_model(x,corr_res[1])
    thechi2 = np.dot(resid.T, np.dot(invcov, resid))
    chi2_corr[i] = thechi2
    if doplot:
        clf()
        plt.errorbar(x,uncorr_data,yerr=np.diag(cov),color='b',fmt='o-',label='data' )
        plt.errorbar(x,corr_data,yerr=np.diag(cov),color='g',fmt='o-',label='corr data')
        plt.plot(x,model(x,res[1]),'r-')
        plt.plot(x,model(x,corr_res[1]),'r--')
        plt.legend(loc=2,numpoints=1)
        plt.show()
        

clf()
hist(chi2_uncorr,bins=20, range=[0,20],color='red', alpha=0.2)
hist(chi2_corr,bins=20, range=[0,20], color='blue', alpha=0.2)

clf()
hist(chi2_uncorr/8,bins=20, range=[0,2],color='red', alpha=0.2)
hist(chi2_corr/8,bins=20, range=[0,2], color='blue', alpha=0.2)





