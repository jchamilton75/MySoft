

####
deltaphi = np.array([15., 25., 35., 45.])

#### 1./sqrt(omega)
un_sqrtomega = np.array([1.177, 1.081, 0.961, 0.866])
un_sqrtomega = un_sqrtomega / np.mean(un_sqrtomega)

#### Omega
omega = 1./un_sqrtomega**2
omega = omega / np.mean(omega)


#### eta
eta = np.array([1.014, 1.011, 0.994, 0.985])
eta = eta / np.mean(eta)

#### sigma2 is NET**2 Omega in Battistelli et al formula
sigma2 = np.array([0.770, 0.866, 1.107, 1.305])
sigma2 = sigma2 / np.mean(sigma2)

#### Error from Battistelli if sigma2 is NET**2 Omega
err = sigma2 * eta / un_sqrtomega
err = err / np.mean(err)


clf()
xlim(15, 45)
ylim(0.6,1.5)
plot(deltaphi, un_sqrtomega, '--', label = '$1/sqrt(\Omega)$')
plot(deltaphi, eta, '--', label='$\eta$')
plot(deltaphi, sigma2, '--', label ='$\sigma^2$')
plot(deltaphi, err, label = '$\sigma^2 \eta / sqrt(\Omega)$', lw=3)
legend(fontsize=10, loc='lower right')


import healpy as hp
from qubic import read_spectra
s = read_spectra(0)
ns = 128
npix = 12*ns**2
fskyvals = np.linspace(0.005, 0.02, 5)
omegavals = 4*np.pi*fskyvals
npixvals = npix * fskyvals
nbmc = 100

ell = np.arange(2*ns+1)
Dlsvals = np.zeros((len(fskyvals), 2*ns+1, nbmc))

for i in xrange(len(fskyvals)):
    for num in xrange(nbmc):
        print(i,num)
        maps = hp.synfast(s, ns, new=True)
        mapi = maps[0]
        mapi[npixvals[i]:] = 0
        Dlsvals[i,:,num] = hp.anafast(mapi,lmax = 2*ns)*ell*(ell+1)/2/np.pi
        
rmsDls = np.std(Dlsvals,axis=2)

clf()
imshow(rmsDls, aspect='auto',interpolation='nearest')


