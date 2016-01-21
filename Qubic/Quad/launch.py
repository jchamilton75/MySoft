from pyquad import pyquad
import glob
from pysimulators import FitsArray

while True:
    execfile('/Users/hamilton/Python/Qubic/pyquad/test_fixedptg.py')

a=np.loadtxt('./cl_r=0.1bis2.txt')
ell=a[:,0]
ctt=np.concatenate([[0,0],a[:,1]*1e12*2*np.pi/(ell*(ell+1))])
cee=np.concatenate([[0,0],a[:,2]*1e12*2*np.pi/(ell*(ell+1))])
cte=np.concatenate([[0,0],a[:,4]*1e12*2*np.pi/(ell*(ell+1))])
cbb=np.concatenate([[0,0],a[:,7]*1e12*2*np.pi/(ell*(ell+1))])
ell=np.concatenate([[0,1],ell])
spectra=[ell,ctt,cte,cee,cbb]


files=glob.glob('/Users/hamilton/Qubic/PyQuad/ClRes/Cl_linlog_FullMap_mc10000_noise0.1_corr/clresults*.fits')

import pickle
infile=open('saved_ellrange_Log2.dat', 'rb')
data=pickle.load(infile)
infile.close()
deltaell=data['deltaell']
ellmin=data['ellmin']
ellmax=data['ellmax']
data=0


nbf=len(files)
f0=FitsArray(files[0])
sz=f0.shape
ellval=f0[0,:]
allcl=np.zeros((nbf,sz[1]))
allerr=np.zeros((nbf,sz[1]))
for i in np.arange(nbf):
    data=FitsArray(files[i])
    allcl[i,:]=data[1,:]
    allerr[i,:]=data[2,:]
    pyquad.progress_bar(i,nbf)

mcl=np.mean(allcl,axis=0)
covcl=np.zeros((sz[1],sz[1]))
for i in np.arange(sz[1]):
    mm0=allcl[:,i]-mcl[i]
    for j in np.arange(i,sz[1]):
        covcl[i,j]=np.mean(mm0*(allcl[:,j]-mcl[j]))
        covcl[j,i]=covcl[i,j]

corcl=pyquad.cov2cor(covcl)


clf()
xlabel('$\ell$')
ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')
xlim(2,400)
xscale('log')
for i in np.arange(nbf): plot(ellval,allcl[i,:]*ellval*(ellval+1)/(2*np.pi),'k',alpha=0.1,lw=2)
plot(spectra[0],spectra[4]*(spectra[0]*(spectra[0]+1))/(2*np.pi),lw=3,color='b')
errorbar(ellval,mcl*ellval*(ellval+1)/(2*np.pi),yerr=sqrt(np.diag(covcl))*ellval*(ellval+1)/(2*np.pi),fmt='ro',lw=2)

#clf()
#imshow(corcl,interpolation='nearest',vmin=0,vmax=1)
#colorbar()




