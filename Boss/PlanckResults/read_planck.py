from matplotlib import rc
rc('text', usetex=False)

rep='/Volumes/Data/ChainsPlanck/PLA/base/planck_lowl_lowLike/'

planck_chains=np.loadtxt(rep+'base_planck_lowl_lowLike_1.txt')
names=np.loadtxt(rep+'base_planck_lowl_lowLike.paramnames',dtype='str',usecols=[0])

planck=dict([names[i],planck_chains[:,i+2]] for i in range(np.size(names)))




h0wmap=np.loadtxt('/Users/hamilton/SDSS/Chains/ChainsWMAP/FlatLCDM/H0')

clf()
hist(h0wmap[:,1],100,range=[60,80],alpha=0.5)
hist(planck['H0*'],100,range=[60,80],alpha=0.5)

clf()
plot(planck['H0*'],planck['omegam*'],'o')

clf()
a,x,y=np.histogram2d(planck['H0*'],planck['omegam*'],[100,100])
extent=[y[0],y[-1],x[-1],x[0]]
imshow(a,interpolation='nearest',extent=extent,aspect='auto',origin='lower')
xlabel('$\Omega_m$')
ylabel('$H_0$')
colorbar()
savefig('toto.pdf')

clf()
a=a/np.max(a)
contour(y[1:],x[1:],a,[0.9,0.5,0.1])
xlabel('$\Omega_m$')
ylabel('$H_0$')

#http://stats.stackexchange.com/questions/12726/calculating-2d-confidence-regions-from-mcmc-samples
import scipy.stats
gkde=scipy.stats.gaussian_kde([planck['H0*'],planck['omegam*']])

nb=15
szx=std(planck['H0*'])/nb
szy=std(planck['omegam*'])/nb
xx,yy = mgrid[min(planck['H0*']):max(planck['H0*']):szx, min(planck['omegam*']):max(planck['omegam*']):szy]
z = array(gkde.evaluate([xx.flatten(),yy.flatten()])).reshape(xx.shape)

from scipy import integrate
from scipy import interpolate
sz=np.sort(z.flatten())[::-1]
cumsz=integrate.cumtrapz(sz)
cumsz=cumsz/max(cumsz)
f=interpolate.interp1d(cumsz,arange(size(cumsz)))
indices=[floor(f(0.68)),floor(f(0.95))]
vals=sz[indices]

clf()
a=0.8
contour(xx, yy, z, vals,lw=2,colors='black')
contourf(xx, yy, z, [vals[1],vals[0],max(sz)],colors=['LightBlue','DodgerBlue'],alpha=a)
ylabel('$\Omega_m$')
xlabel('$H_0$')




######################################################
from matplotlib import rc
rc('text', usetex=False)

om=np.loadtxt('/Volumes/Data/ChainsWMAP/FlatLCDM/omegam',usecols=[1])
h0=np.loadtxt('/Volumes/Data/ChainsWMAP/FlatLCDM/H0',usecols=[1])
ns=np.loadtxt('/Volumes/Data/ChainsWMAP/FlatLCDM/ns002',usecols=[1])



clf()
xlabel('$\Omega_m$')
ylabel('$H_0$')
xlim(0.25,0.40)
ylim(62,72.5)
scatter(om,h0,marker='o',c=ns,lw=0,alpha=1,s=8)
cbar=colorbar()
cbar.set_label(r'$n_s$')
savefig('om_h_ns_scaled.png')


clf()
xlabel('$\Omega_m$')
ylabel('$H_0$')
scatter(om,h0,marker='o',c=ns,lw=0,alpha=1,s=8)
cbar=colorbar()
cbar.set_label(r'$n_s$')
savefig('om_h_ns.png')







