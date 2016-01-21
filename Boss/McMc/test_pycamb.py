from pylab import *
import pycamb

lmax=2000
ns_values = [0.8,0.9,1.0,1.1]
ell = pylab.arange(1,lmax)
for ns in ns_values:
    T,E,B,X = pycamb.camb(lmax, scalar_index=ns)
    pylab.semilogx(ell,T,label="%.2f"%ns)
legend()
xlabel("$\ell$", fontsize=20)
ylabel("$\ell (\ell+1) C_\ell / 2\pi \quad [\mu K^2]$", fontsize=20)
title("Varying Spectral Index $n_s$")
xlim(1,2000)

clf()
lmax=2000
rvals = [0.01,0.05,0.1,0.5]
ell = pylab.arange(1,lmax)
for rv in rvals:
    T,E,B,X = pycamb.camb(lmax, tensor_ratio=rv,WantTensors=True,DoLensing=True)
    pylab.loglog(ell,B,label="%.2f"%rv)
legend()
xlabel("$\ell$", fontsize=20)
ylabel("$\ell (\ell+1) C_\ell / 2\pi \quad [\mu K^2]$", fontsize=20)
title("Varying Tensor-to-scalar ration $r$")
xlim(1,2000)







#### Compare angular diameters
import cosmolopy
import cosmo_utils
nz=10000
zmax=2.
z=linspace(0,zmax,nz)

cosmo=cosmolopy.fidcosmo.copy()
cosmo['h']=0.6704
cosmo['Y_He']=0.247710
obh2=0.022032
onh2=0.000645
och2=0.120376-onh2
cosmo['omega_M_0']=(och2+obh2+onh2)/cosmo['h']**2
cosmo['omega_lambda_0']=1.-cosmo['omega_M_0']
cosmo['omega_k_0']=0
cosmo['omega_b_0']=obh2/cosmo['h']**2
cosmo['omega_n_0']=onh2/cosmo['h']**2
cosmo['n']=0.9619123
zstar=1090.49
rs=cosmo_utils.rs(zd=zstar,**cosmo)
da=cosmo_utils.angdist(zstar,zres=0.001,**cosmo)
rs/(1+zstar)/da*100



omegab=cosmo['omega_b_0']
omegac=cosmo['omega_M_0']-cosmo['omega_b_0']-cosmo['omega_n_0']
omegav=cosmo['omega_lambda_0']
omegan=cosmo['omega_n_0']
H0=cosmo['h']*100
Num_Nu_massless=3.046/3*2
Num_Nu_massive=3.046/3
omegak=1.-omegav-omegac-omegab-omegan
w_lam=cosmo['w']

100*pycamb.cosmomctheta(omegab=omegab,omegac=omegac,omegav=omegav,omegan=omegan,H0=H0,Num_Nu_massless=Num_Nu_massless,Num_Nu_massive=Num_Nu_massive,omegak=omegak,w_lam=w_lam)



##### with code from CAMB
obh2=cosmo['omega_b_0']*cosmo['h']**2
och2=(cosmo['omega_M_0']-cosmo['omega_b_0']-cosmo['omega_n_0'])*cosmo['h']**2
zstar=1048*(1+0.00124*obh2**(-0.738))*(1+(0.0783*obh2**(-0.238)/(1+39.5*obh2**0.763))*(och2+obh2)**(0.560/(1+21.1*obh2**1.81)))

def Nu_rho(am):
    am_min=0.01
    am_minp=am_min*1.1
    am_max=600.
    am_maxp=am_max*0.9
    const2=5./7/np.pi**2
    const=7./120*np.pi**4
    zeta3  = 1.2020569031595942853997
    zeta5  = 1.0369277551433699263313
    nrhopn=2000
    dlnam=-(log(am_min/am_max))/(nrhopn-1)
    if am <= am_minp:
        rhonu=1.+const2*am**2
        return(rhonu)
    elif am >= am_maxp:
        rhonu=3/(2*const)*(zeta3*am + (15*zeta5)/2/am)
        return(rhonu)
    elif:
        d=np.log(am/am_min)/dlnam+1
        i=int(d)
        d=d-i
        rhonu=r1(i)+d*(dr1(i)+d*(3.*(r1(i+1)-r1(i))-2.*dr1(i)-dr1(i+1)+d*(dr1(i)+dr1(i+1)+2.*(r1(i)-r1(i+1)))))
        return(exp(rhonu))


r1[] -> modules.f90
dr1[]



def dsoundda(a,**cosmo):
    R=3.0*a*cosmo['omega_b_0']*cosmo['h']**2
    cs=1.0/np.sqrt(3*(1+R))
    return(dtauda(a)*cs)













