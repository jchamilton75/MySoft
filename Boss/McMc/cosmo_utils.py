import numpy as np
import scipy.integrate
import scipy.interpolate
import random
import string
import pycamb
import sys




############################## This is Eisenstein & Hu slightly modified according to Jim Rich https://trac.sdss3.org/wiki/BOSS/LyaForestsurvey/EisensteinHu
##################################################################################################################
def RR(z,ob,h,theta):
    return(31.492*ob*h**2*theta**(-4)*((1+z)/1000)**(-1))


def rs(zd=1059.25,**cosmo):
    o0=cosmo['omega_M_0']-cosmo['omega_n_0']     ### need to remove omega_neutrino as they are relativistic (massless at high z)
    h=cosmo['h']
    ob=cosmo['omega_b_0']
    theta=2.7255/2.7
    zeq=2.5*1e4*o0*h**2*theta**(-4)
    keq=7.46*0.01*o0*h**2*theta**(-2)
    b1=0.313*(o0*h**2)**(-0.419)*(1+0.607*(o0*h**2)**0.674)
    b2=0.238*(o0*h**2)**0.223
    # This is E&H zdrag
    #zd=1291.*(o0*h**2)**0.251/(1+0.659*(o0*h**2)**0.828)*(1+b1*(ob*h**2)**b2)
    # We use instead the value coming from CAMB and for Planck+WP+Highl Cosmology as suggested by J.Rich (it depends mostly on atomic physics) => zd=1059.25
    req=RR(zeq,ob,h,theta)
    rd=RR(zd*1.,ob,h,theta)
    rs=(2./(3*keq))*np.sqrt(6./req)*np.log((np.sqrt(1+rd)+np.sqrt(rd+req))/(1+np.sqrt(req)))
    return(rs)
##################################################################################################################

def random_string(nchars):
    lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(nchars)]
    str = "".join(lst)
    return(str)

def thetamc(**cosmo):
    omegab=cosmo['omega_b_0']
    omegac=cosmo['omega_M_0']-cosmo['omega_b_0']-cosmo['omega_n_0']
    omegav=cosmo['omega_lambda_0']
    omegan=cosmo['omega_n_0']
    H0=cosmo['h']*100
    Num_Nu_massless=cosmo['Num_Nu_massless']
    Num_Nu_massive=cosmo['Num_Nu_massive']
    omegak=1.-omegav-omegac-omegab-omegan
    w_lam=cosmo['w']
    thetamc=pycamb.cosmomctheta(omegab=omegab,omegac=omegac,omegav=omegav,omegan=omegan,H0=H0,Num_Nu_massless=Num_Nu_massless,Num_Nu_massive=Num_Nu_massive,omegak=omegak,w_lam=w_lam)
    return(thetamc)

def rs_zstar_camb(**cosmo):
    omegab=cosmo['omega_b_0']
    omegac=cosmo['omega_M_0']-cosmo['omega_b_0']-cosmo['omega_n_0']
    omegav=cosmo['omega_lambda_0']
    omegan=cosmo['omega_n_0']
    H0=cosmo['h']*100
    Num_Nu_massless=cosmo['Num_Nu_massless']
    Num_Nu_massive=cosmo['Num_Nu_massive']
    omegak=1.-omegav-omegac-omegab-omegan
    w_lam=cosmo['w']
    rs=pycamb.cosmomcrs_zstar(omegab=omegab,omegac=omegac,omegav=omegav,omegan=omegan,H0=H0,Num_Nu_massless=Num_Nu_massless,Num_Nu_massive=Num_Nu_massive,omegak=omegak,w_lam=w_lam)
    return(rs)

def rs_zdrag_camb(**cosmo):
    omegab=cosmo['omega_b_0']
    omegac=cosmo['omega_M_0']-cosmo['omega_b_0']-cosmo['omega_n_0']
    omegav=cosmo['omega_lambda_0']
    omegan=cosmo['omega_n_0']
    H0=cosmo['h']*100
    Num_Nu_massless=cosmo['Num_Nu_massless']
    Num_Nu_massive=cosmo['Num_Nu_massive']
    omegak=1.-omegav-omegac-omegab-omegan
    w_lam=cosmo['w']
    rs=pycamb.cosmomcrs_zdrag(omegab=omegab,omegac=omegac,omegav=omegav,omegan=omegan,H0=H0,Num_Nu_massless=Num_Nu_massless,Num_Nu_massive=Num_Nu_massive,omegak=omegak,w_lam=w_lam)
    return(rs)

def rs_zdrag_fast_camb(**cosmo):
    omegab=cosmo['omega_b_0']
    omegac=cosmo['omega_M_0']-cosmo['omega_b_0']-cosmo['omega_n_0']
    omegav=cosmo['omega_lambda_0']
    omegan=cosmo['omega_n_0']
    H0=cosmo['h']*100
    Num_Nu_massless=cosmo['Num_Nu_massless']
    Num_Nu_massive=cosmo['Num_Nu_massive']
    omegak=1.-omegav-omegac-omegab-omegan
    w_lam=cosmo['w']
    rs=pycamb.cosmomcrs_zdrag_fast(omegab=omegab,omegac=omegac,omegav=omegav,omegan=omegan,H0=H0,Num_Nu_massless=Num_Nu_massless,Num_Nu_massive=Num_Nu_massive,omegak=omegak,w_lam=w_lam)
    return(rs)


    
def e_z(z,**cosmo):
    omegam=cosmo['omega_M_0']
    omegax=cosmo['omega_lambda_0']
    w0=cosmo['w']
    h=cosmo['h']
    omegak=cosmo['omega_k_0']
    omegaxz=omegax*(1+z)**(3+3*w0)
    e_z=np.sqrt(omegak*(1+z)**2+omegaxz+omegam*(1+z)**3)
    return(e_z)

def inv_e_z(z,**cosmo):
    return(1./e_z(z,**cosmo))

def hz(z,**cosmo):
    return(cosmo['h']*e_z(z,**cosmo))

def correction_horizon(**cosmo):
    num=10000
    zvals=np.linspace(0,100000.,num)
    index=np.arange(num)
    inv_e_z_far=inv_e_z(zvals,**cosmo)
    index1089=int(np.round(np.interp(1089.,zvals,index)))
    intdeno=scipy.integrate.trapz(inv_e_z_far[index1089:],zvals[index1089:])
    intnum=scipy.integrate.trapz(1./np.sqrt(cosmo['omega_M_0']*(1+zvals[index1089:])**3),zvals[index1089:])
    corr=intnum/intdeno
    return(corr)

def propdist(z,zres=0.001,**cosmo):
    ### z range for integration
    zmax=np.max(z)
    if zmax < zres:
        nb=101
    else:
        nb=zmax/zres+1
    zvals=np.linspace(0.,zmax,nb)
    ### integrate
    cumulative=np.zeros(nb)
    cumulative[1:]=scipy.integrate.cumtrapz(1./e_z(zvals,**cosmo),zvals)
    ### interpolation to input z values
    propdist=np.interp(z,zvals,cumulative)
    ### curvature
    omega=cosmo['omega_M_0']+cosmo['omega_lambda_0']
    k=np.abs(1-omega)
    if omega == 1:
        propdist=propdist
    elif omega < 1:
        propdist=np.sinh(np.sqrt(k)*propdist)/np.sqrt(k)
    elif omega > 1:
        propdist=np.sin(np.sqrt(k)*propdist)/np.sqrt(k)
    ### returning
    return(propdist*2.99792458e5/100/cosmo['h'])

def lumdist(z,zres=0.001,accurate=False,**cosmo):
    return(propdist(z,zres=zres,accurate=accurate,**cosmo)*(1+z))

def angdist(z,zres=0.001,accurate=False,**cosmo):
    return(propdist(z,zres=zres,accurate=accurate,**cosmo)/(1+z))

def progress_bar(i,n):
    if n != 1:
        ntot=50
        ndone=ntot*i/(n-1)
        a='\r|'
        for k in np.arange(ndone):
            a += '#'
        for k in np.arange(ntot-ndone):
            a += ' '
        a += '| '+str(int(i*100./(n-1)))+'%'
        sys.stdout.write(a)
        sys.stdout.flush()
        if i == n-1:
            sys.stdout.write(' Done \n')
            sys.stdout.flush()
            
