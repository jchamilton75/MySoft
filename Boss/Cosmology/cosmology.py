import numpy as N
from pylab import *
import warnings
from scipy import integrate
from scipy import interpolate
import numpy


def properdistance(z,omegam=0.3,omegax=0.7,w0=-1,w1=0,wz=None):
    """
    Gives the proper distance in the defined cosmology
    The c/Ho factor is ommited
    Returns dist(z), w(z), omegax(z), H(z), curvature
    """
    # if no wz on input the calculate it from w0 and w1
    if wz is None: wz=w0+(z*1./(1.+z))*w1
    # calculate evolution of omegax accounting for its equation of state
    omegaxz=zeros(z.size)
    omegaxz[0]=omegax
    omegaxz[1:z.size]=omegax*exp(3*integrate.cumtrapz((1.+wz)/(1.+z),x=z))

    # curvature
    omega=omegam+omegax
    
    # calculation of H(z)
    hz=sqrt((1.-omega)*(1+z)**2+omegaxz+omegam*(1+z)**3)

    # calculate chi
    chi=zeros(z.size)
    chi[1:z.size]=integrate.cumtrapz(1./hz,x=z)

    #calculate proper distance
    if omega>1: curv=1
    if omega<1: curv=-1
    if omega==1: curv=0
    kk=abs(1.-omega)
    if curv==1: dist=sin(sqrt(kk)*chi)/sqrt(kk)
    if curv==-1: dist=sinh(sqrt(kk)*chi)/sqrt(kk)
    if curv==0: dist=chi

    return dist,wz,omegaxz,hz,curv,chi

def get_dist(z,type='prop',params=[0.3,0.7,-1,0],h=0.7,wz=None):
    """
    Returns distances in Gpc/h in the defined cosmology
    type can be :
       prop : proper distance
       dl   : Luminosity distance
       dang : Angular distance
       dangco : Comoving angular distance
       wz : equation of state as a function of z
       omegaxz : omegax(z)
       hz : h(z)
       curv : curvature
       vco : Comoving volume
       rapp : proper*H(z)
    """
    c=3e8       #m.s-1
    H0=1000*1000*h*100  #m.s-1.Gpc-1
    omegam=params[0]
    omegax=params[1]
    w0=params[2]
    w1=params[3]

    zvalues=linspace(0.,z.max()*1.2,len(z)*10)

    theomegam = omegam
    dist,wz,omegaxz,hz,curv,chi=properdistance(zvalues,theomegam,omegax,w0=w0,w1=w1,wz=wz)

    if type=='prop':
        res=dist*c/H0
    elif type=='comoving':
        res=chi*c/H0
    elif type=='comoving_transverse':
        res=dist*c/H0
    elif type=='dl':
        res=dist*(1+zvalues)*c/H0
    elif type=='dang':
        res=dist/(1+zvalues)*c/H0
    elif type=='dangco':
        res=dist*c/H0
    elif type=='wz':
        res=wz
    elif type=='omegaxz':
        res=omegaxz
    elif type=='hz':
        res=hz*H0
    elif type=='curv':
        res=curv
    elif type=='vco':
        res=dist**2/hz*(c/H0)**2/H0
    elif type=='rapp':
        res=dist*hz*c
    else:
        print "This type does not exist:",type
        res=-1

    f=interpolate.interp1d(zvalues,res)
    return f(z)

