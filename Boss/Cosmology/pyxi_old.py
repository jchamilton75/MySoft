from pylab import *
import numpy as np
import cosmolopy
from scipy import interpolate
from scipy import integrate
import numpy


class xith:
    def __init__(self,cosmo,z,kmax=10,nk=32768):
        self.cosmo=cosmo.copy()
        self.k=np.linspace(0,kmax,nk)
        pk=cosmolopy.perturbation.power_spectrum(self.k,z,**cosmo)
        pk[0]=0
        self.pk=pk
        r=2*pi/kmax*np.arange(nk)
        pkk=self.k*self.pk
        cric=-np.imag(fft(pkk)/nk)/r/2/pi**2*kmax
        cric[0]=0
        h=cosmo['h']
        self.xi=interpolate.interp1d(r*h,cric)

    def __call__(self,x):
        return(self.xi(x))

    def pk(self, k):
        return

    def nr(self,xin):
        x=linspace(0,max(xin),1000)
        y=self(x)*x**2
        theint=zeros(x.size)
        theint[1:]=1+3*integrate.cumtrapz(y,x=x)/x[1:]**3
        ff=interpolate.interp1d(x,theint)
        return(ff(xin))

    def rh_nr(self,threshold=1.01):
        x=np.linspace(10,200,100)
        ff=interpolate.interp1d(self.nr(x)[::-1],x[::-1])
        return(ff(threshold))
        
    def d2(self,xin,min=0.1):
        logx=linspace(np.log10(min),np.log10(np.max(xin)),1000)
        lognr=np.log10(self.nr(10**logx))
        dlogx=logx[1]-logx[0]
        thed2=np.gradient(lognr)/dlogx+3
        ff=interpolate.interp1d(10**logx,thed2,bounds_error=False)
        return(ff(xin))

    def rh_d2(self,threshold=2.97):
        x=np.linspace(10,200,100)
        ff=interpolate.interp1d(self.d2(x),x)
        return(ff(threshold))







