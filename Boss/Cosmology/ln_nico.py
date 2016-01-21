
####### Pour pouvoir utiliser le code de nico
import scipy as sp
from numpy import random
kmax=100.
rmax = 1600.
nn=int(kmax*rmax/2./sp.pi)
dk = kmax/nn
dr = rmax/nn

z=0
k, pk = get_pk(params, z, kmax=100, nk=nn)
pk *= exp(-k**2/2/10**2)

r = dr*sp.arange(nn)
w=r>rmax/2
r[w]-=rmax

xi_ln = -1./2/sp.pi**2*sp.imag(sp.fft(k*pk))*dk
w=r!=0
xi_ln[w]/=r[w]
xi_ln[0]=1./2/sp.pi**2*sp.sum(k**2*pk)*dk

p1d_ln = dr*sp.real(sp.fft(xi_ln))
x0 = 0.5+sp.sqrt(0.25+xi_ln[0])
xi_g = sp.log(1+xi_ln/x0)

p1d_g = dr*sp.real(sp.fft(xi_g))

del_j = random.normal(size=nn)
del_k = sp.fft(del_j)
del_k*=sp.sqrt(p1d_g*dk/2/sp.pi*nn)
del_j=sp.exp(sp.real(sp.ifft(del_k)))
del_k = sp.fft(del_j-sp.mean(del_j))
xi_meas = sp.real(sp.ifft(abs(del_k)**2)/nn)

clf()
plot(k, p1d_g)
xscale('log')
yscale('log')





def lnsim_1d(k, pk, xmax, nn, doplot=False,padding=None):
	### 1/ input is camb P(k)
	### 2/ get xi(r) with massive zero-padding
	r, xi = pk3d2xi(k,pk, padding=padding)
	dr = r[1]-r[0]
	### 3/ C(r) = log(1+xi(r)/thenorm)
	thenorm = 0.5+np.sqrt(0.25+xi[0])
	cr = log(1+xi/thenorm)
	### 4/ the final sampling and corresponding k
	xvals = linspace(0,xmax, nn)
	kvals = np.fft.fftfreq(nn,xvals[1]-xvals[0])*2*pi
	### 4/ P1D from C(r)
	kout, thepk1d = xi2pk1d(r, cr, padding=None, kout=kvals)
	thepk1d[thepk1d <0] =0
	### 5/ exp(gaussian field)
	gaussnoise=np.random.randn(nn)
	ftgaussnoise = fft(gaussnoise)
	dk = 2 * pi / xmax
	ftgaussnoisenew = ftgaussnoise * np.sqrt(thepk1d*dk/2/pi*nn)
	signal_noexp = np.real(ifft(ftgaussnoisenew))
	signal = np.exp(signal_noexp)
	return xvals, signal


















