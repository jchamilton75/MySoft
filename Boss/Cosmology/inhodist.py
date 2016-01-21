import scipy.integrate
import numpy as np

###################### Initial functions: case of omega(x) fully known (no evolution or a priori evolution)
def deriv(z, x, h, omall, xall, type):
	c=3e8
	H0 = 1000*1000*h*100
	om = np.interp(x, xall, omall)
	ok = 1.-om
	xnew = x/(c/H0)
	if type=='comoving':
		fact=1
	elif type=='comoving_transverse':
		fact = 1./np.sqrt(1+ok*xnew**2)
	else:
		return np.nan
	zprime = np.array(fact*(1 + z) * np.sqrt(1 + z * om))
	return zprime/(c/H0)

def x2z_inho(x, omega, h, type='comoving_transverse'):
	z0 = np.array([0.0])
	zrec = scipy.integrate.odeint(deriv, z0, x,args=(h, omega,x, type))
	return zrec[:,0]




#### Now with the LN lightcones
def deriv_lightcone(z, x, h, lightcone, type):
	print(x)
	c=3e8
	H0 = 1000*h*100
	xnew = x/(c/H0)
	om = lightcone(z,x)*lightcone.omegam
	ok = 1.-om
	if type=='comoving':
		fact=1
	elif type=='comoving_transverse':
		if ok==0:
			fact=1
		elif ok<0:
			fact=1./np.sqrt(1-ok*xnew**2)
		elif ok>0:
			fact = 1./np.sqrt(1+ok*xnew**2)
		else:
			stop
	else:
		return np.nan
	zprime = np.array(fact*(1 + z) * np.sqrt(1 + z * om))
	if np.isnan(zprime): stop
	return zprime/(c/H0)

def x2z_inho_lightcone(lightcone, h, type='comoving_transverse'):
	x = lightcone.xx
	z0 = np.array([0.0])
	zrec = scipy.integrate.odeint(deriv_lightcone, z0, x,args=(h, lightcone, type) )
	zrec = zrec[:,0]
	return zrec
