







def dist(xa,x):
	return np.sqrt((x[0,:]-xa[0])**2 + (x[1,:]-xa[1])**2)

def angle(xa,x):
	return np.arctan2(x[0,:]-xa[0],x[1,:]-xa[1])

def phase(xa,x,nu):
	distance = dist(xa,x)
	return (2*np.pi*nu/3e8*distance) % (2*np.pi)

def dephasage(xa,x,nu):
	phi = phase(xa, x, nu)
	return cos(phi)+complex(0,1)*sin(phi)

def beam(ang):
	return np.exp(-0.5*np.degrees(ang)**2/(13./2.35)**2)

def field(xA, x, nu, einit):
	angA = angle(xA,x)
	phaseA = phase(xA, x, nu)
	expphiA = dephasage(xA, x, nu)
	eA = einit*expphiA*beam(angA)
	return eA

def power(xA, xB, x, nu, einit, random_phase=None):
	eA = field(xA, x, nu, einit)
	eB = field(xB, x, nu, einit)
	if random_phase is None:
		ptot = np.abs(eA + eB)
	else:
		allptot = np.zeros((random_phase, len(einit)))
		for i in xrange(random_phase):
			phases = np.random.rand(len(einit))*2*np.pi
			random_field = (np.cos(phases)+np.complex(0,1)*np.sin(phases))
			allptot[i,:] = np.abs(eA + eB*random_field)
		ptot = np.mean(allptot,axis=0)
	return ptot



#### Frequency
nu = 150e9 ### GHz
#### Thetavals
thrange = 90.
nth = 10000
thvals = np.radians(np.linspace(-thrange,thrange, nth))


#### Now loop with D
def do_all(nu, D, thvals, plane=True, dhorns=0.01, random_phase=None, doplot=False):
	x1 = np.array([-dhorns/2, -D])
	x2 = np.array([dhorns/2, -D])
	if(plane):
		xinit = np.array([np.tan(thvals)*D,np.zeros(nth)])
		xx = xinit[0,:]
	else:
		xinit = np.array([np.sin(thvals)*D, np.cos(thvals)*D-D])
		xx = xinit[0,:]
	Pinit = np.zeros(nth)+1
	phi_init = np.random.rand(nth)*2*np.pi
	einit = np.sqrt(Pinit)*(np.cos(phi_init)+np.complex(0,1)*np.sin(phi_init))
	ptot = power(x1, x2, xinit, nu, einit, random_phase=random_phase)
	if doplot:
		clf()
		plot(x1[0],x1[1],'ro')
		plot(x2[0],x2[1],'bo')
		plot(xinit[0,:], xinit[1,:])
		xlim(-D*1.2,D*1.2)
		ylim(-D*1.2,D*0.2)
	return ptot


dhorns=0.2
clf()
xlim(-20,20)
title('Dhorns = {}m'.format(dhorns))
xlabel('Theta [deg]')
ylabel('Synthesized Beam')
plot(np.degrees(thvals), do_all(nu,1000, thvals, plane=True, dhorns=dhorns), color=color[i],
		label='Distance = 1000m'.format(D))
plot(np.degrees(thvals), 2*beam(thvals), 'k--', label='Primary_beam')
legend()



dhorns=0.2
clf()
xlim(-20,20)
title('Dhorns = {}m'.format(dhorns))
xlabel('Theta [deg]')
ylabel('Synthesized Beam')
Dvals = [10., 1., 0.5]
color= ['blue', 'red', 'green']

for i in xrange(len(Dvals)):
	D=Dvals[i]
	print(D)
	plot(np.degrees(thvals), do_all(nu,D, thvals, plane=True, dhorns=dhorns), color=color[i],
		label='Distance = {}m'.format(D))
plot(np.degrees(thvals), 2*beam(thvals), 'k--', label='Primary_beam')
legend()
savefig('ptsrc_alldist_dhorns{}.png'.format(dhorns))



clf()
xlim(-20,20)
title('Dhorns = {}m'.format(dhorns))
xlabel('Theta [deg]')
ylabel('Synthesized Beam')
Dvals = [10, 1., 0.1]
color= ['blue', 'red', 'green']

for i in xrange(len(Dvals)):
	D=Dvals[i]
	print(D)
	plot(np.degrees(thvals), do_all(nu,D, thvals, 
		plane=True, dhorns=dhorns, random_phase=10000), color=color[i],
		label='Distance = {}m'.format(D))
legend()



###############################################################

## Configuration de Paolo et Silvia

D = 100.
ptot1 = do_all(nu,D, thvals, plane=True, dhorns=0.01)
ptot2 = do_all(nu,D, thvals, plane=True, dhorns=0.2)
clf()
title('Point Source : distance = {}m'.format(D))
plot(np.degrees(thvals), ptot1, 'b', label='Dhorns = 1cm')
plot(np.degrees(thvals), ptot2, 'r', label='Dhorns = 20cm')
plot(np.degrees(thvals), 2*beam(thvals), 'k--', label='Primary_beam')
xlim(-20,20)
xlabel('Theta [deg]')
ylabel('Synthesized Beam')
legend()
savefig('ptsrc_dist_{}m.png'.format(D))

D=1.
clf()
do_all(nu,D, thvals, plane=True, dhorns=0.01, doplot=True)

D = 1.
ptot1 = do_all(nu,D, thvals, plane=True, dhorns=0.01, random_phase=1000)
ptot2 = do_all(nu,D, thvals, plane=True, dhorns=0.2, random_phase=1000)
title('Extended Uniform Source : distance = {}m'.format(D))
clf()
plot(np.degrees(thvals), ptot1, 'b', label='Dhorns = 1cm')
plot(np.degrees(thvals), ptot2, 'r', label='Dhorns = 20cm')
thebeam = beam(thvals)
thebeam = thebeam*np.sum(ptot1)/np.sum(thebeam)
plot(np.degrees(thvals), thebeam, 'k--', label='Primary_beam')
legend()
xlim(-20,20)
xlabel('Theta [deg]')
ylabel('Synthesized Beam')
legend()
savefig('extendedsrc_dist_{}m.png'.format(D))






