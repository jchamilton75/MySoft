import healpy as hp

nside=256
ell=np.arange(3*nside)

sig=0.1
l_low=5
l_mid=20
l_high=60
cl_low=exp(-(ell-l_low)**2/(2*sig**2))
cl_mid=exp(-(ell-l_mid)**2/(2*sig**2))
cl_high=exp(-(ell-l_high)**2/(2*sig**2))
map_low=hp.synfast(cl_low,nside)
map_mid=hp.synfast(cl_mid,nside)
map_high=hp.synfast(cl_high,nside)

hp.mollview(map_low)
hp.mollview(map_mid)
hp.mollview(map_high)

figure()
plot(ell,cl_low,lw=3)
xlim(0,100)
ylim(0,1.2)
xlabel('$\ell$',fontsize=20)
ylabel('$C_\ell$',fontsize=20)

figure()
plot(ell,cl_mid,lw=3)
xlim(0,100)
ylim(0,1.2)
xlabel('$\ell$',fontsize=20)
ylabel('$C_\ell$',fontsize=20)

figure()
plot(ell,cl_high,lw=3)
xlim(0,100)
ylim(0,1.2)
xlabel('$\ell$',fontsize=20)
ylabel('$C_\ell$',fontsize=20)

