import cosmolopy
import cosmolopy.distance as cd


qso_zmin = 2.2
qso_zmax = 2.8
area =5800 #deg2



zvals = np.linspace(0,1e5,1e5)
lcdm = cd.set_omega_k_0({'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.7})
d_lcdm = cd.comoving_distance_transverse(zvals, **lcdm)

clf()
plot(zvals,d_lcdm)
xscale('log')

d_bb = np.max(d_lcdm)
d_min = cd.comoving_distance_transverse(qso_zmin, **lcdm)
d_min = cd.comoving_distance_transverse(qso_zmin, **lcdm)





