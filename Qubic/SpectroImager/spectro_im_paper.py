
def get_primbeam(th, lam, fwhmprimbeam_150=14.):
    fwhmprim = 14. * lam / (3e8/150e9)
    primbeam = np.exp(-0.5 * th**2 / (fwhmprim/2.35)**2)
    return primbeam

def give_sbcut(th, dx, lam, sqnh, Df=1., detpos=0., fwhmprimbeam_150=14.):
    primbeam =  get_primbeam(th, lam, fwhmprimbeam_150=fwhmprimbeam_150)
    theth = th - np.degrees(detpos/Df)
    sb = np.sin(sqnh * np.pi * dx / lam * np.radians(theth))**2 / np.sin(np.pi * dx / lam * np.radians(theth))**2
    return sb/np.max(sb)*primbeam


######### QUBIC Case
import qubic
import scipy
inst = qubic.QubicInstrument()
omega150GHz = inst.primary_beam.solid_angle
deltax = inst.horn.spacing  #m
sqnh = 20.
deltanu_nu = 1./sqnh
fnb=5

duration = 2*365*24*3600
NET_25bw_150GHz = 306.
NET_25bw_220GHz = 515.
fsky = 0.01
#omega150GHz_arcmin2 = omega150GHz * (180/np.pi)**2 * 60**2
sky_arcmin2 = fsky * 4 *np.pi * (180/np.pi)**2 * 60**2
eta = 1.6
epsilon = 0.3
muKarcmin = np.sqrt(2 * eta * NET_25bw_150GHz**2 * sky_arcmin2 / sqnh**2 / duration / epsilon)
### this is for 25% BW

numin_150 = 150.*(1-0.25/2)    #GHz
numax_150 = 150*(1+0.25/2)   #GHz
nulo_150,nuhi_150,nu0_150 = give_bands(numin_150, numax_150, sqnh, exact_numax=True, force_nbands=fnb)
deltanu_150 = nuhi_150-nulo_150
theNET_150 = NET_25bw_150GHz * np.sqrt(numax_150-numin_150) / np.sqrt(deltanu_150)
themuKarcmin_150 = np.sqrt(2 * eta * theNET_150**2 * sky_arcmin2 / sqnh**2 / duration / epsilon)
fwhm_150 = give_fwhm_arcmin(nu0_150, sqnh, deltax)

numin_220 = 220.*(1-0.25/2)    #GHz
numax_220 = 220*(1+0.25/2)   #GHz
nulo_220,nuhi_220,nu0_220 = give_bands(numin_220, numax_220, sqnh, exact_numax=True, force_nbands=fnb)
deltanu_220 = nuhi_220-nulo_220
theNET_220 = NET_25bw_220GHz * np.sqrt(numax_220-numin_220) / np.sqrt(deltanu_220)
themuKarcmin_220 = np.sqrt(2 * eta  * theNET_220**2 * sky_arcmin2 / sqnh**2 / duration / epsilon)
fwhm_220 = give_fwhm_arcmin(nu0_220, sqnh, deltax)

clf()
subplot(2,1,1)
xlim(120, 260)
ylim(0,40)
xlabel('Frequency [GHz]')
ylabel('Noise on maps [$\mu$K.arcmin]')
plot([numin_150, numin_150], [0,40], 'k:', lw=3)
plot([numax_150, numax_150], [0,40], 'k:', lw=3)
plot([numin_220, numin_220], [0,40], 'k:', lw=3)
plot([numax_220, numax_220], [0,40], 'k:', lw=3)
for i in xrange(len(nulo_150)):
    fill_between([nulo_150[i], nuhi_150[i]],[themuKarcmin_150[i], themuKarcmin_150[i]], y2=[0,0], alpha=0.5, color='red')
for i in xrange(len(nulo_220)):
    fill_between([nulo_220[i], nuhi_220[i]],[themuKarcmin_220[i], themuKarcmin_220[i]], y2=[0,0], alpha=0.5, color='blue')
    
subplot(2,1,2)
xlim(120, 260)
ylim(0,30)
xlabel('Frequency [GHz]')
ylabel('FWHM [arcmin]')
plot([numin_150, numin_150], [0,50], 'k:', lw=3)
plot([numax_150, numax_150], [0,50], 'k:', lw=3)
plot([numin_220, numin_220], [0,50], 'k:', lw=3)
plot([numax_220, numax_220], [0,50], 'k:', lw=3)
for i in xrange(len(nulo_150)):
    fill_between([nulo_150[i], nuhi_150[i]],[fwhm_150[i], fwhm_150[i]], y2=[0,0], alpha=0.5, color='red')
for i in xrange(len(nulo_220)):
    fill_between([nulo_220[i], nuhi_220[i]],[fwhm_220[i], fwhm_220[i]], y2=[0,0], alpha=0.5, color='blue')
    
savefig('qubic_1sub.pdf')

print((nulo_150+nuhi_150)/2)
print(themuKarcmin_150)
print(fwhm_150)
print((nulo_220+nuhi_220)/2)
print(themuKarcmin_220)
print(fwhm_220)


#### Synthesized beam cut plot
fwhmprim_150 = 14. #deg
nu = 150e9
lam = 3e8/nu    #m
dx = 14./1000  #m
sqnh = 20
Df = 1. #m
minth = -20.
maxth = 20
nth = 1000
th = np.linspace(minth, maxth, nth)

sb = np.sin(sqnh * np.pi * dx / lam * np.radians(th))**2 / np.sin(np.pi * dx / lam * np.radians(th))**2
sb = sb/max(sb)

fwhmpeak = np.degrees(lam / sqnh / dx)
thetapeak = np.degrees(lam / dx)

clf()
plot(th, give_sbcut(th, dx, lam, sqnh, Df=Df, detpos=0.), lw=2, label='r = 0')
plot(th, give_sbcut(th, dx, lam, sqnh, Df=Df, detpos=50./1000), lw=2, label = 'r = 50 mm')
plot([-fwhmpeak/2, fwhmpeak/2], [0.5,0.5],'m--',lw=2)
plot([-fwhmpeak/2], [0.5],'m',lw=2, marker=5,ms=10)
plot([fwhmpeak/2], [0.5],'m',lw=2, marker=4, ms=10)
text(-9, 0.48, r'$\mathrm{FWHM}=\frac{\lambda}{P\Delta x}$',fontsize=15, color='m')
hh = 0.39
plot([0, thetapeak], [hh,hh],'m--',lw=2)
plot([0], [hh],'m',lw=2, marker=4,ms=10)
plot([thetapeak], [hh],'m',lw=2, marker=5, ms=10)
text(thetapeak/2, 0.32, r'$\theta=\frac{\lambda}{\Delta x}$',fontsize=15, color='m')
plot(th, get_primbeam(th, lam), 'r--', lw=2)
xlabel(r'$\theta$ [deg.]')
ylabel('Synthesized beam')
legend()
savefig('sbfig.pdf')

sbth = give_sbcut(th, dx, lam, sqnh, Df=Df, detpos=0.)

f=open('synthbeam.txt','wb')
for i in xrange(len(th)):
    f.write('{0:10.5f} {1:10.8f}\n'.format(th[i],sbth[i]))
f.close()



dist = [50.]
for d in dist:
    nu0=130.
    nu1=170.
    lam0 = 3e8/(nu0*1e9)
    lam1 = 3e8/(nu1*1e9)
    clf()
    b1 = give_sbcut(th, dx, lam0, sqnh, Df=Df, detpos=d/1000)
    b2 = give_sbcut(th, dx, lam1, sqnh, Df=Df, detpos=d/1000)
    plot(th, b1, 'b', lw=2, label='{0:3.0f} GHz'.format(nu0))
    plot(th, b2, 'g', lw=2, label='{0:3.0f} GHz'.format(nu1))
    plot(th, get_primbeam(th, lam0), 'b--', lw=2)
    plot(th, get_primbeam(th, lam1), 'g--', lw=2)
    xlabel(r'$\theta$ [deg.]')
    ylabel('Synthesized beam')
    legend()
    title(d)
    draw()
    bb1 = b1/np.sqrt(np.sum(b1**2))
    bb2 = b2/np.sqrt(np.sum(b2**2))
    print(np.sum(bb1*bb2))
    
savefig('sb_freq.pdf')


nfreq = 1000
detpos = 0.
dnu_nu = 0.25
nu0 = 150.
numin = nu0*(1.-dnu_nu/2)
numax = nu0*(1.+dnu_nu/2)
nus = np.linspace(numin, numax, nfreq)
lams = 3e8/(nus*1e9)
beams = np.zeros((nfreq, nth))
for i in xrange(nfreq):
    print(i)
    beams[i,:] = give_sbcut(th, dx, lams[i], sqnh, Df=Df, detpos=detpos/1000)

clf()
imshow(10*np.log10(beams), aspect='auto', extent = (minth, maxth, numin, numax), vmin=-35, vmax=0, origin='lower')
colorbar()
xlabel(r'$\theta$ [Deg.]')
ylabel('Frequency [GHz]')



#### Source 1D
amp0 = 1.
amp1 = 1.5
signal0 = amp0 * give_sbcut(th, dx, lam0, sqnh, Df=Df, detpos=0.)
signal1 = amp1 * give_sbcut(th, dx, lam1, sqnh, Df=Df, detpos=0.)

clf()
#plot(th, signal0, 'b--', lw=2, label='{0:3.0f} GHz'.format(nu0))
#plot(th, signal1, 'r--', lw=2, label='{0:3.0f} GHz'.format(nu1))
plot(th, signal0 + signal1, 'k', lw=2, 
    label='amplitudes: \n{0:3.1f} @ {1:3.0f} GHz \n{2:3.1f} @ {3:3.0f} GHz'.format(amp0,nu0,amp1,nu1))
xlabel(r'$\theta$ [deg.]')
ylabel('TOD Signal from point source')
legend()
savefig('signal_sim.pdf')















