import scipy.constants as cst




##### On explore les possibilités de Spectro-Imagerie de l'interférométrie-Bolométrique
def give_bands(numin, appnumax, sqnh, exact_numax=True, force_nbands=False):
    deltanu_nu = 1./sqnh
    lnumin = np.log10(numin)
    if  not exact_numax:
        applnumax = np.log10(appnumax)
        appnbands = (applnumax - lnumin) / np.log10((1+deltanu_nu))
        nbands = int(np.ceil(appnbands))
        lnumax = lnumin + nbands * np.log10((1+deltanu_nu))
    else:
        numax = appnumax
        lnumax = np.log10(numax)
        nbands = int(floor((lnumax - lnumin) / np.log10((1+deltanu_nu))))

    if force_nbands: nbands = force_nbands
    nuall = np.logspace(lnumin, lnumax, nbands+1)
    nulo = nuall[0:-1]
    nuhi = nuall[1:]
    return nulo, nuhi, (nuhi+nulo)/2

def give_fwhm_arcmin(nu, sqnh, deltax):
    return np.degrees(cst.c/(nu*1e9 * sqnh *deltax))*60

def Bnu(nu, T):
    h = 6.62607e-34
    kb = 1.380658e-23
    c = 299792458#
    x = h * nu / (kb * T)
    Bnu = 2. * h * nu**3 / c**2 * 1. / (np.exp(x) - 1)
    return Bnu


def dBnu_dT(nu, T):
    h = 6.62607e-34
    kb = 1.380658e-23
    c = 299792458#
    x = h * nu / (kb * T)   
    b_nu = Bnu(nu, T)
    dBnu_dT = c**2 / 2 / nu**2 * b_nu * b_nu * np.exp(x) / kb / T**2
    return dBnu_dT

def nep2net(NEP, nuGHz, deltanuGHz, epsilon=1, Tcmb = 2.72, nmodes=1):
    thelambda = cst.c/(nuGHz*1e9)
    dbnu_dt = dBnu_dT(nuGHz*1e9, Tcmb)
    return NEP / sqrt(2) / thelambda**2 / nmodes /epsilon / (deltanuGHz*1e9) / dbnu_dt



######## Space mission case
######### Need a model for power from sky + instrument
# from Michel Excel File in same directory
import scipy.integrate
freqs_cpp = np.array([60., 70., 80., 90., 100., 115., 130., 145., 160, 175, 195, 220., 255., 295., 340., 390., 450., 520., 600.])
## this is the power per hertz and per horn
#power_per_Hz_cpp = 1e-9 * np.array([20.8E-15,	19.8E-15,	18.8E-15, 17.9E-15, 17.0E-15, 15.9E-15,	14.9E-15, 14.1E-15,	13.4E-15, 12.7E-15, 12.1E-15, 11.4E-15, 10.7E-15, 10.2E-15,	9.8E-15, 9.6E-15, 9.4E-15, 9.2E-15, 9.0E-15])
#mukarcmincorepp=np.array([2.40E+01, 2.25E+01, 1.79E+01, 1.43E+01, 1.33E+01, 1.12E+01, 9.72E+00, 8.50E+00, 9.30E+00, 9.46E+00, 9.85E+00, 1.56E+01, 2.53E+01, 5.73E+01, 9.37E+01, 1.69E+02, 3.61E+02, 9.16E+02, 2.78E+03])
#cpp_fwhm = np.array([14.0, 12.0, 10.5, 9.3, 8.4, 7.3, 6.5, 5.8, 5.2, 4.8, 4.3, 3.8, 3.3, 2.8, 2.5, 2.2, 1.9, 1.6, 1.4])

### new
freqs_cpp = np.array([60., 70., 80., 90., 100., 115., 130., 145., 160, 175, 195, 220., 255., 295., 340., 390., 450., 520., 600.])
cpp_fwhm = np.array([14.0, 12.0, 10.5, 9.3, 8.4, 7.3, 6.5, 5.8, 5.2, 4.8, 4.3, 3.8, 3.3, 2.8, 2.5, 2.2, 1.9, 1.6, 1.4])
mukarcmincorepp=np.array([2.40E+01, 2.25E+01, 1.79E+01, 1.43E+01, 1.33E+01, 1.12E+01, 9.72E+00, 8.50E+00, 9.30E+00, 9.46E+00, 9.85E+00, 1.56E+01, 2.53E+01, 5.73E+01, 9.37E+01, 1.69E+02, 3.61E+02, 9.16E+02, 2.78E+03])
dnu_nu_cpp = 0.25
dnu_cpp = freqs_cpp * dnu_nu_cpp

#power_per_skybeam_cpp = [312.6E-15,	345.9E-15,	375.4E-15,	401.8E-15,	425.5E-15,	457.1E-15,	485.0E-15,	510.5E-15,	534.4E-15,	557.5E-15,	587.8E-15,	626.4E-15,	683.2E-15,	752.8E-15,	836.3E-15,	933.3E-15,	1.1E-12,	1.2E-12,	1.3E-12]



def power_sky_inetendue_perHz(nuGHz, pol=True, detector_efficiency = 0.5, telescope_em=0.01, telescope_T=100., enclosure_eff=0.005, enclosure_T=100.,
                Tcmb=2.726, dustfrac=1.):
    if pol==True:
        polfrac = 0.5
    else:
        polfrac =1.
    
    telescope_power = telescope_em * Bnu(nuGHz*1e9, telescope_T)
    enclosure_power = enclosure_eff * Bnu(nuGHz*1e9, enclosure_T)
    cmb_power = Bnu(nuGHz*1e9, Tcmb)
    dust_power = dustfrac * 8e-21 * (nuGHz*1./1874)**0.959 * Bnu(nuGHz*1e9, 17.) / Bnu(1874*1e9, 17)
    all_power = polfrac * detector_efficiency * (telescope_power + enclosure_power + cmb_power + dust_power )
    
    wavelength = cst.c / nuGHz / 1e9
    all_power_etendue = all_power * wavelength**2 
    return all_power_etendue





def performances(numin, numax, size, deltax, df, bolosize, nmodemin, fnb=False):
    sqnh = np.floor(size/deltax)
    deltanu_nu = 1./sqnh

    ### Calculate frequencies, fwhm and number of modes assuming quadratic scaling with one mode at the lowest frequency (conservative)
    nulo,nuhi,nu0 = give_bands(numin, numax, sqnh, exact_numax=True, force_nbands=fnb)
    fwhm = give_fwhm_arcmin(nu0, sqnh, deltax)
    wavelength = cst.c/nu0/1e9


    ### Now power from the sky at all frequencies
    nmodes = nmodemin * nu0**2 / np.min(nu0)**2

    ### Lobe des pupilles : constant car le nombre de modes le compense ???
    ### S * Omega = nmodes * lambda**2
    Shorns = np.pi * deltax**2/4
    omega = nmodes * (cst.c / (nu0*1e9))**2 / Shorns
    fwhmhorns = np.degrees(2.35 * np.sqrt(omega/ 2 / np.pi))
    ### attention en fait elles ne sont pas gaussiennes, plutot top-hat

    ### dans le plan focal: deltaphi = 2pi * d/lambda * x/Df = 2pi * u * x/Df
    ### avec d distance dans le plan de cornets et x distance dans le plan focal
    ### frange la plus fine ulim correspond à une taille xlim dans le plan focal (déphasage de 2pi pour une frange) : xlim/Df = 1/ulim
    ### cette frange est samplée avec ns bolometres ns = Df/bolosize / ulim 
    ulim = size / wavelength
    ns = df/ulim/bolosize
    print(ns)
    ### taille de la tache sur le plan focal fwhm_fp = Df * fwhmhorns
    fwhm_fp = df * np.radians(fwhmhorns[0])
    nbols = np.pi*fwhm_fp**2/4 / bolosize**2
    print(fwhm_fp)
    print(nbols, np.sqrt(nbols))

    ### le lobe de pixels sur le ciel est en fait 
    sizepix = np.degrees(bolosize / df)*60
    ### So the total FWHM will besesee proof in Mathematica notebook called "pixel integration"
    fwhmtot = 2.35 * np.sqrt( sizepix**2/12 + (fwhm/2.35)**2)
  
    allfreqs = np.linspace(numin,numax,10000)

    #### Puissance du ciel dans le beam etendue => arrivant sur chaque bolo
    allpower_per_Hz_singlemode = power_sky_inetendue_perHz(allfreqs, enclosure_eff=0.) /nbols
    allpower_per_Hz = allpower_per_Hz_singlemode * allfreqs**2 / np.min(nu0)**2 * nmodemin
    power_total = scipy.integrate.trapz(allpower_per_Hz, allfreqs*1e9)

    ##### NEP Calculation for each detector ################################
    ### From Lamarre 1986 Eq. 7
    nep_shot = np.sqrt(scipy.integrate.trapz(2* cst.h * allfreqs*1e9 *allpower_per_Hz, allfreqs*1e9))
    nep_bunch = np.sqrt(scipy.integrate.trapz(2 *allpower_per_Hz**2, allfreqs*1e9))
    print(nep_shot, nep_bunch, nep_bunch/nep_shot)
    nep = np.sqrt(nep_shot**2 + nep_bunch**2)


    ####### NET and muK.arcmin calculation
    duration = 3*365*24*3600
    fsky = 1.
    sky_arcmin2 = fsky * 4 *np.pi * (180/np.pi)**2 * 60**2
    eta = 1.
    epsilon = 0.5
    allNET= nep2net(nep/np.sqrt(len(nu0))*np.sqrt(nbols), nu0, nuhi-nulo, epsilon=epsilon, nmodes=nmodes)

    themuKarcmin = np.sqrt(2 * eta  * (allNET*1e6)**2 * sky_arcmin2 / sqnh**2 / duration / epsilon)
    return nulo, nuhi, themuKarcmin, fwhmtot, sqnh, deltanu_nu, nbols




from scipy import integrate
numin = 60*(1-0.25/2)    #GHz
numax = 550*(1+0.25/2)   #GHz
size = 1.2   #meter
deltax = 14./1000 * (150.-0.25/2) / numin  #m scaling based on QUBIC 150 GHz band
fnb = False
#fnb = len(freqs_cpp[freqs_cpp <= numax])
df = 2.5 #m
#bolosize = np.max(wavelength)
bolosize = 0.003
nmodemin=1

nulo, nuhi, themuKarcmin, fwhmtot, sqnh, deltanu_nu, nbols = performances(numin, numax, size, 
            deltax, df, bolosize, nmodemin, fnb=fnb)

nulo2, nuhi2, themuKarcmin2, fwhmtot2, sqnh2, deltanu_nu2, nbols2 = performances(numin, numax, size, 
            deltax, df, bolosize, nmodemin, fnb=len(freqs_cpp[freqs_cpp <= numax]))



nunu = np.array([nulo,nuhi]).T.ravel()
ysens = np.array([themuKarcmin, themuKarcmin]).T.ravel()
yres = np.array([fwhmtot, fwhmtot]).T.ravel()

nunu2 = np.array([nulo2,nuhi2]).T.ravel()
ysens2 = np.array([themuKarcmin2, themuKarcmin2]).T.ravel()
yres2 = np.array([fwhmtot2, fwhmtot2]).T.ravel()


clf()
subplot(2,1,1)
ylim(0,100)
xscale('log')
vals = np.floor(logspace(np.log10(60), np.log10(600), 10))
xticks(vals, int32(vals))
xlim(50,700)
yscale('log')
ylim(1,3000)
xlabel('Frequency [GHz]')
ylabel('Noise on maps [$\mu$K.arcmin]')

plot(nunu, ysens, 'r', label='Max. Spec. Resolution: {0:3.0f} bands'.format(len(nulo)), lw=3)
plot(nunu2, ysens2, 'b', label='Spec. Resolution: {0:3.0f} bands'.format(len(nulo2)), lw=3)
errorbar(freqs_cpp, mukarcmincorepp,xerr=freqs_cpp*0.25/2,fmt='k,', lw=3, label='Core++')
title('Horn Array size: {0:3.1f} m - Horns distance: {1:5.1f} mm - {2:3.0f}x{2:3.0f}={3:4.0f} array \n Focal distance={5:3.1f}m - Nbols = {6:6.0f} - Nbol radius = {7:4.0f} - radius = {8:4.2f}m - nmodemin: {9:3.0f}'.format(size, deltax*1000, sqnh,sqnh**2, len(nulo), df, nbols, np.sqrt(nbols/np.pi), np.sqrt(nbols/np.pi)*bolosize, nmodemin))

subplot(2,1,2)
xlim(50,700)
ylim(0,np.max(fwhmtot))
xlabel('Frequency [GHz]')
ylabel('FWHM [arcmin]')
xscale('log')
vals = np.floor(logspace(np.log10(60), np.log10(600), 10))
xticks(vals, int32(vals))
plot(nunu,yres, 'r', label='B.I.: Max. Number of bands: {0:3.0f} bands'.format(len(nulo)), lw=3)
plot(nunu2,yres2, 'b', label='B.I.: {0:3.0f} bands'.format(len(nulo2)), lw=3)
errorbar(freqs_cpp, cpp_fwhm,xerr=freqs_cpp*0.25/2,fmt='k,', lw=3, label='CoRE++ forecasts')
legend(numpoints=1)

savefig('result_sat.pdf')