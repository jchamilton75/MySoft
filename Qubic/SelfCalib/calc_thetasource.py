



nn = 100
dist_src = 40 * sqrt(2) #m
fwhm_src = np.radians(np.linspace(5., 20, nn)) #rad
omega_src = 2 * pi * (fwhm_src / 2.35)**2

d_horn =  0.0127 #m
surf_horn = pi * (d_horn**2) / 4   #m2
omega_horn = surf_horn / dist_src**2 #st


dilution_factor = omega_horn / omega_src

clf()
plot(np.degrees(fwhm_src), dilution_factor * 5e-3)
plot(np.degrees(fwhm_src), np.zeros(nn)+1e-9, 'k--')
plot(np.degrees(fwhm_src), np.zeros(nn)+1e-8, 'k--')
yscale('log')
ylim(1e-10,1e-7)
ylabel('Puissance entrant dans 1 cornet [W]')
xlabel('FWHM de la source [deg]')
title('Puissance Source : 5 mW')
