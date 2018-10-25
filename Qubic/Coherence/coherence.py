

From https://www.researchgate.net/publication/1923639_Coherence_properties_of_infrared_thermal_emission_from_heated_metallic_nanowires

h = 6.62e-34
k = 1.38e-23
T = 300.
nu0=150.e9

nn=100000
nus = np.double(np.linspace(0,5000*nu0,nn))

g = np.nan_to_num(nus**3/(np.exp(h*nus/(k*T))-1))

clf()
plot(nus/1e9,g)
xscale('log')
yscale('log')

ftg = np.abs(np.fft.fft(g))
ftg = ftg/np.max(ftg)
ftnus = np.fft.fftfreq(nn,d=nus[1]-nus[0])

xx = ftnus*3e8*1e6/2

clf()
plot(xx,ftg)
xlim(0,25)

print(np.max(xx[ftg>0.1]))
=> coherence length = 5.4 microns (at 10%)




