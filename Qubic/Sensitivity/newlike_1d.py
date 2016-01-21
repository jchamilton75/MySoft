### to call after dualband_mcmc.py


efficiency = 1

noforegrounds = np.array([0, alphadust, betadust, Tdust])
dataNofg = prepare_inst(thervalue, inst, ellbins,
	[150, 220],
	['bi', 'bi'],
	[net150_concordia/sqrt(efficiency), net220_concordia/sqrt(efficiency)],
	['150, 220 No Foregrounds'],
	'r',
	0.01,
	[qubic_duration, qubic_duration, planck_duration],
	[qubic_epsilon, qubic_epsilon, planck_epsilon],
	camblib=camblib, dustParams=noforegrounds)


clf()
##### new with no foregrounds
spec = dataNofg['specin']
inst_info = dataNofg['inst_info']
paramsdefault = np.append([0], noforegrounds)

nvals = 100
valsamp = linspace(0, 0.1, nvals)
index = 0

thelike = db.like_1d(spec, index, valsamp, inst_info, camblib=camblib, paramsdefault=paramsdefault, CL=0.95)

