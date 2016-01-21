from __future__ import division
from numpy import *
from matplotlib.pyplot import *
import scipy
h = 6.626070040e-34
c = 299792458
k = 1.3806488e-23

## New Instrument
window     = {'Name':'Window',            'T':250.,    'e':0.01,  'trans':0.98}
IRblocker1 = {'Name':'IRblocker1',        'T':250.,    'e':0.01,  'trans':0.98}
cm12_1edge = {'Name':'12cm1edge',         'T':100.,    'e':0.02,  'trans':0.95}
IRblocker2 = {'Name':'IRblocker2',        'T':100.,    'e':0.02,  'trans':0.98}
cm10_1edge = {'Name':'10cm1edge',         'T':6.,      'e':0.02,  'trans':0.95}
hwp        = {'Name':'HWP',               'T':6.,      'e':0.025, 'trans':0.95}
polariser  = {'Name':'Polariser',         'T':6.,      'e':0.025, 'trans':0.95/2}
horns_sw   = {'Name':'B2BHorns+Switches', 'T':6.,      'e':0.05,  'trans':0.99}
combiner   = {'Name':'Optical Combiner',  'T':1.,      'e':0.01,  'trans':0.99}
dichroic   = {'Name':'Dichroic',          'T':1.,      'e':0.02,  'trans':0.95}
#BAND1: 150 GHz
cm6_1edge_150   = {'Name':'6cm1edge-150',                'T':0.3,    'e':0.02,  'trans':0.98}
band_filter_150 = {'Name':'150 GHz band filter',         'T':0.3,    'e':0.02,  'trans':0.98}
#BAND2: 220 GHz
cm9_1edge_220   = {'Name':'9cm1edge-220',                'T':0.3,    'e':0.02,  'trans':0.98}
band_filter_220 = {'Name':'220 GHz band filter',         'T':0.3,    'e':0.02,  'trans':0.80}

optics_150 = [window, IRblocker1, cm12_1edge, IRblocker2, cm10_1edge, hwp, polariser, horns_sw, 
                combiner, dichroic, cm6_1edge_150, band_filter_150]
optics_220 = [window, IRblocker1, cm12_1edge, IRblocker2, cm10_1edge, hwp, polariser, horns_sw, 
                combiner, dichroic, cm9_1edge_220, band_filter_220]
freqsGHz = [150., 220.]
optics = [optics_150, optics_220]
#freqsGHz = [150.]
#optics = [optics_150]
inst = {'deltanu_nu':0.25, 'nhorns':400, 
		'efficiency':1., 'fraction':0.76, 'nbolos':992, 'nep_bolo':4e-17 }



## Old Instrument
#freqsGHz = [150.]
#window     = {'Name':'Window',            'T':250.,    'e':0.02,  'trans':1}
#filt250K   = {'Name':'filters 250K',      'T':250.,    'e':0.01,  'trans':1}
#filt50K    = {'Name':'filters 50K',       'T':50.,     'e':0.01,  'trans':1}
#HWP        = {'Name':'HWP',               'T':30.,     'e':0.02,  'trans':1}
#Horns      = {'Name':'Horns',             'T':30.,     'e':0.1,  'trans':1}
#M1         = {'Name':'M1',                'T':4.,      'e':0.1, 'trans':1}
#M2         = {'Name':'M2',                'T':4.,      'e':0.1,  'trans':1}
#filt100mK  = {'Name':'filters 100mK',     'T':4.,      'e':0.01,  'trans':1}
#optics = [[window, filt250K, filt50K, HWP, Horns, M1, M2, filt100mK]]
#inst = {'deltanu_nu':0.25, 'nhorns':400, 
#		'efficiency':0.3, 'fraction':0.67, 'nbolos':1024, 'nep_bolo':4e-17 }







cmb     = {'Name':'CMB         ', 'T':2.73,    'e':1, 'trans':1}
## Dome C
atmdef  = {'Name':'Atm Dome C  ', 'T':200,    'e':0.033, 'trans':1}

def Bnu(nuGHz, T):
    nu = nuGHz * 1e9
    val = 2 * h * nu**3 / c**2 / (np.exp(h * nu / k / T) -1)
    return val
    
def dBnudT(nuGHz, T):
    bnu = Bnu(nuGHz, T)
    nu = nuGHz * 1e9
    val = c**2 / 2 / nu**2 * bnu * bnu * np.exp(h * nu / k / T) / k / T**2
    return val
    
def PowerOneHorn(nuGHz, T, emissivity,deltanu_nu):
    nu = nuGHz * 1e9
    deltanu = nu * deltanu_nu
    lambd = c / nu
    SOmega = lambd**2
    val = Bnu(nuGHz, T) * deltanu * SOmega * emissivity
    return(val)
    
def PowerOneHornComp(nuGHz, deltanu_nu, component):
    return PowerOneHorn(nuGHz, component['T'], component['e'], deltanu_nu)
     
def PowerInstrumentChannel(nuGHz, optics, atm=atmdef, verbose=True):
    deltanu_nu = inst['deltanu_nu']
    nhorns = inst['nhorns']
    powerins = 0.
    powercmb = PowerOneHornComp(nuGHz, inst['deltanu_nu'], cmb)
    poweratm = PowerOneHornComp(nuGHz, inst['deltanu_nu'], atm)
    powerall = powerins + poweratm + powercmb
    thepinst =0.
    opt_efficiency = 1.
    name = 'Outside window'
    if verbose: print('-------------------------------------------------------------------------------------------------------------------')
    if verbose: print('    Power for one horn')
    if verbose: print('-------------------------------------------------------------------------------------------------------------------')
    if verbose: print('Component           |   T   |   e   |   t   || Pcmb (pW)  || Patm (pW)  || Pinst (pW) | Totinst  (pW) | Pall (pW) |')
    if verbose: print('{0:20s}| {6:5.1f} | {7:5.3f} | {8:5.3f} || {1:8.2g}   || {2:8.2g}   || {3:8.2g}   | {4:8.2g}      | {5:8.2g}  |'.format(name, powercmb*1e12, 
                        poweratm*1e12, thepinst*1e12, powerins*1e12, powerall*1e12,np.nan,np.nan,np.nan))
    for comp in optics:
        ### apply transmission to all
        powercmb *= comp['trans']
        poweratm *= comp['trans']
        powerins *= comp['trans']
        opt_efficiency *= comp['trans']
        ### add component power to instrument power
        name = comp['Name']
        thepinst = PowerOneHornComp(nuGHz, deltanu_nu, comp)
        powerins += thepinst
        powerall = powerins + poweratm + powercmb
        if verbose: print('{0:20s}| {6:5.1f} | {7:5.3f} | {8:5.3f} || {1:8.2f}   || {2:8.2f}   || {3:8.2g}   | {4:8.2f}      | {5:8.2f}  |'.format(name, powercmb*1e12, 
                        poweratm*1e12, thepinst*1e12, powerins*1e12, powerall*1e12, comp['T'],comp['e'], comp['trans']))
    name = 'All'
    if verbose: print('{0:20s}| {6:5.1f} | {7:5.3f} | {8:5.3f} || {1:8.2f}   || {2:8.2f}   || {3:8.2f}   | {4:8.2f}      | {5:8.2f}  |'.format(name, powercmb*1e12, 
                        poweratm*1e12, np.nan, powerins*1e12, powerall*1e12, np.nan, np.nan, np.nan))
    if verbose: print('-------------------------------------------------------------------------------------------------------------------')
    if verbose: print('Component           |   T   |   e   |   t   || Pcmb (pW)  || Patm (pW)  || Pinst (pW) | Totinst  (pW) | Pall (pW) |')
    powercmb *= inst['efficiency']
    poweratm *= inst['efficiency']
    powerins *= inst['efficiency']
    powerall *= inst['efficiency']
    name = 'All w. Bol eff. {0:3.2}'.format(inst['efficiency'])
    if verbose: print('{0:20s}| {6:5.1f} | {7:5.3f} | {8:5.3f} || {1:8.2f}   || {2:8.2f}   || {3:8.2f}   | {4:8.2f}      | {5:8.2f}  |'.format(name, powercmb*1e12, 
                        poweratm*1e12, np.nan, powerins*1e12, powerall*1e12, np.nan, np.nan, np.nan))    
    if verbose: print('-------------------------------------------------------------------------------------------------------------------')
    if verbose: print('    Power for all {0} horns'.format(nhorns))
    powercmb *= nhorns
    poweratm *= nhorns
    powerins *= nhorns
    powerall = powerins + poweratm + powercmb
    if verbose: print('-------------------------------------------------------------------------------------------------------------------')
    if verbose: print('Component           |   T   |   e   |   t   || Pcmb (nW)  || Patm (nW)  || Pinst (nW) | Totinst  (nW) | Pall (nW) |')
    if verbose: print('{0:20s}| {6:5.1f} | {7:5.3f} | {8:5.3f} || {1:8.2f}   || {2:8.2f}   || {3:8.2f}   | {4:8.2f}      | {5:8.2f}  |'.format(name, powercmb*1e9, 
                        poweratm*1e9, np.nan, powerins*1e9, powerall*1e9, np.nan, np.nan, np.nan))
    if verbose: print('-------------------------------------------------------------------------------------------------------------------')
    if verbose: print('Instrument optical efficiency                         : {0:8.2f}'.format(opt_efficiency))
    if verbose: print('Instrument Total efficiency                           : {0:8.2f}'.format(inst['efficiency']*opt_efficiency))
    if verbose: print('Total power arriving on array                         : {0:8.2f} nW'.format(powerall*1e9))
    return powerall, opt_efficiency
        
    
def give_NET(atm=atmdef, verbose=True, freqGHz=150., bol_efficiency=0.8):
    thefreqGHz = freqGHz
    if freqGHz == 150.:
        theoptics = optics_150
    if freqGHz == 220.:
        theoptics = optics_220
    
    inst["efficiency"]=bol_efficiency
    
    if verbose: print('\n\n')
    if verbose: print('============================== {0:5} GHz =========================================================================='.format(thefreqGHz))
    if verbose: print('Atmosphere Model: '+atm['Name'])
    if verbose: print('Atmosphere Temperature: {0:5.1f} K'.format(atm['T']))
    if verbose: print('Atmosphere Emissivity: {0:5.3f} '.format(atm['e']))
    ######## Power on array
    power, optical_efficiency = PowerInstrumentChannel(thefreqGHz, theoptics, atm=atm, verbose=verbose)
    ######## Including External efficiency and Fraction integrated on bolometers
    power_integrated = power * inst['fraction']
    nbsigmas = np.sqrt(-np.log(1-inst['fraction'])*2)   #just analytical calculation
    xx = np.linspace(-nbsigmas, nbsigmas, 1000)
    vv = 1./(2*np.pi)*np.exp(-0.5*xx**2)   ## bivariate gaussian
    miniratio = np.min(vv)/np.mean(vv)
    maxiratio = np.max(vv)/np.mean(vv)
    if verbose: print('Total power Integrated by bolometers (fraction = {1:4.2f}): {0:8.2f} nW'.format(power_integrated*1e9, inst['fraction']))
    power_bol = power_integrated / inst['nbolos']
    mini_power_bol = miniratio * power_bol
    maxi_power_bol = maxiratio * power_bol
    if verbose: print('Total power per bolometer (n={1:4g})                    : {0:8.2f} pW   - Mini = {2:8.2f} pW - Maxi = {3:8.2f} pW'.format(power_bol*1e12, 
        inst['nbolos'], mini_power_bol*1e12, maxi_power_bol*1e12))
    ######## Noise calculation
    if verbose: print('-------------------------------------------------------------------------------------------------------------------')
    shot_noise, bunching_noise, nep_photons = photon_noise(thefreqGHz, power_bol)
    mini_shot_noise, mini_bunching_noise, mini_nep_photons = photon_noise(thefreqGHz, mini_power_bol)
    maxi_shot_noise, maxi_bunching_noise, maxi_nep_photons = photon_noise(thefreqGHz, maxi_power_bol)
    if verbose: print('Shot Noise        : {0:8.2g} W/sqrt(Hz)     Mini = {1:8.2g} W/sqrt(Hz)    Maxi = {2:8.2g} W/sqrt(Hz)'.format(shot_noise, mini_shot_noise, maxi_shot_noise))
    if verbose: print('Bunching Noise    : {0:8.2g} W/sqrt(Hz)     Mini = {1:8.2g} W/sqrt(Hz)    Maxi = {2:8.2g} W/sqrt(Hz)'.format(bunching_noise, mini_bunching_noise, maxi_bunching_noise))
    if verbose: print('Total Photon Noise: {0:8.2g} W/sqrt(Hz)     Mini = {1:8.2g} W/sqrt(Hz)    Maxi = {2:8.2g} W/sqrt(Hz)'.format(nep_photons, mini_nep_photons,maxi_nep_photons))
    nep_bolo = inst['nep_bolo']
    if verbose: print('Bolometer Noise   : {0:8.2g} W/sqrt(Hz)'.format(nep_bolo))
    nep = np.sqrt(nep_photons**2 + nep_bolo**2)
    if verbose: print('-------------------------------------------------------------------------------------------------------------------')
    if verbose: print('NEP (Intensity): {0:8.2g} W/sqrt(Hz)'.format(nep))
    if verbose: print('-------------------------------------------------------------------------------------------------------------------')
    net = nep2net(nep, thefreqGHz, inst['efficiency']*optical_efficiency, inst['deltanu_nu']*thefreqGHz, cmb['T'])
    if verbose: print('NET (Polarized): {0:8.0f} muK.sqrt(s)'.format(net))
    if verbose: print('==================================================================================================================='.format(thefreqGHz))
    return power_bol, optical_efficiency, nep, net

        
def photon_noise(nuGHz, power_bolometer):
    ### this is in W/sqrt(Hz)
    nu = nuGHz * 1e9
    shot_noise = np.sqrt(2 * h * nu * power_bolometer)
    deltanu = nu * inst['deltanu_nu']
    bunching_noise = np.sqrt(2 * power_bolometer**2 / deltanu)
    nep_photons = np.sqrt(shot_noise**2 + bunching_noise**2)
    return shot_noise, bunching_noise, nep_photons

def nep2net(nep, nuGHz, eff, deltanuGHz, T):
    deltanu = deltanuGHz * 1e9
    nu = nuGHz * 1e9
    lambd = c / nu
    net = nep / (lambd**2 * eff * deltanu) / dBnudT(nuGHz, T) * 1e6
    return net/sqrt(2)*sqrt(2)   ### so that we are in muK.sqrt(s) for polarized if NEP is for intensity in muK/sqrt(Hx)
    