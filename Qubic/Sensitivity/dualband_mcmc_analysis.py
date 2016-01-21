from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import pi
import pycamb
from qubic import QubicInstrument
from scipy.constants import c
from Sensitivity import qubic_sensitivity
from Homogeneity import SplineFitting
from pysimulators import FitsArray
from Cosmo import interpol_camb as ic
from Sensitivity import dualband_lib as db
from McMc import mcmc
import pymc
from Sensitivity import data4mcmc





######## Analysis ###########
# rep = 'ChainsSites/Concordia/'
# site = 'Concordia'
# chain_A_r_dl_b = data4mcmc.readchains(rep+'instrumentA_r_dl_b.db')
# chain_B_r_dl_b = data4mcmc.readchains(rep+'instrumentB_r_dl_b.db')
# chain_C_r_dl_b = data4mcmc.readchains(rep+'instrumentC_r_dl_b.db')
# chain_D_r_dl_b = data4mcmc.readchains(rep+'instrumentD_r_dl_b.db')
# chain_E_r_dl_b = data4mcmc.readchains(rep+'instrumentE_r_dl_b.db')
# chain_F_r_dl_b = data4mcmc.readchains(rep+'instrumentF_r_dl_b.db')


# rep = 'ChainsSites/Atacama/'
# site = 'Atacama'
# chain_A_r_dl_b = data4mcmc.readchains(rep+'instrumentAa_r_dl_b.db')
# chain_B_r_dl_b = data4mcmc.readchains(rep+'instrumentBa_r_dl_b.db')
# chain_C_r_dl_b = data4mcmc.readchains(rep+'instrumentCa_r_dl_b.db')
# chain_D_r_dl_b = data4mcmc.readchains(rep+'instrumentDa_r_dl_b.db')
# chain_E_r_dl_b = data4mcmc.readchains(rep+'instrumentEa_r_dl_b.db')
# chain_F_r_dl_b = data4mcmc.readchains(rep+'instrumentFa_r_dl_b.db')


####### Chains for ANR 2015 simulations
rep = '/Users/hamilton/CMB/Interfero/DualBand/SimsANR2015/'
chain_A_r_dl_b = data4mcmc.readchains(rep+'instrumentA_r_dl_b.db')
chain_B_r_dl_b = data4mcmc.readchains(rep+'instrumentB_r_dl_b.db')
chain_C_r_dl_b = data4mcmc.readchains(rep+'instrumentC_r_dl_b.db')
chain_D_r_dl_b = data4mcmc.readchains(rep+'instrumentD_r_dl_b.db')
chain_nofg_r = data4mcmc.readchains(rep+'instrumentNofg_r.db')




def upperlimit(chain,key,level=0.95):
	sorteddata = np.sort(chain[key])
	return sorteddata[level*len(sorteddata)]


truer = 0.
truebeta = 1.59
truedl = 13.4
truealpha = -2.42
trueT = 19.6
level =0.95
cl = int(level*100)


sm=3
histn=3
alpha =0.5




########### r dl and beta

### QUBIC alone
clf()
e=mcmc.matrixplot(chain_E_r_dl_b,['betadust','dldust_80_353','r'], 'pink', sm, limits=[[1.4,2],[12,14.5],[0,0.2]], alpha=alpha,histn=histn, truevals = [truebeta, truedl, truer])
a=mcmc.matrixplot(chain_A_r_dl_b,['betadust','dldust_80_353','r'], 'black', sm, limits=[[1.4,2],[12,14.5],[0,0.2]], alpha=alpha,histn=histn, truevals = [truebeta, truedl, truer])
b=mcmc.matrixplot(chain_B_r_dl_b,['betadust','dldust_80_353','r'], 'brown', sm, limits=[[1.4,2],[12,14.5],[0,0.2]], alpha=alpha,histn=histn, truevals = [truebeta, truedl, truer])
legA = '150x2 : r < {0:5.3f} (95% CL)'.format(upperlimit(chain_A_r_dl_b,'r'))
legB = '150+220 : r < {0:5.3f} (95% CL)'.format(upperlimit(chain_B_r_dl_b,'r'))
legE = '220x2: r < {0:5.3f} (95% CL)'.format(upperlimit(chain_E_r_dl_b,'r'))
legend([a,b,e],[legA, legB, legE], frameon=False, title='QUBIC 2 years ' +site)
savefig('limits_r_beta_dl_qubicalone_'+site+'.png', transparent=True)

### QUBIC + Planck
clf()
f=mcmc.matrixplot(chain_F_r_dl_b,['betadust','dldust_80_353','r'], 'green', sm, limits=[[1.4,2],[12,14.5],[0,0.2]], alpha=alpha,histn=histn, truevals = [truebeta, truedl, truer])
c=mcmc.matrixplot(chain_C_r_dl_b,['betadust','dldust_80_353','r'], 'blue', sm, limits=[[1.4,2],[12,14.5],[0,0.2]], alpha=alpha,histn=histn, truevals = [truebeta, truedl, truer])
d=mcmc.matrixplot(chain_D_r_dl_b,['betadust','dldust_80_353','r'], 'red', sm, limits=[[1.4,2],[12,14.5],[0,0.2]], alpha=alpha,histn=histn, truevals = [truebeta, truedl, truer])
legC = '150x2+353: r < {0:5.3f} (95% CL)'.format(upperlimit(chain_C_r_dl_b,'r'))
legD = '150+220+353: r < {0:5.3f} (95% CL)'.format(upperlimit(chain_D_r_dl_b,'r'))
legF = '220x2+353: r < {0:5.3f} (95% CL)'.format(upperlimit(chain_F_r_dl_b,'r'))
legend([c,d,f],[legC, legD, legF], frameon=False, title='QUBIC 2 years '+site)
savefig('limits_r_beta_dl_qubicwithPlanck_'+site+'.png', transparent=True)

### Au final
clf()
a=mcmc.matrixplot(chain_A_r_dl_b,['betadust','dldust_80_353','r'], 'black', sm, limits=[[1.4,2],[12,14.5],[0,0.2]], alpha=alpha,histn=histn, truevals = [truebeta, truedl, truer])
b=mcmc.matrixplot(chain_B_r_dl_b,['betadust','dldust_80_353','r'], 'brown', sm, limits=[[1.4,2],[12,14.5],[0,0.2]], alpha=alpha,histn=histn, truevals = [truebeta, truedl, truer])
c=mcmc.matrixplot(chain_C_r_dl_b,['betadust','dldust_80_353','r'], 'blue', sm, limits=[[1.4,2],[12,14.5],[0,0.2]], alpha=alpha,histn=histn, truevals = [truebeta, truedl, truer])
d=mcmc.matrixplot(chain_D_r_dl_b,['betadust','dldust_80_353','r'], 'red', sm, limits=[[1.4,2],[12,14.5],[0,0.2]], alpha=alpha,histn=histn, truevals = [truebeta, truedl, truer])
legA = '150x2 : r < {0:5.3f} (95% CL)'.format(upperlimit(chain_A_r_dl_b,'r'))
legB = '150+220 : r < {0:5.3f} (95% CL)'.format(upperlimit(chain_B_r_dl_b,'r'))
legC = '150x2+353: r < {0:5.3f} (95% CL)'.format(upperlimit(chain_C_r_dl_b,'r'))
legD = '150+220+353: r < {0:5.3f} (95% CL)'.format(upperlimit(chain_D_r_dl_b,'r'))
legend([a,b,c,d],[legA, legB, legC, legD], frameon=False, title='QUBIC 2 years '+site)
savefig('limits_r_beta_dl_qubicwithPlanckFinal_'+site+'.png', transparent=True)









