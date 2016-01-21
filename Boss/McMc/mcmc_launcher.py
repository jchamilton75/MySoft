#!/bin/env python
import sys
import pymc
from pymc import Metropolis
import cosmolopy
from McMc import mcmc
from astropy.io import fits
from McMc import cosmo_utils
import scipy
from McMc import data_lyaDR11
reload(data_lyaDR11)
from McMc import data_lyaDR11_Halone
reload(data_lyaDR11_Halone)
from McMc import data_DR7
reload(data_DR7)
from McMc import data_Beutler
reload(data_Beutler)
from McMc import data_Anderson
reload(data_Anderson)
from McMc import data_base_planck_lowl_lowLike as data_planck
reload(data_planck)
from McMc import data_base_omegak_planck_lowl_lowLike as data_planck_ok
reload(data_planck_ok)
from McMc import data_base_w_planck_lowl_lowLike as data_planck_w
reload(data_planck_w)
from McMc import data_Jim_Planck_obh2_och2_theta as data_jimplanck
reload(data_jimplanck)
from McMc import data_hPlanck_1sig
reload(data_hPlanck_1sig)
from McMc import data_hPlanck_2sig
reload(data_hPlanck_2sig)
from McMc import data_hRiess_1sig
reload(data_hRiess_1sig)
from McMc import data_hRiess_2sig
reload(data_hRiess_2sig)
from McMc import data_hRiessPlanck
reload(data_hRiessPlanck)
from McMc import data_obh2_Planck_1sig
reload(data_obh2_Planck_1sig)
from McMc import data_obh2_Planck_2sig
reload(data_obh2_Planck_2sig)
from McMc import data_obh2_BBN
reload(data_obh2_BBN)


model=sys.argv[1]
dsname=sys.argv[2]

##### background cosmology : when not variables, the parameters are fixed as below
mycosmo=cosmolopy.fidcosmo.copy()
mycosmo['h']=0.6704
mycosmo['Y_He']=0.247710
obh2=0.022032
onh2=0.000645
och2=0.120376-onh2
mycosmo['omega_M_0']=(och2+obh2+onh2)/mycosmo['h']**2
mycosmo['omega_lambda_0']=1.-mycosmo['omega_M_0']
mycosmo['omega_k_0']=0
mycosmo['omega_b_0']=obh2/mycosmo['h']**2
mycosmo['omega_n_0']=onh2/mycosmo['h']**2
mycosmo['n']=0.9619123
mycosmo['Num_Nu_massless']=3.046/3*2
mycosmo['Num_Nu_massive']=3.046/3

repchains='/Users/hamilton/SDSS/LymanAlpha/JCMC_Chains/'
reload(mcmc)
reload(cosmo_utils)
library='jc'
flat=False

if model=='lambdacdm':
    variables=['h','omega_M_0','omega_b_0']
    thedata_planck=data_planck
elif model=='olambdacdm':
    variables=['h','omega_M_0','omega_k_0','omega_b_0']
    thedata_planck=data_planck_ok
elif model=='flatwcdm':
    variables=['h','omega_M_0','w','omega_b_0']
    thedata_planck=data_planck_w
elif model=='owcdm':
    variables=['h','omega_M_0','omega_k_0','w','omega_b_0']
    thedata_planck=data_jimplanck
elif model=='olambdacdm_fixed_h_obh2':
    hriess=0.738
    hplanck=0.674
    hval=(hriess+hplanck)/2
    mycosmo['h']=hval
    obh2=0.02207
    onh2=0.000645
    och2=0.120376-onh2
    mycosmo['omega_M_0']=(och2+obh2+onh2)/mycosmo['h']**2
    mycosmo['omega_lambda_0']=1.-mycosmo['omega_M_0']
    mycosmo['omega_k_0']=0
    mycosmo['omega_b_0']=obh2/mycosmo['h']**2
    mycosmo['omega_n_0']=onh2/mycosmo['h']**2
    variables=['omega_M_0','omega_k_0']
    thedata_planck=data_planck_ok
elif model=='flatwcdm_fixed_h_obh2':
    hriess=0.738
    hplanck=0.674
    hval=(hriess+hplanck)/2
    mycosmo['h']=hval
    obh2=0.02207
    onh2=0.000645
    och2=0.120376-onh2
    mycosmo['omega_M_0']=(och2+obh2+onh2)/mycosmo['h']**2
    mycosmo['omega_lambda_0']=1.-mycosmo['omega_M_0']
    mycosmo['omega_k_0']=0
    mycosmo['omega_b_0']=obh2/mycosmo['h']**2
    mycosmo['omega_n_0']=onh2/mycosmo['h']**2
    variables=['omega_M_0','w']
    thedata_planck=data_planck_w
elif model=='owcdm_fixed_h_obh2':
    hriess=0.738
    hplanck=0.674
    hval=(hriess+hplanck)/2
    mycosmo['h']=hval
    obh2=0.02207
    onh2=0.000645
    och2=0.120376-onh2
    mycosmo['omega_M_0']=(och2+obh2+onh2)/mycosmo['h']**2
    mycosmo['omega_lambda_0']=1.-mycosmo['omega_M_0']
    mycosmo['omega_k_0']=0
    mycosmo['omega_b_0']=obh2/mycosmo['h']**2
    mycosmo['omega_n_0']=onh2/mycosmo['h']**2
    variables=['omega_M_0','omega_k_0','w',]
    thedata_planck=data_jimplanck

if dsname=='planck':
    dataset=[thedata_planck]
elif dsname=='planck+BAO':
    dataset=[thedata_planck,data_DR7,data_Beutler,data_Anderson]
elif dsname=='planck+BAO+LyaDR11':
    dataset=[thedata_planck,data_DR7,data_Beutler,data_Anderson,data_lyaDR11]
elif dsname=='planck+LyaDR11':
    dataset=[thedata_planck,data_lyaDR11]
elif dsname=='BAO_DR7':
    dataset=[data_DR7]
elif dsname=='BAO_Beutler':
    dataset=[data_Beutler]
elif dsname=='BAO_Anderson':
    dataset=[data_Anderson]
elif dsname=='BAO':
    dataset=[data_DR7,data_Beutler,data_Anderson]
elif dsname=='LyaDR11':
    dataset=[data_lyaDR11]
elif dsname=='LyaDR11+BAO':
    dataset=[data_lyaDR11,data_DR7,data_Beutler,data_Anderson]
elif dsname=='planck+BAO+hRiess':
    dataset=[data_planck,data_DR7,data_Beutler,data_Anderson,data_hRiess]
elif dsname=='planck+BAO+LyaDR11+hRiess':
    dataset=[data_planck,data_DR7,data_Beutler,data_Anderson,data_lyaDR11,data_hRiess]
elif dsname=='planckHp':
    dataset=[thedata_planck,data_hplanck]
elif dsname=='planckHp+BAO':
    dataset=[thedata_planck,data_hplanck,data_DR7,data_Beutler,data_Anderson]
elif dsname=='planckHp+BAO+hRiess':
    dataset=[data_planck,data_hplanck,data_DR7,data_Beutler,data_Anderson,data_hRiess]
elif dsname=='LyaDR11_Halone':
    dataset=[data_lyaDR11_Halone]
elif dsname=='LyaDR11_Halone+BAO':
    dataset=[data_lyaDR11_Halone,data_DR7,data_Beutler,data_Anderson]
elif dsname=='LyaDR11_HPlanck1s':
    dataset=[data_lyaDR11,data_hPlanck_1sig]
elif dsname=='LyaDR11_HPlanck2s':
    dataset=[data_lyaDR11,data_hPlanck_2sig]
elif dsname=='LyaDR11_HRiess1s':
    dataset=[data_lyaDR11,data_hRiess_1sig]
elif dsname=='LyaDR11_HRiess2s':
    dataset=[data_lyaDR11,data_hRiess_2sig]
elif dsname=='LyaDR11_HRiessPlanck':
    dataset=[data_lyaDR11,data_hRiessPlanck]
elif dsname=='LyaDR11_HPlanck1s_obh2Planck1s':
    dataset=[data_lyaDR11,data_hPlanck_1sig,data_obh2_Planck_1sig]
elif dsname=='LyaDR11_HPlanck2s_obh2Planck1s':
    dataset=[data_lyaDR11,data_hPlanck_2sig,data_obh2_Planck_1sig]
elif dsname=='LyaDR11_HRiess1s_obh2Planck1s':
    dataset=[data_lyaDR11,data_hRiess_1sig,data_obh2_Planck_1sig]
elif dsname=='LyaDR11_HRiess2s_obh2Planck1s':
    dataset=[data_lyaDR11,data_hRiess_2sig,data_obh2_Planck_1sig]
elif dsname=='LyaDR11_HRiessPlanck_obh2Planck1s':
    dataset=[data_lyaDR11,data_hRiessPlanck,data_obh2_Planck_1sig]
elif dsname=='LyaDR11_HPlanck1s_obh2Planck2s':
    dataset=[data_lyaDR11,data_hPlanck_1sig,data_obh2_Planck_2sig]
elif dsname=='LyaDR11_HPlanck2s_obh2Planck2s':
    dataset=[data_lyaDR11,data_hPlanck_2sig,data_obh2_Planck_2sig]
elif dsname=='LyaDR11_HRiess1s_obh2Planck2s':
    dataset=[data_lyaDR11,data_hRiess_1sig,data_obh2_Planck_2sig]
elif dsname=='LyaDR11_HRiess2s_obh2Planck2s':
    dataset=[data_lyaDR11,data_hRiess_2sig,data_obh2_Planck_2sig]
elif dsname=='LyaDR11_HRiessPlanck_obh2Planck2s':
    dataset=[data_lyaDR11,data_hRiessPlanck,data_obh2_Planck_2sig]
elif dsname=='LyaDR11_Fixhobh2PlanckRiess':
    dataset=[data_lyaDR11,data_fixed_h_obh2_PlanckRiess]
elif dsname=='LyaDR11_BAO_HRiessPlanck_obh2Planck1s':
    dataset=[data_lyaDR11,data_DR7,data_Beutler,data_Anderson,data_hRiessPlanck,data_obh2_Planck_1sig]
elif dsname=='BAO_HRiessPlanck_obh2Planck1s':
    dataset=[data_DR7,data_Beutler,data_Anderson,data_hRiessPlanck,data_obh2_Planck_1sig]
elif dsname=='LyaDR11_HRiessPlanck_obh2BBN':
    dataset=[data_lyaDR11,data_hRiessPlanck,data_obh2_BBN]
elif dsname=='LyaDR11_BAO_HRiessPlanck_obh2BBN':
    dataset=[data_lyaDR11,data_DR7,data_Beutler,data_Anderson,data_hRiessPlanck,data_obh2_BBN]
elif dsname=='BAO_HRiessPlanck_obh2BBN':
    dataset=[data_DR7,data_Beutler,data_Anderson,data_hRiessPlanck,data_obh2_BBN]

niter=500000
nburn=200000
nthin=10
print('')
print('#############################################################')
print('#    JCMC                                                   #')
print('#############################################################')
print('Variables : ')
print(variables)
print('Running the configuration: '+dsname)
name=repchains+model+'-'+dsname
print('the output will be in: '+name)
mcmc.run(niter,nburn,nthin,variables,dataset,name,cosmo=mycosmo)





