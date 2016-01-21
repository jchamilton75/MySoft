import numpy as np
from matplotlib import rc
rc('text', usetex=False)
import glob
import pickle

######### Planck Legacy Archive 2013 ##################################
#http://www.sciops.esa.int/wikiSI/planckpla/index.php?title=Cosmological_Parameters&instance=Planck_Public_PLA
#planck     high-L Planck temperature (CamSpec, 50 <= l <= 2500)
#lowl       low-L: Planck temperature (2 <= l <= 49)
#lensing    Planck lensing power spectrum reconstruction
#lowLike    low-L WMAP 9 polarization (WP)
#tauprior	A Gaussian prior on the optical depth, tau = 0.09 +- 0.013
#BAO        Baryon oscillation data from DR7, DR9 and and 6DF
#SNLS       Supernova data from the Supernova Legacy Survey
#Union2     Supernova data from the Union compilation
#HST        Hubble parameter constraint from HST (Riess et al)
#WMAP       The full WMAP (temperature and polarization) 9 year data


#### Base directory for Planck MCMC chains from Planck Legacy Archive 2013
rep='/Volumes/Data/ChainsPlanck/PLA/'

#### Relevant parameter sets:
partype=['base','base_omegak','base_w']
pars=[['omegabh2','omegach2','theta'],
      ['omegabh2','omegach2','theta','omegak'],
      ['omegabh2','omegach2','theta','w']]

configs=['planck_lowl_lowLike','planck_lowl_lowLike_highL','planck_lowl_lowLike_highl_lensing']


for i in np.arange(np.size(partype)):
    for j in np.arange(np.size(configs)):
        thepar=partype[i]
        theconf=configs[j]
        params=pars[i]
        listchains=glob.glob(rep+thepar+'/'+theconf+'/'+thepar+'_'+theconf+'_[0-9].txt')
        if np.size(listchains) != 0:
            print(rep+thepar+'/'+theconf+'/'+thepar+'_'+theconf)
            ### file with parameter names
            parnamefile=rep+thepar+'/'+theconf+'/'+thepar+'_'+theconf+'.paramnames'
            ### read planck MCMC
            planck_chains=np.loadtxt(listchains[0])
            for num in np.arange(np.size(listchains)-1)+1:
                planck_chains=np.append(planck_chains,np.loadtxt(listchains[num]),axis=0)
            names=np.loadtxt(parnamefile,dtype='str',usecols=[0])
            planck=dict([names[i],planck_chains[:,i+2]] for i in range(np.size(names)))
            ### Calculate statistics
            mpar=np.zeros(np.size(params))
            spar=np.zeros(np.size(params))
            for k in np.arange(np.size(params)):
                mpar[k]=np.mean(planck[params[k]])
                spar[k]=np.std(planck[params[k]])
            covpar=np.zeros((np.size(params),np.size(params)))
            for k in np.arange(np.size(params)):
                for l in np.arange(np.size(params)):
                    covpar[k,l]=np.mean((planck[params[k]]-mpar[k])*(planck[params[l]]-mpar[l]))
            ### Some Output
            print(params)
            print(covpar)
            ### write stat files
            print('Writing Stat file')
            data={'params':params, 'mean':mpar, 'sig':spar, 'covar':covpar}
            output=open(str('stats_'+thepar+'_'+theconf+'.pkl'),'wb')
            pickle.dump(data,output)
            output.close()
            ### write chain files
            print('Writing Chain file')
            output=open(str('chains_'+thepar+'_'+theconf+'.pkl'),'wb')
            pickle.dump(planck,output)
            output.close()
            print('')






