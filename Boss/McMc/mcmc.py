import scipy.stats
import numpy as np
from matplotlib import *
from matplotlib.pyplot import *
from scipy import integrate
from scipy import interpolate
from scipy import ndimage
import cosmolopy
import pymc
from matplotlib import rc
rc('text', usetex=False)
import pickle
from scipy.ndimage import gaussian_filter1d
from McMc import cosmo_utils

### Requirements to use this:
# install cosmolopy
# install camb
# install the JCH-modified version of pycamb
# then it should work...



fidcosmo=cosmolopy.fidcosmo.copy()
### we use the PlanckWP2013 LambdaCDM cosmology as the fiducial model
fidcosmo['h']=0.6704
fidcosmo['Y_He']=0.247710
obh2=0.022032
onh2=0.000645
och2=0.120376-onh2
fidcosmo['omega_M_0']=(och2+obh2+onh2)/fidcosmo['h']**2
fidcosmo['omega_lambda_0']=1.-fidcosmo['omega_M_0']
fidcosmo['omega_k_0']=0
fidcosmo['omega_b_0']=obh2/fidcosmo['h']**2
fidcosmo['omega_n_0']=onh2/fidcosmo['h']**2
fidcosmo['n']=0.9619123
fidcosmo['Num_Nu_massless']=3.046/3*2
fidcosmo['Num_Nu_massive']=3.046/3

def stats(chain):
    print(np.mean(chain),np.std(chain))

def getcols(color):
    if color is 'blue':
        cols=['SkyBlue','MediumBlue']
    elif color is 'red':
        cols=['LightCoral','Red']
    elif color is 'green':
        cols=['LightGreen','Green']
    elif color is 'pink':
        cols=['LightPink','HotPink']
    elif color is 'orange':
        cols=['Coral','OrangeRed']
    elif color is 'yellow':
        cols=['Yellow','Gold']
    elif color is 'purple':
        cols=['Violet','DarkViolet']
    elif color is 'brown':
        cols=['BurlyWood','SaddleBrown']
    elif color is 'black':
        cols=['Grey','Black']
    return(cols)

def cont(x,y,xlim=None,ylim=None,levels=[0.95,0.683],alpha=0.7,color='blue',nbins=256,nsmooth=4,Fill=True,**kwargs):
    levels.sort()
    levels.reverse()
    cols=getcols(color)
    dx=np.max(x)-np.min(x)
    dy=np.max(y)-np.min(y)
    if xlim is None: xlim=[np.min(x)-dx/3,np.max(x)+dx/3]
    if ylim is None: ylim=[np.min(y)-dy/3,np.max(y)+dy/3]
    range=[xlim,ylim]

    a,xmap,ymap=scipy.histogram2d(x,y,bins=256,range=range)
    a=np.transpose(a)
    xmap=xmap[:-1]
    ymap=ymap[:-1]
    dx=xmap[1]-xmap[0]
    dy=ymap[1]-ymap[0]
    z=scipy.ndimage.filters.gaussian_filter(a,nsmooth)
    z=z/np.sum(z)/dx/dy
    sz=np.sort(z.flatten())[::-1]
    cumsz=integrate.cumtrapz(sz)
    cumsz=cumsz/max(cumsz)
    f=interpolate.interp1d(cumsz,np.arange(np.size(cumsz)))
    indices=f(levels).astype('int')
    vals=sz[indices].tolist()
    vals.append(np.max(sz))
    vals.sort()
    if Fill:
        for i in np.arange(np.size(levels)):
            contourf(xmap, ymap, z, vals[i:i+2],colors=cols[i],alpha=alpha,**kwargs)
    else:
        contour(xmap, ymap, z, vals[0:1],colors=cols[1],**kwargs)
        contour(xmap, ymap, z, vals[1:2],colors=cols[1],**kwargs)
    a=Rectangle((np.max(xmap),np.max(ymap)),0.1,0.1,fc=cols[1])
    return(a)



def cont_gkde(x,y,levels=[0.95,0.683],nsig=10,alpha=0.7,color='blue',fill=True,**kwargs):
    levels.sort()
    levels.reverse()
    cols=getcols(color)

    mx=np.mean(x)
    my=np.mean(y)
    sx=np.std(x)
    sy=np.std(y)
    szx=sx/nsig
    szy=sy/nsig
    minx=mx-4*sx
    maxx=mx+4*sx
    miny=my-4*sy
    maxy=my+4*sy
    xx,yy = np.mgrid[minx:maxx:szx, miny:maxy:szy]
    dx=xx[1,0]-xx[0,0]
    dy=yy[0,1]-yy[0,0]
    gkde=scipy.stats.gaussian_kde([x,y])
    z = np.array(gkde.evaluate([xx.flatten(),yy.flatten()])).reshape(xx.shape)
    sz=np.sort(z.flatten())[::-1]
    cumsz=integrate.cumtrapz(sz)
    cumsz=cumsz/max(cumsz)
    f=interpolate.interp1d(cumsz,np.arange(np.size(cumsz)))
    indices=f(levels).astype('int')
    vals=sz[indices].tolist()
    vals.append(np.max(sz))
    vals.sort()
    if fill:
        for i in np.arange(np.size(levels)):
            contourf(xx, yy, z, vals[i:i+2],colors=cols[i],alpha=alpha,**kwargs)
    else:
        contour(xx, yy, z, vals[0:1],colors=cols[0],alpha=alpha,**kwargs)
        contour(xx, yy, z, vals[1:2],colors=cols[1],alpha=alpha)
    a=Rectangle((np.max(xx),np.max(yy)),0.1,0.1,fc=cols[1])
    return(a)



def generic_model(datasets,variables=['h','omega_M_0','omega_k_0'],flat=False,range=3,library='astropy',cosmo=fidcosmo):
    ### define distributions
    N_nu = pymc.Uniform('N_nu',0.,5.,value=cosmo['N_nu'],observed='N_nu' not in variables)
    Y_He = pymc.Uniform('Y_He',0.,1.,value=cosmo['Y_He'],observed='Y_He' not in variables)
    h = pymc.Uniform('h',0.,1.5,value=cosmo['h'],observed='h' not in variables)
    n = pymc.Uniform('n',0.,2.,value=cosmo['n'],observed='n' not in variables)
    omega_M_0 = pymc.Uniform('omega_M_0',0,2.,value=cosmo['omega_M_0'],observed='omega_M_0' not in variables)
    omega_b_0 = pymc.Uniform('omega_b_0',0.,0.1,value=cosmo['omega_b_0'],observed='omega_b_0' not in variables)
    omega_k_0 = pymc.Uniform('omega_k_0',-1.,4.,value=cosmo['omega_k_0'],observed='omega_k_0' not in variables)
    omega_n_0 = pymc.Uniform('omega_n_0',0.,1.,value=cosmo['omega_n_0'],observed='omega_n_0' not in variables)
    sigma_8 = pymc.Uniform('sigma_8',0.,2.,value=cosmo['sigma_8'],observed='sigma_8' not in variables)
    t_0 = pymc.Uniform('t_0',0.,20.,value=cosmo['t_0'],observed='t_0' not in variables)
    tau = pymc.Uniform('tau',0.,1.,value=cosmo['tau'],observed='tau' not in variables)
    w = pymc.Uniform('w',-3,1,value=cosmo['w'],observed='w' not in variables)
    z_reion = pymc.Uniform('z_reion',0,30,value=cosmo['z_reion'],observed='z_reion' not in variables)
    nmassless = pymc.Uniform('Num_Nu_massless',0,5,value=cosmo['Num_Nu_massless'],observed='Num_Nu_massless' not in variables)
    nmassive = pymc.Uniform('Num_Nu_massive',0,5,value=cosmo['Num_Nu_massive'],observed='Num_Nu_massive' not in variables)
    if flat:
        print('Forcing Flat Geometry')
        @pymc.deterministic(plot=False)
        def omega_k_0():
            return(0.)
    @pymc.deterministic(plot=False)
    def omega_lambda_0(omega_M_0=omega_M_0,omega_k_0=omega_k_0):
        return(1.-omega_M_0-omega_k_0)
        
    
    ### Log-Likelihood function
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, N_nu=N_nu, Y_He=Y_He, h=h, n=n, omega_M_0=omega_M_0, omega_b_0=omega_b_0, omega_lambda_0=omega_lambda_0, omega_n_0=omega_n_0, sigma_8=sigma_8, t_0=t_0, tau=tau, w=w, z_reion=z_reion,nmassless=nmassless,nmassive=nmassive,library=library):
        ll=0.
        for ds in datasets:
            ll=ll+ds.log_likelihood(N_nu=N_nu, Y_He=Y_He, h=h, n=n, omega_M_0=omega_M_0, omega_b_0=omega_b_0, omega_lambda_0=omega_lambda_0, omega_n_0=omega_n_0, sigma_8=sigma_8, t_0=t_0, tau=tau, w=w, z_reion=z_reion,nmassless=nmassless,nmassive=nmassive,library=library)
        return(ll)
    return(locals())



def get_cosmology(N_nu,Y_He,h,n,omega_M_0,omega_b_0,omega_lambda_0,omega_n_0,sigma_8,t_0,tau,w,z_reion,nmassless,nmassive,library='astropy'):
    cosmo=fidcosmo.copy()
    cosmo['N_nu']=N_nu.flatten()[0]
    cosmo['Y_He']=Y_He.flatten()[0]
    cosmo['h']=h.flatten()[0]
    cosmo['n']=n.flatten()[0]
    cosmo['omega_M_0']=omega_M_0.flatten()[0]
    cosmo['omega_b_0']=omega_b_0.flatten()[0]
    cosmo['omega_lambda_0']=omega_lambda_0.flatten()[0]
    cosmo['omega_n_0']=omega_n_0.flatten()[0]
    cosmo['omega_k_0']=1.-omega_M_0.flatten()[0]-omega_lambda_0.flatten()[0]
    cosmo['sigma_8']=sigma_8.flatten()[0]
    cosmo['t_0']=t_0.flatten()[0]
    cosmo['tau']=tau.flatten()[0]
    #### note the weird formula here due to a bug in cosmolopy e_z 
    if library == 'cosmolopy':
        cosmo['w']=(w.flatten()[0]+1)*3-1
        #print('get_cosmology: using cosmolopy')
    else:
        cosmo['w']=w.flatten()[0]
        #print('get_cosmology: using astropy or jc')
    cosmo['z_reion']=z_reion.flatten()[0]
    cosmo['Num_Nu_massless']=nmassless.flatten()[0]
    cosmo['Num_Nu_massive']=nmassive.flatten()[0]
    return(cosmo)

def run(niter,nburn,nthin,variables,dataset,name,library='jc',flat=False,cosmo=fidcosmo):
    chain=pymc.MCMC(generic_model(dataset,variables=variables,library=library,flat=flat,cosmo=fidcosmo),db='pickle',dbname=name+'.db')
    chain.use_step_method(pymc.AdaptiveMetropolis,chain.stochastics,delay=1000)
    chain.sample(iter=niter,burn=nburn,thin=nthin)
    chain.db.close()


def readchains(filename,add_extra=None):
    pkl_file=open(filename,'rb')
    data=pickle.load(pkl_file)
    pkl_file.close()
    print(data.keys())
    if 'omega_b_0' in data.keys():
        obh2=data['omega_b_0'][0]*data['h'][0]**2
        och2=(data['omega_M_0'][0]-data['omega_b_0'][0]-fidcosmo['omega_n_0'])*data['h'][0]**2
    bla={}
    dk=data.keys()
    maskok = data['omega_lambda_0'][0] >= 0
    for kk in data.keys(): 
        if ((kk != '_state_') and (kk != 'deviance')):
            bla[kk]=data[kk][0][maskok]
    if 'omega_k_0' not in bla.keys(): bla['omega_k_0']=1.-bla['omega_M_0']-bla['omega_lambda_0']
    if 'omega_b_0' in data.keys():
        bla['obh2']=obh2[maskok]
        bla['och2']=och2[maskok]
    if add_extra:
        bla['om_ol']=bla['omega_M_0']/bla['omega_lambda_0']
        bla['rs']=get_rs_values(bla)
        bla['rssqrtomh2']=bla['rs']*np.sqrt(bla['omega_M_0']*bla['h']**2)
        bla['c_H0rs']=3e5/(100*bla['h']*bla['rs'])
    return(bla)

def get_rs_values(data):
    nn=len(data['h'])
    rs=np.zeros(nn)
    mycosmo=cosmolopy.fidcosmo.copy()
    mycosmo['Y_He']=0.247710
    mycosmo['n']=0.9619123
    mycosmo['omega_n_0']=0
    mycosmo['Num_Nu_massless']=3.046/3*2
    mycosmo['Num_Nu_massive']=3.046/3
    if 'w' in data.keys():
        wvals=data['w']
    else:
        wvals=np.zeros(nn)-1
        
    for i in np.arange(nn):
        cosmo_utils.progress_bar(i,nn)
        mycosmo['h']=data['h'][i]
        mycosmo['omega_M_0']=data['omega_M_0'][i]
        mycosmo['omega_lambda_0']=data['omega_lambda_0'][i]
        mycosmo['omega_k_0']=data['omega_k_0'][i]
        mycosmo['omega_b_0']=data['omega_b_0'][i]
        mycosmo['w']=wvals[i]
        rs[i]=cosmo_utils.rs(**mycosmo)

    return(rs)


def matrixplot(chain,vars,col,sm,limits=None,nbins=None,doit=None,alpha=0.7,labels=None,histn=2, truevals=None):
    nplots=len(vars)
    if labels is None: labels = vars
    if doit is None: doit=np.repeat([True],nplots)
    mm=np.zeros(nplots)
    ss=np.zeros(nplots)
    for i in np.arange(nplots):
        if vars[i] in chain.keys():
            mm[i]=np.mean(chain[vars[i]])
            ss[i]=np.std(chain[vars[i]])
    if limits is None:
        limits=[]
        for i in np.arange(nplots):
            limits.append([mm[i]-3*ss[i],mm[i]+3*ss[i]])
    num=0
    for i in np.arange(nplots):
         for j in np.arange(nplots):
            num+=1
            if (i == j):
                a=subplot(nplots,nplots,num)
                a.tick_params(labelsize=8)
                if i == nplots-1: xlabel(labels[j])
                var=vars[j]
                if truevals: plot([truevals[i], truevals[i]],[0,1.2], 'k:') 
                xlim(limits[i])
                ylim(0,1.2)
                if (var in chain.keys()) and (doit[j]==True):
                    if nbins is None: nbins=100
                    bla=np.histogram(chain[var],bins=nbins,normed=True)
                    xhist=(bla[1][0:nbins]+bla[1][1:nbins+1])/2
                    yhist=gaussian_filter1d(bla[0],ss[i]/histn/(xhist[1]-xhist[0]), mode='nearest')
                    plot(xhist,yhist/max(yhist),color=col)
            if (i>j):
                a=subplot(nplots,nplots,num)
                a.tick_params(labelsize=8)
                var0=labels[j]
                var1=labels[i]
                if truevals: 
                    plot(limits[j], [truevals[i],truevals[i]], 'k:')
                    plot([truevals[j], truevals[j]], limits[i], 'k:')
                xlim(limits[j])
                ylim(limits[i])
                if i == nplots-1: xlabel(var0)
                if j == 0: ylabel(var1)
                if (vars[i] in chain.keys()) and (vars[j] in chain.keys()) and (doit[j]==True) and (doit[i]==True):
                    a0=cont(chain[vars[j]],chain[vars[i]],color=col,nsmooth=sm,alpha=alpha)
    subplot(nplots, nplots, nplots)
    axis('off')
    return(a0)
    
          





