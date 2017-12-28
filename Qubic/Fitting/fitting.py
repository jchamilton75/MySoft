import scipy.integrate
import numpy as np
from matplotlib import *
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter1d
from scipy import integrate
from scipy import interpolate
from scipy import ndimage
import pymc
import iminuit


###############################################################################
################################### Fitting ###################################
###############################################################################
### Generic polynomial function ##########
def thepolynomial(x,pars):
    f=np.poly1d(pars)
    return(f(x))
    
### Generic fitting function ##############
def dothefit(x,y,covarin,guess,functname=thepolynomial,method='minuit'):
    if method == 'minuit':
        print('Fitting with Minuit')
        return(do_minuit(x,y,covarin,guess,functname))
    else:
        print('method must be among: minuit')
        return(0,0,0,0)

### Class defining the minimizer and the data
class MyChi2:
    def __init__(self,xin,yin,covarin,functname):
        self.x=xin
        self.y=yin
        self.covar=covarin
        self.invcov=np.linalg.inv(covarin)
        self.functname=functname
            
    def __call__(self,*pars):
        val=self.functname(self.x,pars)
        chi2=np.dot(np.dot(self.y-val,self.invcov),self.y-val)
        return(chi2)
        
### Call Minuit
def do_minuit(x,y,covarin,guess,functname=thepolynomial, fixpars = False):
    # check if covariance or error bars were given
    covar=covarin
    if np.size(np.shape(covarin)) == 1:
        err=covarin
        covar=np.zeros((np.size(err),np.size(err)))
        covar[np.arange(np.size(err)),np.arange(np.size(err))]=err**2
                                    
    # instantiate minimizer
    chi2=MyChi2(x,y,covar,functname)
    # variables
    ndim=np.size(guess)
    parnames=[]
    for i in range(ndim): parnames.append('c'+np.str(i))
    # initial guess
    theguess=dict(zip(parnames,guess))
    # fixed parameters
    dfix = {}
    if fixpars:
        for i in xrange(len(parnames)): dfix['fix_'+parnames[i]]=fixpars[i]
    else:
        for i in xrange(len(parnames)): dfix['fix_'+parnames[i]]=False
    #stop
    # Run Minuit
    print('Fitting with Minuit')
    theargs = dict(theguess.items() + dfix.items())
    print(theargs)
    m = iminuit.Minuit(chi2,forced_parameters=parnames,errordef=1.,**theargs)
    m.migrad()
    m.hesse()
    # build np.array output
    parfit=[]
    for i in parnames: parfit.append(m.values[i])
    errfit=[]
    for i in parnames: errfit.append(m.errors[i])
    ndimfit = int(np.sqrt(len(m.covariance)))
    covariance=np.zeros((ndimfit,ndimfit))
    if fixpars:
        parnamesfit = []
        for i in xrange(len(parnames)):
            if fixpars[i] == False: parnamesfit.append(parnames[i])
            if fixpars[i] == True: errfit[i]=0
    else:
        parnamesfit = parnames
    for i in xrange(ndimfit):
        for j in xrange(ndimfit):
            covariance[i,j]=m.covariance[(parnamesfit[i],parnamesfit[j])]

    print('Chi2=',chi2(*parfit))
    print('ndf=',np.size(x)-ndim)
    return(m,np.array(parfit), np.array(errfit), np.array(covariance))

###############################################################################
###############################################################################










###############################################################################
########################## Monte-Carlo Markov-Chains Functions ################
###############################################################################
### define data classes
class Data():
    def __init__(self, xvals=None, yvals=None, errors = None, model=None, prior=False):
        self.prior = prior
        self.model = model
        self.xvals = xvals
        self.yvals = yvals
        if not self.prior:
            if np.size(np.shape(errors)) == 1:
                self.covar=np.zeros((np.size(errors),np.size(errors)))
                self.covar[np.arange(np.size(errors)),np.arange(np.size(errors))]=errors**2
            else:
                self.covar = errors
            self.invcov = np.linalg.inv(self.covar)
    
    def __call__(self,*pars):
        if  not self.prior:
            val=self.model(self.xvals,pars[0])
            chi2=np.dot(np.dot(self.yvals-val,self.invcov),self.yvals-val)
        else:
            chi2 = self.model(self.xvals, pars[0])
        return(-0.5*chi2)

    
def generic_ll_model(datasets, allvariables, fitvariables = None, fidvalues = None, limits=None):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    if fitvariables is None:
        fitvariables = allvariables
    nvar = len(allvariables)
    if fidvalues is None:
        fidvalues = np.zeros(nvar)
    if limits is None:
        limits = np.zeros((2,nvar))
        limits[0,:] = -1000.
        limits[1,:] = 1000.
    allvars = []
    for i in xrange(nvar):
        allvars.append(pymc.Uniform(allvariables[i], limits[0,i], limits[1,i], value = fidvalues[i], 
                                 observed = allvariables[i] not in fitvariables))
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, pars = allvars):
        ll=0.
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

def run_mcmc(data, allvariables, niter=80000, nburn=20000, nthin=1, external=None,
             fitvariables = None, fidvalues = None, limits = None,
             the_ll_model=generic_ll_model,delay=1000, fid_params=None):
    if fitvariables is None: fitvariables=allvariables
    chain = pymc.MCMC(the_ll_model(data, allvariables, fitvariables = fitvariables,
                                   fidvalues=fidvalues, limits=limits))
    chain.use_step_method(pymc.AdaptiveMetropolis,chain.stochastics,delay=delay)
    chain.sample(iter=niter,burn=nburn,thin=nthin)
    ch ={}
    for v in fitvariables: ch[v] = chain.trace(v)[:]
    return ch

def best_fit(chain, verbose=True):
    pars = np.sort(chain.keys())
    npars = len(pars)
    sz = len(chain[pars[0]])
    means = np.zeros(npars)
    marg_std = np.zeros(npars)
    allvals = np.zeros((npars, sz))
    for i in xrange(len(pars)):
        means[i] = np.mean(chain[pars[i]])
        marg_std[i] = np.std(chain[pars[i]])
        allvals[i,:] = chain[pars[i]]
    covariance = np.cov(allvals)
    errorbars = np.sqrt(np.diag(covariance))
    if verbose:
        for i in xrange(len(pars)):
            print(pars[i]+' : {} +/- {}'.format(means[i], errorbars[i]))
        print('Covariance Matrix:')
        print(covariance)
    return means, errorbars, covariance, marg_std

def matrixplot(chain,vars,col,sm,limits=None,nbins=None,doit=None,alpha=0.7,
               labels=None, printbestfit=False, addpoints=None):
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
                xlim(limits[i])
                ylim(0,1.2)
                if (var in chain.keys()) and (doit[j]==True):
                    if nbins is None: nbins=100
                    bla=np.histogram(chain[var],bins=nbins,normed=True)
                    xhist=(bla[1][0:nbins]+bla[1][1:nbins+1])/2
                    yhist=gaussian_filter1d(bla[0],ss[i]/5/(xhist[1]-xhist[0]))
                    plot(xhist,yhist/max(yhist),color=col, label = '{0:.2g} +/- {1:.2g}'.format(np.mean(chain[var]), np.std(chain[var])))
                    if printbestfit: plot([np.mean(chain[var]), np.mean(chain[var])],[0,1.2],'k--')
                    if addpoints:
                        addnames = addpoints[0]
                        addcol = addpoints[1]
                        addvals = addpoints[2]
                        plot([addvals[i], addvals[i]], [0., 1.2], ':', color=addcol)
                    legend(loc='upper left',frameon=False,fontsize=8)
            if (i>j):
                a=subplot(nplots,nplots,num)
                a.tick_params(labelsize=8)
                var0=labels[j]
                var1=labels[i]
                xlim(limits[j])
                ylim(limits[i])
                if i == nplots-1: xlabel(var0)
                if j == 0: ylabel(var1)
                if (vars[i] in chain.keys()) and (vars[j] in chain.keys()) and (doit[j]==True) and (doit[i]==True):
                    a0=cont(chain[vars[j]],chain[vars[i]],color=col,nsmooth=sm,alpha=alpha)
                    if printbestfit: plot(np.mean(chain[vars[j]]),np.mean(chain[vars[i]]), 'k+',lw=2)
                    if addpoints:
                        addnames = addpoints[0]
                        addcol = addpoints[1]
                        addvals = addpoints[2]
                        plot(addvals[j], addvals[i], '+', color=addcol)
    return(a0)
    
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
    return(cols)


def cont(x,y,xlim=None,ylim=None,levels=[0.9545,0.6827],alpha=0.7,color='blue',nbins=256,nsmooth=4,Fill=True,**kwargs):
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

###############################################################################
###############################################################################
            

