import scipy
import scipy.special
import numpy as np
import math
from Quad import pyquad
import healpy as hp
import multiprocessing
import matplotlib.pyplot as mp



class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                #print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break
            #print '******* %s: %s' % (proc_name, next_task)
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return




def cov_from_maps(maps0,maps1):
    sh=np.shape(maps0)
    npix=sh[1]
    nbmc=sh[0]
    covmc=np.zeros((npix,npix))
    mm0=np.mean(maps0,axis=0)
    mm1=np.mean(maps1,axis=0)
    themaps0=np.zeros((nbmc,npix))
    themaps1=np.zeros((nbmc,npix))
    for i in np.arange(npix):
        pyquad.progress_bar(i,npix)
        themaps0[:,i]=maps0[:,i]-mm0[i]
        themaps1[:,i]=maps1[:,i]-mm1[i]
    for i in np.arange(npix):
        pyquad.progress_bar(i,npix)
        for j in np.arange(npix):
            covmc[i,j]=np.mean(themaps0[:,i]*themaps1[:,j])
            #covmc[j,i]=covmc[i,j]
    return(covmc)


##### polynomes de legendre et fonctions F1 et F2
## il est vraiment inutile de reconder les polynomes de legendre: le code python et ~500 fois plus
## rapide qu'en codant la formule de recurrence

################################## Polynomes de legendre (m=0) ####################################
#### tres lent
def pl0rec(z,lmax):
    p=np.zeros(lmax+1)
    p[0]=1.
    p[1]=z
    for i in np.arange(2,lmax+1):
        p[i]=((2*i-1)*z*p[i-1]-(i-1)*p[i-2])/i
    return(p)

#### Fast
def pl0(z,lmax):
    return(scipy.special.lpn(lmax,z)[0])

################################## Polynomes de legendre associes ####################################
#### tres lent
def pl2rec(z,lmax):
    p=np.zeros(lmax+1)
    p[2]=3*(1-z**2)
    p[3]=5*z*p[2]
    for i in np.arange(4,lmax+1):
        p[i]=((2*i-1)*z*p[i-1]-(i+1)*p[i-2])/(i-2)
    return(p)

#### Fast
def pl2(z,lmax):
    return(scipy.special.lpmn(2,lmax,z)[0][2])


#### Fact(n)/Fact(m)
def factratio(n,m):
    mask = (n >= 0) & (m >= 0)
    res=np.zeros(n.shape)
    res[mask]=np.exp(scipy.special.gammaln(n[mask]+1)-scipy.special.gammaln(m[mask]+1))
    return res

################################# F1 and F2 functions ###############################################
############# They come from Tegmark & De Oliveira-Costa, 2000 ######################################
def F1l0(z,lmax):
    if z==1.0:
        return(np.zeros(lmax+1))
    else:
        ell=np.arange(lmax+1)
        thepl=pl0(z,lmax)
        theplm1=np.append(0,pl0(z,lmax-1))
        a0=2./np.sqrt((ell-1)*ell*(ell+1)*(ell+2))
        a1=ell*z*theplm1/(1-z**2)
        a2=(ell/(1-z**2)+ell*(ell-1)/2)*thepl
        bla=a0*(a1-a2)
        bla[0]=0
        bla[1]=0
        return bla

def F1l2(z,lmax):
    if z==1.0:
        return(np.ones(lmax+1)*0.5)
    elif z==-1.0:
        ell=np.arange(lmax+1)
        return(0.5*(-1)**ell)
    else:
        ell=np.arange(lmax+1)
        thepl2=pl2(z,lmax)
        theplm1_2=np.append(0,pl2(z,lmax-1))
        a0=2./((ell-1)*ell*(ell+1)*(ell+2))
        a1=(ell+2)*z*theplm1_2/(1-z**2)
        a2=((ell-4)/(1-z**2)+ell*(ell-1)/2)*thepl2
        bla=a0*(a1-a2)
        bla[0]=0
        bla[1]=0
        return bla

def F2l2(z,lmax):
    if z==1.0:
        return(-0.5*np.ones(lmax+1))
    elif z==-1.0:
        ell=np.arange(lmax+1)
        return(0.5*(-1)**ell)
    else:
        ell=np.arange(lmax+1)
        thepl2=pl2(z,lmax)
        theplm1_2=np.append(0,pl2(z,lmax-1))
        a0=4./((ell-1)*ell*(ell+1)*(ell+2)*(1-z**2))
        a1=(ell+2)*theplm1_2
        a2=(ell-1)*z*thepl2
        bla=a0*(a1-a2)
        bla[0]=0
        bla[1]=0
        return bla
################################################################################################

################################################################################################
############## Other version of F1 F2 functions thatdon't give good covariance matrices for IQU
############## it is an attempt to be the ones described in Zaldarriaga's papers and PhD...
############## Don't know why they don't work
def F1l0_zal(z,lmax):
    ell=np.arange(lmax+1)
    thepl2=pl2(z,lmax)
    bla=np.sqrt(factratio(ell-2,ell+2))
    return np.nan_to_num(bla*thepl2)

def F1l2_zal(z,lmax):
    ell=np.arange(lmax+1)
    thepl2=pl2(z,lmax)
    theplm1_2=np.append(0,pl2(z,lmax-1))
    bla=factratio(ell-2,ell+2)
    cth=np.cos(z)
    sth2=np.sin(z)**2
    return np.nan_to_num(2*bla*( -((ell**2-4)/sth2 + 0.5*ell*(ell-1))*thepl2 + (ell+2)*cth/sth2*theplm1_2 ))

def F2l2_zal(z,lmax):
    ell=np.arange(lmax+1)
    thepl2=pl2(z,lmax)
    theplm1_2=np.append(0,pl2(z,lmax-1))
    bla=factratio(ell-2,ell+2)
    cth=np.cos(z)
    sth2=np.sin(z)**2
    return np.nan_to_num(4*bla/sth2*( -(ell-1)*cth*thepl2 + (ell+2)*theplm1_2))
#################################################################################################



################################# Rotation Angles ###############################################
def cosangij(ri,rj):
    return np.dot(ri,np.transpose(rj))

def crosspn(x1,x2):
    xr=np.zeros(3)
    xr[0] = x1[1]*x2[2] - x1[2]*x2[1]
    xr[1] = x1[2]*x2[0] - x1[0]*x2[2]
    xr[2] = x1[0]*x2[1] - x1[1]*x2[0]
    return xr

def polrotangle(ri,rj):
    # Adapted from a code by M. Tristram 08/10/2013
    z=np.array([0.,0.,1.])

    # Compute ri^rj : unit vector for the great circle connecting i and j
    rij=np.cross(ri,rj)
    norm=np.sqrt(np.dot(rij,np.transpose(rij)))
    # case where pixels are identical or diametrically opposed on the sky
    if norm<=1e-15:
        cos2a=1.
        sin2a=0.
        return cos2a,sin2a
    rij=rij/norm

    # Compute z^ri : unit vector for the meridian passing through pixel i
    ris=np.cross(z,ri)
    norm=np.sqrt(np.dot(ris,np.transpose(ris)))
    # case where pixels is at the pole
    if norm<=1e-15:
        cos2a=1.
        sin2a=0.
        return cos2a,sin2a
    ris=ris/norm

    # Now, the angle we want is that between these two great circles: defined by
    cosa=np.dot(rij,np.transpose(ris))
    # the sign is more subtle : see tegmark et de oliveira costa 2000 eq. A6
    rijris=np.cross(rij,ris)
    sina=np.dot(rijris,np.transpose(ri))
    # so now we have directly cos2a and sin2a
    cos2a=2.*cosa*cosa-1.
    sin2a=2.*cosa*sina
    return cos2a,sin2a
################################################################################################    



################################################################################################    

def compute_ds_dcb(ellbins,nside,ipok,bl,polar=True,temp=True,EBTB=False, verbose=False):
    if verbose: print('dS/dCb Calulation:')
    if verbose: print('Temp='+str(temp))
    if verbose: print('Polar='+str(polar))
    if verbose: print('EBTB='+str(EBTB))
    # order of the derivatives ell, TT, EE, BB, TE, EB, TB
    all=['I','Q','U']
    if EBTB:
        der=['TT','EE','BB','TE','EB','TB']
        ind=[1,2,3,4,5,6]
    else:
        der=['TT','EE','BB','TE']
        ind=[1,2,3,4]
        
    if  not temp:
        all=['Q','U']
        if EBTB:
            der=['EE','BB','EB']
            ind=[2,3,5]
        else:
            der=['EE','BB']
            ind=[2,3]
    if not polar:
        all=['I']
        der=['TT']
        ind=[1]
    if verbose: print('Stokes parameters :',all)
    if verbose: print('Derivatives w.r.t. :',der)
    ### prepare spectra with zeros and ones for calculating the derivatives
    ell=np.arange(np.min(ellbins),np.max(ellbins)+1)
    speczero=np.zeros(len(ell))
    specone=np.ones(len(ell))
    nder=len(der)
    if EBTB:
        allspeczero=[ell,speczero,speczero,speczero,speczero,speczero,speczero]
    else:
        allspeczero=[ell,speczero,speczero,speczero,speczero]
    
    #### define pixels
    rpix=np.array(hp.pix2vec(nside,ipok))
    allcosang=np.dot(np.transpose(rpix),rpix)

    ### dimensions of the large matrix and loop for filling it
    nstokes=len(all)
    nbins=len(ellbins)-1
    npix=len(ipok)
    dcov=np.zeros((nder,nbins,nstokes*npix,nstokes*npix))        
    for i in np.arange(nder):
        thespec=list(allspeczero)
        thespec[ind[i]]=specone
        if verbose: print('Calculating dS/dCb for '+der[i])
        dcov[i,:,:,:]=covth_bins(ellbins,nside,ipok,allcosang,bl,thespec,polar=polar,temp=temp)
        
    return dcov



########## Parallel version of the above core: each threads calculates one derivative
########## This is not very efficient of course as there are more processors than derivatives,
########## but still represents a significant gain
def compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=True,temp=True,EBTB=False):
    print('dS/dCb Calulation:')
    print('Temp='+str(temp))
    print('Polar='+str(polar))
    print('EBTB='+str(EBTB))
    # order of the derivatives ell, TT, EE, BB, TE, EB, TB
    all=['I','Q','U']
    if EBTB:
        der=['TT','EE','BB','TE','EB','TB']
        ind=[1,2,3,4,5,6]
    else:
        der=['TT','EE','BB','TE']
        ind=[1,2,3,4]
        
    if  not temp:
        all=['Q','U']
        if EBTB:
            der=['EE','BB','EB']
            ind=[2,3,5]
        else:
            der=['EE','BB']
            ind=[2,3]
    if not polar:
        all=['I']
        der=['TT']
        ind=[1]
    print('Stokes parameters :',all)
    print('Derivatives w.r.t. :',der)
    ### prepare spectra with zeros and ones for calculating the derivatives
    ell=np.arange(np.min(ellbins),np.max(ellbins)+1)
    speczero=np.zeros(len(ell))
    specone=np.ones(len(ell))
    nder=len(der)
    if EBTB:
        allspeczero=[ell,speczero,speczero,speczero,speczero,speczero,speczero]
    else:
        allspeczero=[ell,speczero,speczero,speczero,speczero]
    
    #### define pixels
    rpix=np.array(hp.pix2vec(nside,ipok))
    allcosang=np.dot(np.transpose(rpix),rpix)

    ### dimensions of the large matrix and loop for filling it
    nstokes=len(all)
    nbins=len(ellbins)-1
    npix=len(ipok)
    dcov=np.zeros((nder,nbins,nstokes*npix,nstokes*npix))        
    # initiate multithreading
    tasks=multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    # start consumers
    num_consumers = nder
    consumers = [ Consumer(tasks, results)
                  for i in xrange(num_consumers) ]
    for w in consumers:
        w.start()
    for i in np.arange(num_consumers):
        thespec=list(allspeczero)
        thespec[ind[i]]=specone
        print('I am consumer #{} - Calculating dS/dCb for {}:'.format(i,der[i]))
        tasks.put(Task(i,ellbins,nside,ipok,allcosang,bl,thespec,polar,temp))
    # poison them when finished
    for i in np.arange(num_consumers):
        tasks.put(None)
    tasks.join()
    while num_consumers:
        result = results.get()
        bla=result[0]
        num=result[1]
        dcov[num,:,:,:]=bla
        num_consumers -= 1
    tasks.close()
    return dcov

class Task(object):
    def __init__(self, i, ellbins,nside,ipok,allcosang,bl,thespec,polar,temp):
        self.i = i
        self.ellbins = ellbins
        self.nside = nside
        self.ipok = ipok
        self.allcosang=allcosang
        self.bl = bl
        self.thespec = thespec
        self.polar = polar
        self.temp = temp
    def __call__(self):
        aa=covth_bins(self.ellbins,self.nside,self.ipok,self.allcosang,self.bl,self.thespec,polar=self.polar,temp=self.temp)
        return([aa,self.i])

def compute_ds_dcb_parpix(ellbins,nside,ipok,bl,polar=True,temp=True,EBTB=False, verbose=False, nprocs=6):
    if verbose: print('dS/dCb Calulation:')
    if verbose: print('Temp='+str(temp))
    if verbose: print('Polar='+str(polar))
    if verbose: print('EBTB='+str(EBTB))
    # order of the derivatives ell, TT, EE, BB, TE, EB, TB
    all=['I','Q','U']
    if EBTB:
        der=['TT','EE','BB','TE','EB','TB']
        ind=[1,2,3,4,5,6]
    else:
        der=['TT','EE','BB','TE']
        ind=[1,2,3,4]
        
    if  not temp:
        all=['Q','U']
        if EBTB:
            der=['EE','BB','EB']
            ind=[2,3,5]
        else:
            der=['EE','BB']
            ind=[2,3]
    if not polar:
        all=['I']
        der=['TT']
        ind=[1]
    if verbose: print('Stokes parameters :',all)
    if verbose: print('Derivatives w.r.t. :',der)
    ### prepare spectra with zeros and ones for calculating the derivatives
    ell=np.arange(np.min(ellbins),np.max(ellbins)+1)
    speczero=np.zeros(len(ell))
    specone=np.ones(len(ell))
    nder=len(der)
    if EBTB:
        allspeczero=[ell,speczero,speczero,speczero,speczero,speczero,speczero]
    else:
        allspeczero=[ell,speczero,speczero,speczero,speczero]
    
    #### define pixels
    rpix=np.array(hp.pix2vec(nside,ipok))

    ### dimensions of the large matrix and loop for filling it
    nstokes=len(all)
    nbins=len(ellbins)-1
    npix=len(ipok)
    dcov=np.zeros((nder,nbins,nstokes*npix,nstokes*npix))        
    for i in np.arange(nder):
        thespec=list(allspeczero)
        thespec[ind[i]]=specone
        if verbose: print('Calculating dS/dCb for '+der[i])
        dcov[i,:,:,:]=covth_bins_par(ellbins,nside,ipok,bl,thespec,polar=polar,temp=temp, allinone = True, verbose=True, nprocs=nprocs)
        
    return dcov



def covth_bins_par(ellbins,nside,ipok,bl,spectra,polar=True,temp=True,allinone=True, verbose=False, nprocs=6):
    #### define bins in ell
    nbins=len(ellbins)-1
    minell=np.array(ellbins[0:nbins])
    maxell=np.array(ellbins[1:nbins+1])-1
    maxell[nbins-1]=np.max(ellbins)
    ellval=(minell+maxell)*0.5
    lmax=np.max(ellbins)
    #maxell[nbins-1]=lmax
    if verbose: print('minell:',minell)
    if verbose: print('maxell:',maxell)
    #### define Stokes
    all=['I','Q','U']
    if  not temp:
        all=['Q','U']
    if not polar:
        all=['I']

    #### define pixels
    rpix=np.array(hp.pix2vec(nside,ipok))
    
    #### define Pixel window function
    pixwin=hp.pixwin(nside)[0:lmax+1]

    #### get ell values from spectra and restrict spectra to good values
    nspec,nell=np.shape(spectra)
    ell=spectra[0]
    maskl=ell<(lmax+1)
    ell=ell[maskl]
    ctt=spectra[1][maskl]
    cee=spectra[2][maskl]
    cbb=spectra[3][maskl]
    cte=spectra[4][maskl]
    if nspec==5:
        if verbose: print('Assuming EB and TB are zero')
        ceb=0
        ctb=0
    else:
        ceb=spectra[5][maskl]
        ctb=spectra[6][maskl]

    # Effective spectra
    norm=(2*ell+1)/(4*np.pi)*(pixwin**2)*(bl[maskl]**2)
    norm[1:]=norm[1:]
    effctt=norm*ctt
    effcte=norm*cte
    effcee=norm*cee
    effcbb=norm*cbb
    effceb=norm*ceb
    effctb=norm*ctb

    #### define masks for ell bins
    masks=[]
    for i in np.arange(nbins):
        masks.append((ell>=minell[i]) & (ell<=maxell[i]))

    ### Create array for covariances matrices per bin
    nbpixok=ipok.size
    nstokes=np.size(all)
    if verbose: print('creating array of pairs')
    cov=np.zeros((nbins,nstokes,nstokes,nbpixok,nbpixok))
    
    ### Create array of pixel pairs and pair angles and separate them in chunks for each processor
    pairs = []
    rpix_pairs = []
    for i in np.arange(nbpixok):
        for j in np.arange(i,nbpixok):
            pairs.append([i,j])
            rpix_pairs.append([rpix[:,i],rpix[:,j]])
    npairs = len(pairs)
    ## randomize them for equal charge of the processors
    #order = np.argsort(np.random.rand(npairs))
    #pairs = pairs[order]
    #rpix_pairs =rpix_pairs[order]
    ## divide into chunks
    npchunks = npairs // nprocs ## Integer division
    pairschunks = []
    rpixpairschunks = []
    for i in xrange(nprocs-1):
        pairschunks.append(pairs[i*npchunks:(i+1)*npchunks])
        rpixpairschunks.append(rpix_pairs[i*npchunks:(i+1)*npchunks])
    pairschunks.append(pairs[(nprocs-1)*npchunks:])
    rpixpairschunks.append(rpix_pairs[(nprocs-1)*npchunks:])
    
    # initiate multithreading
    tasks=multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    # start consumers
    num_consumers = nprocs
    consumers = [ Consumer(tasks, results)
                  for i in xrange(num_consumers) ]
    for w in consumers:
        w.start()
    for i in np.arange(num_consumers):
        thepairs = pairschunks[i]
        therpix_pairs = rpixpairschunks[i]
        print('I am consumer #{} - Calculating Covariance for {} pairs'.format(i,len(thepairs)))
        tasks.put(TaskPairs(i, cov, thepairs, therpix_pairs, lmax, nbins, nstokes, masks, effctt, effcee, 
                    effcbb, effcte, effceb, effctb))
    # poison them when finished
    for i in np.arange(num_consumers):
        tasks.put(None)
    tasks.join()
    while num_consumers:
        result = results.get()
        cov = cov + result[0]
        num=result[1]      
        num_consumers -= 1
        
    if allinone:
        newcov=np.zeros((nbins,nstokes*nbpixok,nstokes*nbpixok))
        for i in np.arange(nbins):
            for si in np.arange(nstokes):
                for sj in np.arange(si, nstokes):
                    newcov[i,si*nbpixok:(si+1)*nbpixok,sj*nbpixok:(sj+1)*nbpixok]=cov[i,si,sj,:,:]
                    newcov[i,sj*nbpixok:(sj+1)*nbpixok,si*nbpixok:(si+1)*nbpixok]=cov[i,sj,si,:,:]
        return(newcov)
    else:
        return(cov)


class TaskPairs(object):
    def __init__(self, i, cov, pairs, rpix_pairs, lmax, nbins, nstokes, masks, effctt, effcee, 
                    effcbb, effcte, effceb, effctb):
        self.i = i
        self.cov = cov
        self.pairs = pairs
        self.rpix_pairs = rpix_pairs
        self.lmax = lmax
        self.nbins = nbins
        self.nstokes = nstokes
        self.masks = masks
        self.effctt = effctt
        self.effcee = effcee
        self.effcbb = effcbb
        self.effcte = effcte
        self.effceb = effceb
        self.effctb = effctb
        self.npairs = len(pairs)
    def __call__(self):
        for ij in xrange(self.npairs):
            if ij % 10: pyquad.progress_bar(ij,self.npairs)
            i = self.pairs[ij][0]
            rpix_i = self.rpix_pairs[ij][0]
            j = self.pairs[ij][1]
            rpix_j = self.rpix_pairs[ij][1]
            res = covth_bins_ij(rpix_i, rpix_j, self.lmax, self.nbins, self.nstokes, 
                                                self.masks, self.effctt, self.effcee, self.effcbb, 
                                                self.effcte, self.effceb, self.effctb)
            self.cov[:,:,:,i,j] = res[0]
            self.cov[:,:,:,j,i] = res[1]
        return([self.cov,self.i])
        
################ New try        
def compute_ds_dcb_parpix_direct(ellbins,nside,ipok,bl,polar=True,temp=True,EBTB=False, verbose=False, nprocs=6):
    if verbose: print('dS/dCb Calulation:')
    if verbose: print('Temp='+str(temp))
    if verbose: print('Polar='+str(polar))
    if verbose: print('EBTB='+str(EBTB))
    # order of the derivatives ell, TT, EE, BB, TE, EB, TB
    all=['I','Q','U']
    # if EBTB:
    #     der=['TT','EE','BB','TE','EB','TB']
    #     ind=[1,2,3,4,5,6]
    # else:
    der=['TT','EE','BB','TE']
    ind=[1,2,3,4]
        
    if  not temp:
        all=['Q','U']
        # if EBTB:
        #     der=['EE','BB','EB']
        #     ind=[2,3,5]
        # else:
        der=['EE','BB']
        ind=[2,3]
    if not polar:
        all=['I']
        der=['TT']
        ind=[1]
    if verbose: print('Stokes parameters :',all)
    if verbose: print('Derivatives w.r.t. :',der)
    nder=len(der)
    nbpixok=ipok.size
    nstokes=np.size(all)
    
    #### define Pixel window function
    nbins=len(ellbins)-1
    minell=np.array(ellbins[0:nbins])
    maxell=np.array(ellbins[1:nbins+1])-1
    maxell[nbins-1]=np.max(ellbins)
    ellval=(minell+maxell)*0.5
    lmax=np.max(ellbins)
    ell=np.arange(lmax)+1
    pixwin=hp.pixwin(nside)[1:lmax+1]    
    norm=(2*ell+1)/(4*np.pi)*(pixwin**2)*(bl[1:lmax+1]**2)
    
    #### define masks for ell bins
    masks=[]
    for i in np.arange(nbins):
        masks.append((ell>=minell[i]) & (ell<=maxell[i]))

    #### define pixels
    rpix=np.array(hp.pix2vec(nside,ipok))

    ### Create the large matrix
    nstokes=len(all)
    nbins=len(ellbins)-1
    npix=len(ipok)
    ds_dc=np.zeros((nder,nbins,nstokes,nstokes,npix,npix))                

    ### Create array of pixel pairs and pair angles and separate them in chunks for each processor
    pairs = []
    rpix_pairs = []
    for i in np.arange(nbpixok):
        for j in np.arange(i,nbpixok):
            pairs.append([i,j])
            rpix_pairs.append([rpix[:,i],rpix[:,j]])
    npairs = len(pairs)
    ## divide into chunks
    npchunks = npairs // nprocs ## Integer division
    pairschunks = []
    rpixpairschunks = []
    for i in xrange(nprocs-1):
        pairschunks.append(pairs[i*npchunks:(i+1)*npchunks])
        rpixpairschunks.append(rpix_pairs[i*npchunks:(i+1)*npchunks])
    pairschunks.append(pairs[(nprocs-1)*npchunks:])
    rpixpairschunks.append(rpix_pairs[(nprocs-1)*npchunks:])
    
    # initiate multithreading
    tasks=multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    # start consumers
    num_consumers = nprocs
    consumers = [ Consumer(tasks, results)
                  for i in xrange(num_consumers) ]
    for w in consumers:
        w.start()
    for i in np.arange(num_consumers):
        thepairs = pairschunks[i]
        therpix_pairs = rpixpairschunks[i]
        print('I am consumer #{} - Calculating Covariance for {} pairs'.format(i,len(thepairs)))
        tasks.put(TaskdSdCPairs(i, ds_dc, thepairs, therpix_pairs, lmax, nbins, nstokes, nder, masks, norm))
    # poison them when finished
    for i in np.arange(num_consumers):
        tasks.put(None)
    tasks.join()
    while num_consumers:
        result = results.get()
        ds_dc = ds_dc + result[0]
        num=result[1]      
        num_consumers -= 1
        
        
    newds_dc=np.zeros((nder, nbins,nstokes*nbpixok,nstokes*nbpixok))
    for i in np.arange(nbins):
        for si in np.arange(nstokes):
            for sj in np.arange(si, nstokes):
                newds_dc[:,i,si*nbpixok:(si+1)*nbpixok,sj*nbpixok:(sj+1)*nbpixok]=ds_dc[:,i,si,sj,:,:]
                newds_dc[:,i,sj*nbpixok:(sj+1)*nbpixok,si*nbpixok:(si+1)*nbpixok]=ds_dc[:,i,sj,si,:,:]
    
    return newds_dc


class TaskdSdCPairs(object):
    def __init__(self, i, ds_dc, pairs, rpix_pairs, lmax, nbins, nstokes, nder, masks, norm):
        self.i = i
        self.ds_dc = ds_dc
        self.pairs = pairs
        self.rpix_pairs = rpix_pairs
        self.lmax = lmax
        self.nbins = nbins
        self.nstokes = nstokes
        self.masks = masks
        self.npairs = len(pairs)
        self.norm = norm
        self.nder = nder
    def __call__(self):
        for ij in xrange(self.npairs):
            if ij % 10: pyquad.progress_bar(ij,self.npairs)
            i = self.pairs[ij][0]
            rpix_i = self.rpix_pairs[ij][0]
            j = self.pairs[ij][1]
            rpix_j = self.rpix_pairs[ij][1]
            res = ds_dcb_bins_ij(rpix_i, rpix_j, self.lmax, self.nbins, self.nstokes, self.nder, self.masks, self.norm)
            self.ds_dc[:,:,:,:,i,j] = res[0]
            self.ds_dc[:,:,:,:,j,i] = res[1]
        return([self.ds_dc,self.i])
    
def ds_dcb_bins_ij(rpix_i, rpix_j, lmax, nbins, nstokes, nder, masks, norm):
    ds_dc=np.zeros((nder, nbins, nstokes, nstokes))
    ds_dc_ji=np.zeros((nder, nbins, nstokes, nstokes))
    cosangij = np.dot(np.transpose(rpix_i),rpix_j)
    if nstokes==1:
        pl=pl0(cosangij,lmax)[1:]
        thec = pl * norm
        for b in np.arange(nbins):
            ds_dc[0,b,0,0]=np.sum(thec[masks[b]])
            ds_dc_ji[0,b,0,0] = ds_dc[0,b,0,0]
    if nstokes==2:
        cij,sij=polrotangle(rpix_i,rpix_j)
        cji,sji=polrotangle(rpix_j,rpix_i)        
        f12=F1l2(cosangij,lmax)[1:]*norm
        f22=F2l2(cosangij,lmax)[1:]*norm
        #### der w.r.t EE and BB
        for b in np.arange(nbins):
            theQQ = [f12[masks[b]], -f22[masks[b]]]
            theUU = [-f22[masks[b]], f12[masks[b]]]
            theQU = [0,0]
            for ider in np.arange(nder):
                cQQpsQU = ( cij*theQQ[ider] + sij*theQU[ider] )
                cQUpsUU = ( cij*theQU[ider] + sij*theUU[ider] )
                cQUmsQQ = ( cij*theQU[ider] - sij*theQQ[ider] )
                cUUmsQU = ( cij*theUU[ider] - sij*theQU[ider] )
                ds_dc[ider, b,0,0]   = np.sum( cji*cQQpsQU + sji*cQUpsUU )
                ds_dc_ji[ider, b,0,0] = ds_dc[ider, b,0,0]
                ds_dc[ider, b,0,1]   = np.sum(-sji*cQQpsQU + cji*cQUpsUU )
                ds_dc_ji[ider, b,1,0] = ds_dc[ider, b,0,1]
                ds_dc[ider, b,1,1]   = np.sum(-sji*cQUmsQQ + cji*cUUmsQU )
                ds_dc_ji[ider, b,1,1] = ds_dc[ider, b,1,1]
                ds_dc[ider, b,1,0]   = np.sum( cji*cQUmsQQ + sji*cUUmsQU )
                ds_dc_ji[ider, b,0,1] = ds_dc[ider, b,1,0]
    return(ds_dc, ds_dc_ji)


# def ds_dcb_bins_ij(rpix_i, rpix_j, lmax, nbins, nstokes, nder, masks, norm):
#     ds_dc=np.zeros((nder, nbins, nstokes, nstokes))
#     ds_dc_ji=np.zeros((nder, nbins, nstokes, nstokes))
#     cosangij = np.dot(np.transpose(rpix_i),rpix_j)
#     if nstokes==1:
#         pl=pl0(cosangij,lmax)[1:]
#         thec = pl * norm
#         for b in np.arange(nbins):
#             ds_dc[0,b,0,0]=np.sum(thec[masks[b]])
#             ds_dc_ji[0,b,0,0] = ds_dc[0,b,0,0]
#     if nstokes==2:
#         cij,sij=polrotangle(rpix_i,rpix_j)
#         cji,sji=polrotangle(rpix_j,rpix_i)
#         f12=F1l2(cosangij,lmax)[1:]*norm
#         f22=F2l2(cosangij,lmax)[1:]*norm
#         #### der w.r.t EE and BB
#         theQQ = [f12, -f22]
#         theUU = [-f22, f12]
#         theQU = [0,0]
#         for ider in np.arange(nder):
#             cQQpsQU = ( cij*theQQ[ider] + sij*theQU[ider] )
#             cQUpsUU = ( cij*theQU[ider] + sij*theUU[ider] )
#             cQUmsQQ = ( cij*theQU[ider] - sij*theQQ[ider] )
#             cUUmsQU = ( cij*theUU[ider] - sij*theQU[ider] )
#             for b in np.arange(nbins):
#                 ds_dc[ider, b,0,0]   = np.sum( cji*cQQpsQU[masks[b]] + sji*cQUpsUU[masks[b]] )
#                 ds_dc_ji[ider, b,0,0] = ds_dc[ider, b,0,0]
#                 ds_dc[ider, b,0,1]   = np.sum(-sji*cQQpsQU[masks[b]] + cji*cQUpsUU[masks[b]] )
#                 ds_dc_ji[ider, b,1,0] = ds_dc[ider, b,0,1]
#                 ds_dc[ider, b,1,1]   = np.sum(-sji*cQUmsQQ[masks[b]] + cji*cUUmsQU[masks[b]] )
#                 ds_dc_ji[ider, b,1,1] = ds_dc[ider, b,1,1]
#                 ds_dc[ider, b,1,0]   = np.sum( cji*cQUmsQQ[masks[b]] + sji*cUUmsQU[masks[b]] )
#                 ds_dc_ji[ider, b,0,1] = ds_dc[ider, b,1,0]
#     return(ds_dc, ds_dc_ji)



################
        
        

def covth_bins_ij(rpix_i, rpix_j, lmax, nbins, nstokes, masks, effctt, effcee, effcbb, effcte, effceb, effctb):
    cov=np.zeros((nbins,nstokes,nstokes))
    covji=np.zeros((nbins,nstokes,nstokes))
    cosangij = np.dot(np.transpose(rpix_i),rpix_j)
    if nstokes==1:
        pl=pl0(cosangij,lmax)
        TT=effctt*pl
        for b in np.arange(nbins):
            cov[b,0,0]=np.sum(TT[masks[b]])
            covji[b,0,0] = cov[b,0,0]
    if nstokes==2:
        cij,sij=polrotangle(rpix_i,rpix_j)
        cji,sji=polrotangle(rpix_j,rpix_i)
        f12=F1l2(cosangij,lmax)
        f22=F2l2(cosangij,lmax)
        QQ=f12*effcee-f22*effcbb
        UU=f12*effcbb-f22*effcee
        QU=(f12+f22)*effceb
        cQQpsQU = ( cij*QQ + sij*QU )
        cQUpsUU = ( cij*QU + sij*UU )
        cQUmsQQ = ( cij*QU - sij*QQ )
        cUUmsQU = ( cij*UU - sij*QU )
        for b in np.arange(nbins):
            cov[b,0,0]   = np.sum( cji*cQQpsQU[masks[b]] + sji*cQUpsUU[masks[b]] )
            covji[b,0,0] = cov[b,0,0]
            cov[b,0,1]   = np.sum(-sji*cQQpsQU[masks[b]] + cji*cQUpsUU[masks[b]] )
            covji[b,1,0] = cov[b,0,1]
            cov[b,1,1]   = np.sum(-sji*cQUmsQQ[masks[b]] + cji*cUUmsQU[masks[b]] )
            covji[b,1,1] = cov[b,1,1]
            cov[b,1,0]   = np.sum( cji*cQUmsQQ[masks[b]] + sji*cUUmsQU[masks[b]] )
            covji[b,0,1] = cov[b,1,0]
    if nstokes==3:
        cij,sij=polrotangle(rpix_i,rpix_j)
        cji,sji=polrotangle(rpix_j,rpix_i)
        pl=pl0(cosangij,lmax)
        f10=F1l0(cosangij,lmax)
        f12=F1l2(cosangij,lmax)
        f22=F2l2(cosangij,lmax)
        TT=effctt*pl
        QQ=f12*effcee-f22*effcbb
        UU=f12*effcbb-f22*effcee
        TQ=-f10*effcte
        TU=-f10*effctb
        QU=(f12+f22)*effceb
        cQQpsQU = ( cij*QQ + sij*QU )
        cQUpsUU = ( cij*QU + sij*UU )
        cQUmsQQ = ( cij*QU - sij*QQ )
        cUUmsQU = ( cij*UU - sij*QU )
        for b in np.arange(nbins):
            cov[b,0,0] = np.sum( TT[masks[b]] )
            cov[b,0,1] = np.sum( cji*TQ[masks[b]] + sji*TU[masks[b]] )
            cov[b,1,0] = np.sum( cij*TQ[masks[b]] + sij*TU[masks[b]] )
            cov[b,0,2] = np.sum(-sji*TQ[masks[b]] + cij*TU[masks[b]] )
            cov[b,2,0] = np.sum(-sij*TQ[masks[b]] + cij*TU[masks[b]] )
            cov[b,1,1] = np.sum( cji*cQQpsQU[masks[b]] + sji*cQUpsUU[masks[b]] )
            cov[b,1,2] = np.sum(-sji*cQQpsQU[masks[b]] + cji*cQUpsUU[masks[b]] )
            cov[b,2,1] = np.sum( cji*cQUmsQQ[masks[b]] + sji*cUUmsQU[masks[b]] )
            cov[b,2,2] = np.sum(-sji*cQUmsQQ[masks[b]] + cji*cUUmsQU[masks[b]] )
            covji[b,0,0] = cov[b,0,0]
            covji[b,0,1] = cov[b,1,0]
            covji[b,1,0] = cov[b,0,1]
            covji[b,0,2] = cov[b,2,0]
            covji[b,2,0] = cov[b,0,2]
            covji[b,1,1] = cov[b,1,1]
            covji[b,1,2] = cov[b,2,1]
            covji[b,2,1] = cov[b,1,2]
            covji[b,2,2] = cov[b,2,2]
            
    return cov,covji
    
    




def covth_bins(ellbins,nside,ipok,allcosang,bl,spectra,polar=True,temp=True,allinone=True, verbose=False):
    #### define bins in ell
    nbins=len(ellbins)-1
    minell=np.array(ellbins[0:nbins])
    maxell=np.array(ellbins[1:nbins+1])-1
    maxell[nbins-1]=np.max(ellbins)
    ellval=(minell+maxell)*0.5
    lmax=np.max(ellbins)
    #maxell[nbins-1]=lmax
    if verbose: print('minell:',minell)
    if verbose: print('maxell:',maxell)
    #### define Stokes
    all=['I','Q','U']
    if  not temp:
        all=['Q','U']
    if not polar:
        all=['I']

    #### define pixels
    rpix=np.array(hp.pix2vec(nside,ipok))
    #allcosang=np.dot(np.transpose(rpix),rpix)
    
    #### define Pixel window function
    pixwin=hp.pixwin(nside)[0:lmax+1]

    #### get ell values from spectra and restrict spectra to good values
    nspec,nell=np.shape(spectra)
    ell=spectra[0]
    maskl=ell<(lmax+1)
    ell=ell[maskl]
    ctt=spectra[1][maskl]
    cee=spectra[2][maskl]
    cbb=spectra[3][maskl]
    cte=spectra[4][maskl]
    if nspec==5:
        if verbose: print('Assuming EB and TB are zero')
        ceb=0
        ctb=0
    else:
        ceb=spectra[5][maskl]
        ctb=spectra[6][maskl]

    # Effective spectra
    norm=(2*ell+1)/(4*np.pi)*(pixwin**2)*(bl[maskl]**2)
    norm[1:]=norm[1:]
    effctt=norm*ctt
    effcte=norm*cte
    effcee=norm*cee
    effcbb=norm*cbb
    effceb=norm*ceb
    effctb=norm*ctb

    #### define masks for ell bins
    masks=[]
    for i in np.arange(nbins):
        masks.append((ell>=minell[i]) & (ell<=maxell[i]))

    ### Create array for covariances matrices per bin
    nbpixok=ipok.size
    nstokes=np.size(all)
    print('creating array')
    cov=np.zeros((nbins,nstokes,nstokes,nbpixok,nbpixok))
    for i in np.arange(nbpixok):
        pyquad.progress_bar(i,nbpixok)
        for j in np.arange(i,nbpixok):
            if nstokes==1:
                pl=pl0(allcosang[i,j],lmax)
                TT=effctt*pl
                for b in np.arange(nbins):
                    cov[b,0,0,i,j]=np.sum(TT[masks[b]])
                    cov[b,0,0,j,i]=cov[b,0,0,i,j]
            if nstokes==2:
                cij,sij=polrotangle(rpix[:,i],rpix[:,j])
                cji,sji=polrotangle(rpix[:,j],rpix[:,i])
                f12=F1l2(allcosang[i,j],lmax)
                f22=F2l2(allcosang[i,j],lmax)
                QQ=f12*effcee-f22*effcbb
                UU=f12*effcbb-f22*effcee
                QU=(f12+f22)*effceb
                cQQpsQU = ( cij*QQ + sij*QU )
                cQUpsUU = ( cij*QU + sij*UU )
                cQUmsQQ = ( cij*QU - sij*QQ )
                cUUmsQU = ( cij*UU - sij*QU )
                for b in np.arange(nbins):
                    cov[b,0,0,i,j] = np.sum( cji*cQQpsQU[masks[b]] + sji*cQUpsUU[masks[b]] )
                    cov[b,0,0,j,i] = cov[b,0,0,i,j]
                    cov[b,0,1,i,j] = np.sum(-sji*cQQpsQU[masks[b]] + cji*cQUpsUU[masks[b]] )
                    cov[b,1,0,j,i] = cov[b,0,1,i,j]
                    cov[b,1,1,i,j] = np.sum(-sji*cQUmsQQ[masks[b]] + cji*cUUmsQU[masks[b]] )
                    cov[b,1,1,j,i] = cov[b,1,1,i,j]
                    cov[b,0,1,j,i] = np.sum( cji*cQUmsQQ[masks[b]] + sji*cUUmsQU[masks[b]] )
                    cov[b,1,0,i,j] = cov[b,0,1,j,i]
            if nstokes==3:
                cij,sij=polrotangle(rpix[:,i],rpix[:,j])
                cji,sji=polrotangle(rpix[:,j],rpix[:,i])
                pl=pl0(allcosang[i,j],lmax)
                f10=F1l0(allcosang[i,j],lmax)
                f12=F1l2(allcosang[i,j],lmax)
                f22=F2l2(allcosang[i,j],lmax)
                TT=effctt*pl
                QQ=f12*effcee-f22*effcbb
                UU=f12*effcbb-f22*effcee
                TQ=-f10*effcte
                TU=-f10*effctb
                QU=(f12+f22)*effceb
                cQQpsQU = ( cij*QQ + sij*QU )
                cQUpsUU = ( cij*QU + sij*UU )
                cQUmsQQ = ( cij*QU - sij*QQ )
                cUUmsQU = ( cij*UU - sij*QU )
                for b in np.arange(nbins):
                    cov[b,0,0,i,j] = np.sum( TT[masks[b]] )
                    cov[b,0,0,j,i] = cov[b,0,0,i,j]                
                    cov[b,0,1,i,j] = np.sum( cji*TQ[masks[b]] + sji*TU[masks[b]] )
                    cov[b,1,0,j,i] = cov[b,0,1,i,j]
                    cov[b,1,0,i,j] = np.sum( cij*TQ[masks[b]] + sij*TU[masks[b]] )
                    cov[b,0,1,j,i] = cov[b,1,0,i,j]
                    cov[b,0,2,i,j] = np.sum(-sji*TQ[masks[b]] + cij*TU[masks[b]] )
                    cov[b,2,0,j,i] = cov[b,0,2,i,j]
                    cov[b,2,0,i,j] = np.sum(-sij*TQ[masks[b]] + cij*TU[masks[b]] )
                    cov[b,0,2,j,i] = cov[b,2,0,i,j]
                    cov[b,1,1,i,j] = np.sum( cji*cQQpsQU[masks[b]] + sji*cQUpsUU[masks[b]] )
                    cov[b,1,1,j,i] = cov[b,1,1,i,j]
                    cov[b,1,2,i,j] = np.sum(-sji*cQQpsQU[masks[b]] + cji*cQUpsUU[masks[b]] )
                    cov[b,2,1,j,i] = cov[b,1,2,i,j]
                    cov[b,2,1,i,j] = np.sum( cji*cQUmsQQ[masks[b]] + sji*cUUmsQU[masks[b]] )
                    cov[b,1,2,j,i] = cov[b,2,1,i,j]
                    cov[b,2,2,i,j] = np.sum(-sji*cQUmsQQ[masks[b]] + cji*cUUmsQU[masks[b]] )
                    cov[b,2,2,j,i] = cov[b,2,2,i,j]
    if allinone:
        newcov=np.zeros((nbins,nstokes*nbpixok,nstokes*nbpixok))
        for i in np.arange(nbins):
            for si in np.arange(nstokes):
                for sj in np.arange(nstokes):
                    newcov[i,si*nbpixok:(si+1)*nbpixok,sj*nbpixok:(sj+1)*nbpixok]=cov[i,si,sj,:,:]
        return(newcov)
    else:
        return(cov)
    

###################### Covariance Matrices for I, Q, U #########################################
def covth(nside,ipok,lmax,bl,spectra,polar=True,temp=True,allinone=True):
    all=['I','Q','U']
    if  not temp:
        all=['Q','U']
    if not polar:
        all=['I']
    rpix=np.array(hp.pix2vec(nside,ipok))
    allcosang=np.dot(np.transpose(rpix),rpix)
    pixwin=hp.pixwin(nside)[0:lmax+1]
    nspec,nell=np.shape(spectra)
    ell=spectra[0]
    maskl=ell<(lmax+1)
    ell=ell[maskl]
    ctt=spectra[1][maskl]
    cee=spectra[2][maskl]
    cbb=spectra[3][maskl]
    cte=spectra[4][maskl]
    if nspec==5:
        print('Assuming EB and TB are zero')
        ceb=0
        ctb=0
    else:
        ceb=spectra[5][maskl]
        ctb=spectra[6][maskl]
    norm=(2*ell+1)/(4*np.pi)*(pixwin**2)*(bl[maskl]**2)
    effctt=norm*ctt
    effcte=norm*cte
    effcee=norm*cee
    effcbb=norm*cbb
    effceb=norm*ceb
    effctb=norm*ctb
    nbpixok=ipok.size
    nstokes=np.size(all)
    print(nstokes)
    cov=np.zeros((nstokes,nstokes,nbpixok,nbpixok))
    for i in np.arange(nbpixok):
        pyquad.progress_bar(i,nbpixok)
        for j in np.arange(i,nbpixok):
            if nstokes==1:
                pl=pl0(allcosang[i,j],lmax)
                TT=effctt*pl
                cov[0,0,i,j]=np.sum(TT)
                cov[0,0,j,i]=cov[0,0,i,j]
            if nstokes==2:
                cij,sij=polrotangle(rpix[:,i],rpix[:,j])
                cji,sji=polrotangle(rpix[:,j],rpix[:,i])
                f12=F1l2(allcosang[i,j],lmax)
                f22=F2l2(allcosang[i,j],lmax)
                QQ=f12*effcee-f22*effcbb
                UU=f12*effcbb-f22*effcee
                QU=(f12+f22)*effceb
                cQQpsQU = ( cij*QQ + sij*QU )
                cQUpsUU = ( cij*QU + sij*UU )
                cQUmsQQ = ( cij*QU - sij*QQ )
                cUUmsQU = ( cij*UU - sij*QU )
                cov[0,0,i,j] = np.sum( cji*cQQpsQU + sji*cQUpsUU )
                cov[0,0,j,i] = cov[0,0,i,j]
                cov[0,1,i,j] = np.sum(-sji*cQQpsQU + cji*cQUpsUU )
                cov[1,0,j,i] = cov[0,1,i,j]
                cov[1,1,i,j] = np.sum(-sji*cQUmsQQ + cji*cUUmsQU )
                cov[1,1,j,i] = cov[1,1,i,j]
                cov[0,1,j,i] = np.sum( cji*cQUmsQQ + sji*cUUmsQU )
                cov[1,0,i,j] = cov[0,1,j,i]
            if nstokes==3:
                cij,sij=polrotangle(rpix[:,i],rpix[:,j])
                cji,sji=polrotangle(rpix[:,j],rpix[:,i])
                pl=pl0(allcosang[i,j],lmax)
                f10=F1l0(allcosang[i,j],lmax)
                f12=F1l2(allcosang[i,j],lmax)
                f22=F2l2(allcosang[i,j],lmax)
                TT=effctt*pl
                QQ=f12*effcee-f22*effcbb
                UU=f12*effcbb-f22*effcee
                TQ=-f10*effcte
                TU=-f10*effctb
                QU=(f12+f22)*effceb
                cQQpsQU = ( cij*QQ + sij*QU )
                cQUpsUU = ( cij*QU + sij*UU )
                cQUmsQQ = ( cij*QU - sij*QQ )
                cUUmsQU = ( cij*UU - sij*QU )
                cov[0,0,i,j] = np.sum( TT )
                cov[0,0,j,i] = cov[0,0,i,j]                
                cov[0,1,i,j] = np.sum( cji*TQ + sji*TU )
                cov[1,0,j,i] = cov[0,1,i,j]
                cov[1,0,i,j] = np.sum( cij*TQ + sij*TU )
                cov[0,1,j,i] = cov[1,0,i,j]
                cov[0,2,i,j] = np.sum(-sji*TQ + cij*TU )
                cov[2,0,j,i] = cov[0,2,i,j]
                cov[2,0,i,j] = np.sum(-sij*TQ + cij*TU )
                cov[0,2,j,i] = cov[2,0,i,j]
                cov[1,1,i,j] = np.sum( cji*cQQpsQU + sji*cQUpsUU )
                cov[1,1,j,i] = cov[1,1,i,j]
                cov[1,2,i,j] = np.sum(-sji*cQQpsQU + cji*cQUpsUU )
                cov[2,1,j,i] = cov[1,2,i,j]
                cov[2,1,i,j] = np.sum( cji*cQUmsQQ + sji*cUUmsQU )
                cov[1,2,j,i] = cov[2,1,i,j]
                cov[2,2,i,j] = np.sum(-sji*cQUmsQQ + cji*cUUmsQU )
                cov[2,2,j,i] = cov[2,2,i,j]
    if allinone==True:
        return(allmat2bigmat(cov))
    else:
        return cov

def allmat2bigmat(allmat):
    sh=np.shape(allmat)
    nx=sh[0]
    ny=sh[1]
    npixx=sh[2]
    npixy=sh[3]
    newmat=np.zeros((nx*npixx,ny*npixy))
    for i in np.arange(nx):
        for j in np.arange(ny):
            newmat[i*npixx:(i+1)*npixx,j*npixy:(j+1)*npixy]=allmat[i,j,:,:]         
    return newmat





def compute_ds_dcb_parpix_new(ellbins,nside,ipok,bl,polar=True,temp=True,EBTB=False, verbose=False, nprocs=6):
    if verbose: print('dS/dCb Calulation:')
    if verbose: print('Temp='+str(temp))
    if verbose: print('Polar='+str(polar))
    if verbose: print('EBTB='+str(EBTB))
    # order of the derivatives ell, TT, EE, BB, TE, EB, TB
    all=['I','Q','U']
    if EBTB:
        der=['TT','EE','BB','TE','EB','TB']
        ind=[1,2,3,4,5,6]
    else:
        der=['TT','EE','BB','TE']
        ind=[1,2,3,4]
        
    if  not temp:
        all=['Q','U']
        if EBTB:
            der=['EE','BB','EB']
            ind=[2,3,5]
        else:
            der=['EE','BB']
            ind=[2,3]
    if not polar:
        all=['I']
        der=['TT']
        ind=[1]
    if verbose: print('Stokes parameters :',all)
    if verbose: print('Derivatives w.r.t. :',der)
    ### prepare spectra with zeros and ones for calculating the derivatives
    ell=np.arange(np.min(ellbins),np.max(ellbins)+1)
    speczero=np.zeros(len(ell))
    specone=np.ones(len(ell))
    nder=len(der)
    if EBTB:
        allspeczero=[ell,speczero,speczero,speczero,speczero,speczero,speczero]
    else:
        allspeczero=[ell,speczero,speczero,speczero,speczero]
    
    #### define pixels
    rpix=np.array(hp.pix2vec(nside,ipok))

    ### dimensions of the large matrix and loop for filling it
    nstokes=len(all)
    nbins=len(ellbins)-1
    npix=len(ipok)
    dcov=np.zeros((nder,nbins,nstokes*npix,nstokes*npix))        
    for i in np.arange(nder):
        thespec=list(allspeczero)
        thespec[ind[i]]=specone
        if verbose: print('Calculating dS/dCb for '+der[i])
        dcov[i,:,:,:]=covth_bins_par(ellbins,nside,ipok,bl,thespec,polar=polar,temp=temp, allinone = True, verbose=True, nprocs=nprocs)
        
    return dcov












######## Quadratic estimator
def qml(maps,mask,covmap,ellbins,fwhmrad,guess,ds_dcb,spectra,itmax=20,plot=False,
    cholesky=True,polar=True,temp=True,EBTB=False,getmat=False, message=''):
    print('QML Estimator')
    ###### What will be calculated ?
    print('Temp='+str(temp))
    print('Polar='+str(polar))
    print('EBTB='+str(EBTB))
    all=['I','Q','U']
    if EBTB:
        xx=['TT','EE','BB','TE','EB','TB']
        ind=[1,2,3,4,5,6]
    else:
        xx=['TT','EE','BB','TE']
        ind=[1,2,3,4]
        
    if  not temp:
        all=['Q','U']
        if EBTB:
            xx=['EE','BB','EB']
            ind=[2,3,5]
        else:
            xx=['EE','BB']
            ind=[2,3]
    if not polar:
        all=['I']
        xx=['TT']
        ind=[1]
    print('Number of maps :',len(maps))
    print('Stokes parameters :',all)
    print('Spectra considered. :',xx)
    nspec=len(xx)

    if len(maps) != len(all):
        print('Inconstitent number of maps and Stokes parameters: Exiting !')
        return -1

    ### maps information
    nside=hp.npix2nside(len(maps[0]))
    ip=np.arange(12*nside**2)
    ipok=ip[~mask]
    npix=len(ipok)
    npixtot=len(all)*npix
    masktot=np.zeros(len(all)*len(mask),dtype=bool)
    map=np.zeros(len(all)*npix)
    for i in np.arange(len(all)):
        map[i*npix:(i+1)*npix]=maps[i][~mask]
        masktot[i*len(mask):(i+1)*len(mask)]=mask

    ### ell binning
    nbins=len(ellbins)-1
    ellmin=np.array(ellbins[0:nbins])
    ellmax=np.array(ellbins[1:nbins+1])-1
    ellval=(ellmin+ellmax)*1./2
    ll=np.arange(int(np.max(ellbins))+1)
    bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)

    ### Initial guess
    nbinstot=nspec*nbins
    specinit=np.zeros(nbinstot)
    for s in np.arange(nspec):
        specinit[s*nbins:(s+1)*nbins]=guess[ind[s]]
    thespectrum=np.zeros((nbinstot,itmax+1))
    thespectrum[:,0]=specinit

    ####################### P1 dS/dC Calculation #################################
    if ds_dcb is None:
        ds_dcb=compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=polar,temp=temp,EBTB=EBTB)
    ##############################################################################
    
    ### Start iterations
    num=0
    convergence=0
    lk=np.zeros(itmax)

    while convergence==0:
        ##########################################################################
        ## P2 pixpix covariance matrix for maps and solving z=M-1.m and likelihood
        print('    P2')
        matsky=np.zeros((npixtot,npixtot))
        for s in np.arange(nspec):
            for i in np.arange(nbins):
                matsky += thespectrum[s*nbins+i,num]*ds_dcb[s,i,:,:]
        print('       - Done matsky')
        if getmat:
            return matsky
        matcov=covmap+matsky
        
        if cholesky is True:
            print('      Doing Cholesky decomposition')
            U=scipy.linalg.cho_factor(matcov)
            z=scipy.linalg.cho_solve(U,map)
        else:
            print('      Brute force inversion (No Cholesky)')
            minv=scipy.linalg.inv(matcov)
            z=np.dot(minv,map)
                
        mapw=np.zeros(map.size)
        mapw=z
        lk[num]=-0.5*(np.dot(z,map)+np.sum(np.log(np.diag(matcov))))
        ##########################################################################

        ########################################################################
        # P3 Solve the equations Wb=M^(-1).ds_dcb ##############################
        print('    P3')
        wb=np.zeros((nbinstot,npixtot,npixtot))
        for s in np.arange(nspec):
            pyquad.progress_bar(s,nspec)
            for i in np.arange(nbins):
                if cholesky is True:
                    wb[s*nbins+i,:,:]=scipy.linalg.cho_solve(U,ds_dcb[s,i,:,:])
                else:
                    wb[s*nbins+i,:,:]=np.dot(minv,ds_dcb[s,i,:,:])
        ########################################################################

        ########################################################################
        # P4 First derivatives of the likelihood ##############################
        print('    P4')
        dldcb=np.zeros(nspec*nbins)
        for i in np.arange(nbinstot):
            pyquad.progress_bar(i,nbinstot)
            dldcb[i]=0.5*(np.dot(map,np.dot(wb[i,:,:],z))-np.trace(wb[i,:,:]))
        #######################################################################
         
        ##########################################################################
        # P5 second derivatives of the likelihood ################################
        print('    P5')
        d2ldcbdcb=np.zeros((nspec*nbins,nspec*nbins))
        fisher=np.zeros((nspec*nbins,nspec*nbins))
        kk=0
        for i in np.arange(nbinstot):
            for j in np.arange(i,nbinstot):
                pyquad.progress_bar(kk,(nbinstot+1)*nbinstot/2)
                wbwb=np.dot(wb[i,:,:],wb[j,:,:])
                fisher[i,j]=np.trace(wbwb)
                d2ldcbdcb[i,j]=-np.dot(map,np.dot(wbwb,z))+0.5*fisher[i,j]
                fisher[j,i]=fisher[i,j]
                d2ldcbdcb[j,i]=d2ldcbdcb[i,j]
                kk+=1
        ###########################################################################
        
        ###########################################################################
        # P6 Correction to be applied to the spectrum #############################
        print('    P6')
        invfisher=(np.linalg.inv(fisher))
        err=np.sqrt(np.diag(invfisher))
        deltac=-np.dot(np.linalg.inv(d2ldcbdcb),dldcb)
        print('    Result for iteration',num)
        print('      C : ',min(thespectrum[:,num]),max(thespectrum[:,num]))
        print('     dC : ',min(deltac),max(deltac))
        newspectrum=(thespectrum[:,num]+deltac)
        deltac[newspectrum < 0]=0
        thespectrum[:,num+1]=(thespectrum[:,num]+deltac)
        ############################################################################
    
        spec=thespectrum[:,num+1]
        finalspectrum=[ellval,0,0,0,0,0,0]
        error=[ellval,0,0,0,0,0,0]
        for s in np.arange(nspec):
            finalspectrum[ind[s]]=spec[s*nbins:(s+1)*nbins]
            error[ind[s]]=err[s*nbins:(s+1)*nbins]
            
        ############################################################################
        # Critere de convergence ###################################################
        conv=deltac/np.sqrt(np.abs(np.diag(invfisher)))
        print('     Conv : ',max(abs(conv)))
        print('     Likelihood : ',lk[num])
        nplot_tot=1+nspec
        sqrn=int(np.sqrt(nplot_tot))+1
        nr=sqrn
        nc=sqrn

        iplot=0
        if plot is True:
            mp.clf()
            for s in np.arange(nspec):
                mp.subplot(nr,nc,iplot+1)
                mp.xlim(0,np.max(ellmax)*1.2)
                ell=spectra[0]
                cl=spectra[ind[s]]
                ellb=finalspectrum[0]
                mp.plot(ell,cl*(ell*(ell+1))/(2*np.pi),lw=3)
                msk = ell < np.max(ellmax)
                mp.ylim(0,np.max(cl[msk]*(ell[msk]*(ell[msk]+1))/(2*np.pi))*1.5)
                mp.plot(ellb,guess[ind[s]]*ellb*(ellb+1)/(2*np.pi),'go',alpha=0.5)
                mp.xlabel('$\ell$')
                mp.ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')
                mp.errorbar(ellb,finalspectrum[ind[s]]*ellb*(ellb+1)/(2*np.pi),
                    error[ind[s]]*ellb*(ellb+1)/(2*np.pi),xerr=(ellmax+1-ellmin)/2,
                    label=str(i),fmt='ro')
                mp.draw()
                mp.title(xx[s])
                iplot+=1
                
            mp.subplot(nr,nc,iplot+1)
            if num>0: mp.plot(lk[:num+1],'o-')
            mp.xlabel('Iteration')
            mp.ylabel('Likelihood')
            mp.title(message)
            mp.draw()
        if num==itmax-1:
            convergence=-1
        if np.max(np.abs(conv)) <= 0.01:
            convergence=1
        else:
            num=num+1
        ###########################################################################

    return finalspectrum,error,invfisher,lk,num,ds_dcb,convergence





######## Quadratic estimator
def qml_cross(maps,maps2,mask,covmap,ellbins,fwhmrad,guess,ds_dcb,spectra,itmax=20,plot=False,cholesky=True,polar=True,temp=True,EBTB=False,getmat=False):
    print('QML Estimator')
    ###### What will be calculated ?
    print('Temp='+str(temp))
    print('Polar='+str(polar))
    print('EBTB='+str(EBTB))
    all=['I','Q','U']
    if EBTB:
        xx=['TT','EE','BB','TE','EB','TB']
        ind=[1,2,3,4,5,6]
    else:
        xx=['TT','EE','BB','TE']
        ind=[1,2,3,4]
        
    if  not temp:
        all=['Q','U']
        if EBTB:
            xx=['EE','BB','EB']
            ind=[2,3,5]
        else:
            xx=['EE','BB']
            ind=[2,3]
    if not polar:
        all=['I']
        xx=['TT']
        ind=[1]
    print('Number of maps :',len(maps))
    print('Stokes parameters :',all)
    print('Spectra considered. :',xx)
    nspec=len(xx)

    if len(maps) != len(all):
        print('Inconstitent number of maps and Stokes parameters: Exiting !')
        return -1

    ### maps information
    nside=hp.npix2nside(len(maps[0]))
    ip=np.arange(12*nside**2)
    ipok=ip[~mask]
    npix=len(ipok)
    npixtot=len(all)*npix
    masktot=np.zeros(len(all)*len(mask),dtype=bool)
    map=np.zeros(len(all)*npix)
    for i in np.arange(len(all)):
        map[i*npix:(i+1)*npix]=maps[i][~mask]
        masktot[i*len(mask):(i+1)*len(mask)]=mask

    map2=np.zeros(len(all)*npix)
    for i in np.arange(len(all)):
        map2[i*npix:(i+1)*npix]=maps2[i][~mask]
        masktot[i*len(mask):(i+1)*len(mask)]=mask

    ### ell binning
    nbins=len(ellbins)-1
    ellmin=np.array(ellbins[0:nbins])
    ellmax=np.array(ellbins[1:nbins+1])-1
    ellval=(ellmin+ellmax)*1./2
    ll=np.arange(int(np.max(ellbins))+1)
    bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)

    ### Initial guess
    nbinstot=nspec*nbins
    specinit=np.zeros(nbinstot)
    for s in np.arange(nspec):
        specinit[s*nbins:(s+1)*nbins]=guess[ind[s]]
    thespectrum=np.zeros((nbinstot,itmax+1))
    thespectrum[:,0]=specinit

    ####################### P1 dS/dC Calculation #################################
    if ds_dcb is None:
        ds_dcb=compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=polar,temp=temp,EBTB=EBTB)
    ##############################################################################
    
    ### Start iterations
    num=0
    convergence=0
    lk=np.zeros(itmax)

    while convergence==0:
        ##########################################################################
        ## P2 pixpix covariance matrix for maps and solving z=M-1.m and likelihood
        print('    P2')
        matsky=np.zeros((npixtot,npixtot))
        for s in np.arange(nspec):
            for i in np.arange(nbins):
                matsky += thespectrum[s*nbins+i,num]*ds_dcb[s,i,:,:]
        print('       - Done matsky')
        if getmat:
            return matsky
        matcov=covmap+matsky
        
        if cholesky is True:
            print('      Doing Cholesky decomposition')
            U=scipy.linalg.cho_factor(matcov)
            z=scipy.linalg.cho_solve(U,map2)
        else:
            print('      Brute force inversion (No Cholesky)')
            minv=scipy.linalg.inv(matcov)
            z=np.dot(minv,map2)
                
        mapw=np.zeros(map.size)
        mapw=z
        lk[num]=-0.5*(np.dot(z,map)+np.sum(np.log(np.diag(matcov))))
        ##########################################################################

        ########################################################################
        # P3 Solve the equations Wb=M^(-1).ds_dcb ##############################
        print('    P3')
        wb=np.zeros((nbinstot,npixtot,npixtot))
        for s in np.arange(nspec):
            pyquad.progress_bar(s,nspec)
            for i in np.arange(nbins):
                if cholesky is True:
                    wb[s*nbins+i,:,:]=scipy.linalg.cho_solve(U,ds_dcb[s,i,:,:])
                else:
                    wb[s*nbins+i,:,:]=np.dot(minv,ds_dcb[s,i,:,:])
        ########################################################################

        ########################################################################
        # P4 First derivatives of the likelihood ##############################
        print('    P4')
        dldcb=np.zeros(nspec*nbins)
        for i in np.arange(nbinstot):
            pyquad.progress_bar(i,nbinstot)
            dldcb[i]=0.5*(np.dot(map,np.dot(wb[i,:,:],z))-np.trace(wb[i,:,:]))
        #######################################################################
         
        ##########################################################################
        # P5 second derivatives of the likelihood ################################
        print('    P5')
        d2ldcbdcb=np.zeros((nspec*nbins,nspec*nbins))
        fisher=np.zeros((nspec*nbins,nspec*nbins))
        kk=0
        for i in np.arange(nbinstot):
            for j in np.arange(i,nbinstot):
                pyquad.progress_bar(kk,(nbinstot+1)*nbinstot/2)
                wbwb=np.dot(wb[i,:,:],wb[j,:,:])
                fisher[i,j]=np.trace(wbwb)
                d2ldcbdcb[i,j]=-np.dot(map,np.dot(wbwb,z))+0.5*fisher[i,j]
                fisher[j,i]=fisher[i,j]
                d2ldcbdcb[j,i]=d2ldcbdcb[i,j]
                kk+=1
        ###########################################################################
        
        ###########################################################################
        # P6 Correction to be applied to the spectrum #############################
        print('    P6')
        invfisher=(np.linalg.inv(fisher))
        err=np.sqrt(np.diag(invfisher))
        deltac=-np.dot(np.linalg.inv(d2ldcbdcb),dldcb)
        print('    Result for iteration',num)
        print('      C : ',min(thespectrum[:,num]),max(thespectrum[:,num]))
        print('     dC : ',min(deltac),max(deltac))
        newspectrum=(thespectrum[:,num]+deltac)
        deltac[newspectrum < 0]=0
        thespectrum[:,num+1]=(thespectrum[:,num]+deltac)
        ############################################################################
    
        spec=thespectrum[:,num+1]
        finalspectrum=[ellval,0,0,0,0,0,0]
        error=[ellval,0,0,0,0,0,0]
        for s in np.arange(nspec):
            finalspectrum[ind[s]]=spec[s*nbins:(s+1)*nbins]
            error[ind[s]]=err[s*nbins:(s+1)*nbins]
            
        ############################################################################
        # Critere de convergence ###################################################
        conv=deltac/np.sqrt(np.abs(np.diag(invfisher)))
        print('     Conv : ',max(abs(conv)))
        print('     Likelihood : ',lk[num])
        nplot_tot=1+nspec
        sqrn=int(np.sqrt(nplot_tot))+1
        nr=sqrn
        nc=sqrn

        iplot=0
        if plot is True:
            mp.clf()
            for s in np.arange(nspec):
                mp.subplot(nr,nc,iplot+1)
                mp.xlim(0,np.max(ellmax)*1.2)
                ell=spectra[0]
                cl=spectra[ind[s]]
                ellb=finalspectrum[0]
                mp.plot(ell,cl*(ell*(ell+1))/(2*np.pi),lw=3)
                msk = ell < np.max(ellmax)
                mp.ylim(0,np.max(cl[msk]*(ell[msk]*(ell[msk]+1))/(2*np.pi))*1.5)
                mp.plot(ellb,guess[ind[s]]*ellb*(ellb+1)/(2*np.pi),'go',alpha=0.5)
                mp.xlabel('$\ell$')
                mp.ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')
                mp.errorbar(ellb,finalspectrum[ind[s]]*ellb*(ellb+1)/(2*np.pi),
                    error[ind[s]]*ellb*(ellb+1)/(2*np.pi),xerr=(ellmax+1-ellmin)/2,
                    label=str(i),fmt='ro')
                mp.draw()
                mp.title(xx[s])
                iplot+=1
                
            mp.subplot(nr,nc,iplot+1)
            if num>0: mp.plot(lk[:num+1],'o-')
            mp.xlabel('Iteration')
            mp.ylabel('Likelihood')
            mp.draw()
        if num==itmax-1:
            convergence=-1
        if np.max(np.abs(conv)) <= 0.01:
            convergence=1
        else:
            num=num+1
        ###########################################################################

    return finalspectrum,error,invfisher,lk,num,ds_dcb,convergence



########### QML cross with no iterations
def qml_cross_noiter(maps,maps2,mask,covmap,ellbins,fwhmrad,guess,ds_dcb,spectra,plot=False,
    cholesky=True,polar=True,temp=True,EBTB=False,getmat=False, message=''):
    print('QML Estimator')
    ###### What will be calculated ?
    print('Temp='+str(temp))
    print('Polar='+str(polar))
    print('EBTB='+str(EBTB))
    all=['I','Q','U']
    if EBTB:
        xx=['TT','EE','BB','TE','EB','TB']
        ind=[1,2,3,4,5,6]
    else:
        xx=['TT','EE','BB','TE']
        ind=[1,2,3,4]
        
    if  not temp:
        all=['Q','U']
        if EBTB:
            xx=['EE','BB','EB']
            ind=[2,3,5]
        else:
            xx=['EE','BB']
            ind=[2,3]
    if not polar:
        all=['I']
        xx=['TT']
        ind=[1]
    print('Number of maps :',len(maps))
    print('Stokes parameters :',all)
    print('Spectra considered. :',xx)
    nspec=len(xx)

    if len(maps) != len(all):
        print('Inconstitent number of maps and Stokes parameters: Exiting !')
        return -1

    ### maps information
    nside=hp.npix2nside(len(maps[0]))
    ip=np.arange(12*nside**2)
    ipok=ip[~mask]
    npix=len(ipok)
    npixtot=len(all)*npix
    masktot=np.zeros(len(all)*len(mask),dtype=bool)
    map=np.zeros(len(all)*npix)
    for i in np.arange(len(all)):
        map[i*npix:(i+1)*npix]=maps[i][~mask]
        masktot[i*len(mask):(i+1)*len(mask)]=mask

    map2=np.zeros(len(all)*npix)
    for i in np.arange(len(all)):
        map2[i*npix:(i+1)*npix]=maps2[i][~mask]
        masktot[i*len(mask):(i+1)*len(mask)]=mask

    ### ell binning
    nbins=len(ellbins)-1
    ellmin=np.array(ellbins[0:nbins])
    ellmax=np.array(ellbins[1:nbins+1])-1
    ellval=(ellmin+ellmax)*1./2
    ll=np.arange(int(np.max(ellbins))+1)
    bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)

    ### Initial guess
    nbinstot=nspec*nbins
    specinit=np.zeros(nbinstot)
    for s in np.arange(nspec):
        specinit[s*nbins:(s+1)*nbins]=guess[ind[s]]
    thespectrum=np.zeros((nbinstot,1))
    thespectrum[:,0]=specinit

    ##############################################################################
    # dS/dC Calculation ##########################################################
    if ds_dcb is None:
        ds_dcb=compute_ds_dcb_par(ellbins,nside,ipok,bl,polar=polar,temp=temp,EBTB=EBTB)
    ##############################################################################
    
    num=0
    ##########################################################################
    ## pixpix covariance matrix for maps and solving z=M-1.m and likelihood
    print('    P2')
    matsky=np.zeros((npixtot,npixtot))
    for s in np.arange(nspec):
        for i in np.arange(nbins):
            matsky += thespectrum[s*nbins+i,num]*ds_dcb[s,i,:,:]
    print('       - Done matsky')
    if getmat:
        return matsky
    matcov=covmap+matsky
    
    if cholesky is True:
        print('      Doing Cholesky decomposition')
        U=scipy.linalg.cho_factor(matcov)
        z=scipy.linalg.cho_solve(U,map2)
    else:
        print('      Brute force inversion (No Cholesky)')
        minv=scipy.linalg.inv(matcov)
        z=np.dot(minv,map2)
    ##########################################################################


    ########################################################################
    # Solve the equations Wb=M^(-1).ds_dcb ##############################
    print('    P3')
    wb=np.zeros((nbinstot,npixtot,npixtot))
    for s in np.arange(nspec):
        pyquad.progress_bar(s,nspec)
        for i in np.arange(nbins):
            if cholesky is True:
                wb[s*nbins+i,:,:]=scipy.linalg.cho_solve(U,ds_dcb[s,i,:,:])
            else:
                wb[s*nbins+i,:,:]=np.dot(minv,ds_dcb[s,i,:,:])
    ########################################################################

    ########################################################################
    # First term mt.M-1.dsdcb.M-1.m ##############################
    print('    P4')
    term1=np.zeros(nspec*nbins)
    for i in np.arange(nbinstot):
        pyquad.progress_bar(i,nbinstot)
        term1[i]=np.dot(map,np.dot(wb[i,:,:],z))
    #######################################################################
         
    ##########################################################################
    # Fisher ################################
    print('    P5')
    fisher=np.zeros((nspec*nbins,nspec*nbins))
    kk=0
    for i in np.arange(nbinstot):
        for j in np.arange(i,nbinstot):
            pyquad.progress_bar(kk,(nbinstot+1)*nbinstot/2)
            wbt = wb[i,:,:].T
            fisher[i,j] = np.dot(np.ravel(wbt), np.ravel(wb[j,:,:]))
            #wbwb=np.dot(wb[i,:,:],wb[j,:,:])
            #fisher[i,j]=np.trace(wbwb)
            fisher[j,i]=fisher[i,j] 
            kk+=1
    invfisher = scipy.linalg.inv(fisher)  
    ###########################################################################
    
    spec = np.dot(invfisher, term1)       
    err=np.sqrt(np.diag(invfisher)) 
     
    finalspectrum=[ellval,0,0,0,0,0,0]
    error=[ellval,0,0,0,0,0,0]
    for s in np.arange(nspec):
        finalspectrum[ind[s]]=spec[s*nbins:(s+1)*nbins]
        error[ind[s]]=err[s*nbins:(s+1)*nbins]
        

    nplot_tot=1+nspec
    sqrn=int(np.sqrt(nplot_tot))
    nr=sqrn
    nc=sqrn+1
    iplot=0
    if plot is True:
        mp.clf()
        for s in np.arange(nspec):
            mp.subplot(nr,nc,iplot+1)
            mp.xlim(0,np.max(ellmax)*1.2)
            ell=spectra[0]
            cl=spectra[ind[s]]
            ellb=finalspectrum[0]
            mp.plot(ell,cl*(ell*(ell+1))/(2*np.pi),lw=3)
            msk = ell < np.max(ellmax)
            mp.ylim(0,np.max(cl[msk]*(ell[msk]*(ell[msk]+1))/(2*np.pi))*1.5)
            mp.plot(ellb,guess[ind[s]]*ellb*(ellb+1)/(2*np.pi),'go',alpha=0.5)
            mp.xlabel('$\ell$')
            mp.ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')
            mp.errorbar(ellb,finalspectrum[ind[s]]*ellb*(ellb+1)/(2*np.pi),
                error[ind[s]]*ellb*(ellb+1)/(2*np.pi),xerr=(ellmax+1-ellmin)/2,
                label=str(i),fmt='ro')
            mp.draw()
            mp.title(xx[s])
            iplot+=1
            
        ###########################################################################

    return finalspectrum,error,invfisher,ds_dcb
    











