from __future__ import division

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import scipy.special
import scipy.linalg
import time
import sys
import multiprocessing
import random
import string

def random_string(nchars):
    lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(nchars)]
    str = "".join(lst)
    return(str)

def cov2cor(mat):
    cor=np.zeros((mat.shape[0],mat.shape[1]))
    diagvals=np.diag(mat)
    for i in np.arange(mat.shape[0]):
        progress_bar(i,mat.shape[0])
        for j in np.arange(i,mat.shape[1]):
            cor[i,j]=mat[i,j]/np.sqrt(diagvals[i]*diagvals[j])
            cor[j,i]=cor[i,j]
    return(cor)

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


def meancut(array,nsig=3,niter=3):
    thearray=array
    for i in np.arange(niter):
        m=np.mean(thearray)
        s=np.std(thearray)
        w=np.where(np.abs(thearray-m) <= nsig*s)
        thearray=thearray[w[0]]
    
    return(m,s)

def profile(x,y,xmin,xmax,nbins,bins=0,plot=True,fmt='bo',dispersion=True):
    if bins is 0:
        bins=np.linspace(xmin,xmax,nbins+1)
    else:
        nbins=len(bins)-1
    indices=np.digitize(x,bins)
    ynew=np.array([y[indices == i].mean() for i in np.arange(1,len(bins))])
    if dispersion is True:
        dynew=np.array([y[indices == i].std() for i in np.arange(1,len(bins))])
    else:
        dynew=np.array([y[indices == i].std()/np.sqrt(len(y[indices == i])) for i in np.arange(1,len(bins))])
    xnew=(bins[:nbins]+bins[1:nbins+1])/2
    dxnew=(bins[1:nbins+1]-bins[:nbins])/2
    cond=np.isfinite(ynew) & (dynew != 0)
    xnew=xnew[cond == True]
    ynew=ynew[cond == True]
    dxnew=dxnew[cond == True]
    dynew=dynew[cond == True]
    if plot is True: mp.errorbar(xnew,ynew,xerr=dxnew,yerr=dynew,fmt=fmt)
    return(xnew,ynew,dxnew,dynew)

def binspectrum(spectra,ellmin,ellmax):
    ellval=0.5*(ellmin+ellmax)
    ellbins=ellmin.size
    ell=spectra[0]
    binspec=np.zeros((ellbins,5))
    for i in np.arange(ellbins):
        for j in np.arange(4)+1:
            thespec=spectra[j]
            w=np.where((ell >= ellmin[i]) & (ell < ellmax[i]))
            norm=ell[w]*(ell[w]+1)/(ellval[i]*(ellval[i]+1))
            #norm=ell[w]*0+1
            mask=ell[w]<2
            norm[mask]=0
            binspec[i,j]=np.mean(thespec[w]*norm)

    binspec[:,0]=ellval
    return(binspec)


def wfsimple(map,mask,fwhmrad,maxell,verbose=True):
    if verbose: print('Calculating window functions: ')
    npix=np.size(np.where(mask))
    nside=np.sqrt(map.size/12)
    iprings=np.arange(map.size)
    if verbose: print(' * allocating array: '+str('%.0f'%maxell)+str(' x %.0f'%npix)+str(' x %.0f'%npix))
    vecs=hp.pix2vec(int(nside),iprings[mask])
    cosangles=np.dot(np.transpose(vecs),vecs)
    wf=np.zeros((maxell,npix,npix))
    ll=np.arange(maxell)
    bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)
    for i in np.arange(npix):
        sys.stdout.write("\r * Pixel# %d"%i + str(' out of %.0f'%npix))
        sys.stdout.flush()
        for j in np.arange(i,npix):
            wf[:,i,j]=scipy.special.lpn(maxell-1,cosangles[i,j])[0]*bl**2
            wf[:,j,i]=wf[:,i,j]
    return(wf)

def compute_ds_dcb(map,mask,ellmin,ellmax,fwhmrad):
    npix=np.size(np.where(mask))
    maxell=np.max(ellmax)
    ellbins=ellmin.size
    ellval=(ellmin+ellmax)/2
    print('    Calculating dS/dCb')
    pixwin=hp.pixwin(hp.npix2nside(len(map)))[0:maxell+1]
    ll=np.arange(int(maxell)+1)
    bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)
    
    nside=np.sqrt(map.size/12)
    iprings=np.arange(map.size)
    vecs=hp.pix2vec(int(nside),iprings[mask])
    cosangles=np.dot(np.transpose(vecs),vecs)
    
    ds_dcb=np.zeros((ellbins,npix,npix))
    for i in np.arange(npix):
        bla,aa=do_a_line(cosangles[i,:],i,ll,bl,pixwin,ellmin,ellmax,ellval)
        ds_dcb[:,i,i:]=bla[:,i:]
        ds_dcb[:,i:,i]=bla[:,i:]
    
    return(ds_dcb)


def progress_bar(i,n):
    if n != 1:
        ntot=50
        ndone=ntot*i/(n-1)
        a='\r|'
        for k in np.arange(ndone):
            a += '#'
        for k in np.arange(ntot-ndone):
            a += ' '
        a += '| '+str(int(i*100./(n-1)))+'%'
        sys.stdout.write(a)
        sys.stdout.flush()
        if i == n-1:
            sys.stdout.write(' Done \n')
            sys.stdout.flush()
            

def do_a_line(cosangles_line,i,ll,bl,pixwin,ellmin,ellmax,ellval):
    npix=cosangles_line.size
    ellbins=ellmin.size
    maxell=np.max(ellmax)+1
    sys.stdout.write("\r * Pixel# %d"%i + str(' out of %.0f'%npix))
    sys.stdout.flush()
    result=np.zeros((ellbins,npix))
    for j in np.arange(i,npix):
        pl=scipy.special.lpn(maxell-1,cosangles_line[j])[0]
        vals=(2*ll+1)/(4*np.pi)*pl[ll]*bl[ll]**2*pixwin[ll]**2
        vals[1:]=vals[1:]/(ll[1:]*(ll[1:]+1))
        for k in np.arange(ellbins):
            blo=vals[ellmin[k]:ellmax[k]+1]*ellval[k]*(ellval[k]+1)
            bla=np.sum(blo)
            result[k,j]=bla
    #print('Done line ',i)
    return(result,i)


def compute_ds_dcb_line_par(map,mask,ellmin,ellmax,fwhmrad,nthreads):
    npix=np.size(np.where(mask))
    ellbins=ellmin.size
    ellval=(ellmin+ellmax)/2
    print('    Calculating dS/dCb')
    pixwin=hp.pixwin(hp.npix2nside(len(map)))
    maxell=np.max(ellmax)
    ll=np.arange(int(maxell)+1)
    bl=np.exp(-0.5*ll**2*(fwhmrad/2.35)**2)
    
    nside=np.sqrt(map.size/12)
    iprings=np.arange(map.size)
    vecs=hp.pix2vec(int(nside),iprings[mask])
    cosangles=np.dot(np.transpose(vecs),vecs)

    nshots=npix//nthreads+1
    ntot=nshots*nthreads
    indices=np.arange(ntot)
    indices[indices>npix-1]=npix-1
    lines=np.reshape(indices,(nshots,nthreads))
    
    ds_dcb=np.zeros((ellbins,npix,npix))

    for num in np.arange(nshots):
        # initiate multithreading
        tasks=multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        # start consumers
        num_consumers = nthreads
        #print 'Creating %d consumers' % num_consumers
        consumers = [ Consumer(tasks, results)
                for i in xrange(num_consumers) ]
        for w in consumers:
            w.start()
        # Enqueue jobs
        num_jobs = nthreads
        for i in np.arange(num_jobs):
            #print('Calling task ',lines[num,i])
            tasks.put(Task(cosangles[lines[num,i],:],lines[num,i],ll,bl,pixwin,ellmin,ellmax,ellval))
        # poison them when finished
        for i in np.arange(num_jobs):
            #print 'Killing Task %d ' % i
            tasks.put(None)
            
        #print('****** Now ready to join tasks ******')
        tasks.join()
        #print('###### Tasks were joined ######')

        while num_jobs:
            #print('Getting result for task %d' % num_jobs)
            result = results.get()
            bla=result[0]
            num=result[1]
            ds_dcb[:,num,num:]=bla[:,num:]
            ds_dcb[:,num:,num]=bla[:,num:]
            num_jobs -= 1
    
    return(ds_dcb)


class Task(object):
    def __init__(self, cosangles_line,i,ll,bl,pixwin,ellmin,ellmax,ellval):
        self.cosangles_line = cosangles_line
        self.i = i
        self.ll = ll
        self.bl = bl
        self.pixwin = pixwin
        self.ellmin = ellmin
        self.ellmax = ellmax
        self.ellval = ellval
    def __call__(self):
        return(do_a_line(self.cosangles_line, self.i, self.ll, self.bl, self.pixwin, self.ellmin, self.ellmax, self.ellval))




##############################################################################




def quadest(map,mask,covmap,ellmin,ellmax,fwhmrad,guess,cltemp,ds_dcb,itmax=20,plot=False,cholesky=True):
    npix=np.size(np.where(mask))
    ellbins=ellmin.size
    ellval=(ellmin+ellmax)/2
    specinit=np.zeros(ellbins)+guess
    num=0
    convergence=0
    lk=np.zeros(itmax)
    thespectrum=np.zeros((ellbins,itmax+1))
    thespectrum[:,0]=specinit
    
    ####################### P1 Calcul des dS/dC ##################################
    if ds_dcb is 0:
        ds_dcb=compute_ds_dcb(map,mask,covmap,ellmin,ellmax,fwhmrad)
    ##############################################################################

    while convergence==0:
        ########################################################################
        # P2 pixpix covariance matrix for map and solving of z=M^(-1).m and likelihood
        print('    P2')
        print(npix)
        matsky=np.zeros((npix,npix))
        for i in np.arange(ellbins):
            matsky += thespectrum[i,num]*ds_dcb[i,:,:]
            progress_bar(i,ellbins)
    
        matcov=covmap+matsky
        
        if cholesky is True:
            print('      Doing Cholesky decomposition')
            U=scipy.linalg.cho_factor(matcov)
            z=scipy.linalg.cho_solve(U,map[mask])
        else:
            print('      Brute force inversion (No Cholesky)')
            minv=scipy.linalg.inv(matcov)
            z=np.dot(minv,map[mask])
                
        mapw=np.zeros(map.size)
        mapw[mask]=z
        #hp.gnomview(mapw,rot=[0,90],reso=15)
        lk[num]=-0.5*(np.dot(z,map[mask])+np.sum(np.log(np.diag(matcov))))
        ########################################################################
    
        ########################################################################
        # P3 Solve the equations Wb=M^(-1).ds_dcb ##############################
        print('    P3')
        wb=np.zeros((ellbins,npix,npix))
        for i in np.arange(ellbins):
            progress_bar(i,ellbins)
            if cholesky is True:
                wb[i,:,:]=scipy.linalg.cho_solve(U,ds_dcb[i,:,:])
            else:
                wb[i,:,:]=np.dot(minv,ds_dcb[i,:,:])
        ########################################################################

        ########################################################################
        # P4 First derivatives of the likelihood ##############################
        print('    P4')
        dldcb=np.zeros(ellbins)
        for i in np.arange(ellbins):
            progress_bar(i,ellbins)
            dldcb[i]=0.5*(np.dot(map[mask],np.dot(wb[i,:,:],z))-np.trace(wb[i,:,:]))
        #######################################################################
    
        ##########################################################################
        # P5 second derivatives of the likelihood ################################
        print('    P5')
        d2ldcbdcb=np.zeros((ellbins,ellbins))
        fisher=np.zeros((ellbins,ellbins))
        kk=0
        for i in np.arange(ellbins):
            for j in np.arange(i,ellbins):
                progress_bar(kk,(ellbins+1)*ellbins/2)
                wbwb=np.dot(wb[i,:,:],wb[j,:,:])
                fisher[i,j]=np.trace(wbwb)
                d2ldcbdcb[i,j]=-np.dot(map[mask],np.dot(wbwb,z))+0.5*fisher[i,j]
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
    
        ############################################################################
        # Critere de convergence ###################################################
        conv=deltac/np.sqrt(np.abs(np.diag(invfisher)))
        print('     Conv : ',max(abs(conv)))
        print('     Likelihood : ',lk[num])
        if plot is True:
            mp.clf()
            mp.subplot(211)
            mp.plot(lk[:num],'o-')
            mp.draw()
            mp.xlabel('Iteration')
            mp.ylabel('Likelihood')
            mp.subplot(212)
            mp.xlim(0,np.max(ellmax)*1.3)
            ell=np.arange(cltemp.size)+2
            mp.plot(ell,cltemp*(ell*(ell+1))/(2*np.pi),lw=3)
            mp.plot(ellval,guess*ellval*(ellval+1)/(2*np.pi),'o')
            mp.xlabel('$\ell$')
            mp.ylabel('$\ell(\ell+1)C_\ell/(2\pi)$')
            mp.errorbar(ellval,thespectrum[:,num+1]*ellval*(ellval+1)/(2*np.pi),err*ellval*(ellval+1)/(2*np.pi),xerr=(ellmax+1-ellmin)/2,label=str(i),fmt='o')
            mp.draw()
                
        if np.max(np.abs(conv)) <= 0.01 or num==itmax-1:
            convergence=1
        else:
            num=num+1
        ###########################################################################

    return(thespectrum,err,invfisher,lk,num)

