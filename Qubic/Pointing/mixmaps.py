from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import os
import healpy as hp
from scipy.ndimage import gaussian_filter1d
import scipy
from scipy import integrate
from scipy import interpolate
from scipy import ndimage


#### Get input Power spectra
a=np.loadtxt('/Users/hamilton/Qubic/cl_r=0.1bis2.txt')
#a=np.loadtxt('./cl_r=0.1bis2.txt')
ell=a[:,0]
ctt=np.concatenate([[0,0],a[:,1]*1e12*2*np.pi/(ell*(ell+1))])
cee=np.concatenate([[0,0],a[:,2]*1e12*2*np.pi/(ell*(ell+1))])
cte=np.concatenate([[0,0],a[:,4]*1e12*2*np.pi/(ell*(ell+1))])
cbb=np.concatenate([[0,0],a[:,7]*1e12*2*np.pi/(ell*(ell+1))])
ell=np.concatenate([[0,1],ell])
spectra=[ell,ctt,cee,cbb,cte]   #### This is the new ordering

clf()
plot(ell,np.sqrt(spectra[1]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TT}$')
plot(ell,np.sqrt(abs(spectra[4])*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{TE}$')
plot(ell,np.sqrt(spectra[2]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{EE}$')
plot(ell,np.sqrt(spectra[3]*(ell*(ell+1))/(2*np.pi)),label='$C_\ell^{BB} (r=0.1)$')
yscale('log')
xlim(0,600)
ylim(0.01,100)
xlabel('$\ell$')
ylabel('$\sqrt{\ell(\ell+1)C_\ell/(2\pi)}$'+'    '+'$[\mu K]$ ')
legend(loc='lower right',frameon=False)



nside=128
lmax=3*nside
ell=spectra[0]
maskl=ell<(lmax+1)
bl=np.ones(maskl.size)

fwhmrad=0.5*np.pi/180
mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
hp.mollview(mapi)
hp.mollview(mapq)
hp.mollview(mapu)
maps=[mapi,mapq,mapu]


def rhoepsilon_from_maps(QUin,QUout,QUNoiseout):
    mapq=QUin[0]
    mapu=QUin[1]
    qprime=QUout[0]
    uprime=QUout[1]
    noiseQ=QUNoiseout[0]
    noiseU=QUNoiseout[1]
    ### voire code mathematica leakage_QU_pointing.nb
    qu_sq2=np.sum(mapq*mapu/noiseQ**2)
    qu_su2=np.sum(mapq*mapu/noiseU**2)
    qpu_sq2=np.sum(qprime*mapu/noiseQ**2)
    qup_su2=np.sum(mapq*uprime/noiseU**2)
    q2_su2=np.sum(mapq**2/noiseU**2)
    u2_su2=np.sum(mapu**2/noiseU**2)
    u2_sq2=np.sum(mapu**2/noiseQ**2)
    q2_sq2=np.sum(mapq**2/noiseQ**2)
    qqp_sq2=np.sum(mapq*qprime/noiseQ**2)
    uup_su2=np.sum(mapu*uprime/noiseU**2)
    rhorec=((qu_sq2 - qu_su2)*(qpu_sq2 - qu_sq2 + qu_su2 - qup_su2) + (q2_su2 + u2_sq2)*(q2_sq2 - qqp_sq2 + u2_su2 - uup_su2))/((qu_sq2 - qu_su2)**2 - (q2_su2 + u2_sq2)*(q2_sq2 + u2_su2))
    epsilonrec=(-(qpu_sq2 - qup_su2)*(q2_sq2 + u2_su2) + (qu_sq2 - qu_su2)*(qqp_sq2 + uup_su2))/((qu_sq2 - qu_su2)**2 - (q2_su2 + u2_sq2)*(q2_sq2 + u2_su2))
    return(rhorec,epsilonrec)

###########
# define rho and epsilon
#  ( 1+rho   e   )
#  ( -e    1+rho )
###########
rho=-0.01
epsilon=0.02
noise=1.
noiseQ=noise
noiseU=noiseQ

nbmc=100
rhorec=np.zeros(nbmc)
epsilonrec=np.zeros(nbmc)
for i in np.arange(nbmc):
    print(i)
    mapi,mapq,mapu=hp.synfast(spectra[1:],nside,fwhm=fwhmrad,pixwin=True,new=True)
    qprime=(1+rho)*mapq+epsilon*mapu+np.random.normal(0.,noiseQ,len(mapq))
    uprime=-epsilon*mapq+(1+rho)*mapu+noise+np.random.normal(0.,noiseU,len(mapq))
    rhorec[i],epsilonrec[i]=rhoepsilon_from_maps([mapq,mapu],[qprime,uprime],[noiseQ,noiseU])

mrho=np.mean(rhorec)
srho=np.std(rhorec)
mepsilon=np.mean(epsilonrec)
sepsilon=np.std(epsilonrec)


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


def matrixplot(chain,vars,col,sm,limits=None,nbins=None,doit=None,truevals=None):
    nplots=len(vars)
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
                if i == nplots-1: xlabel(vars[j])
                var=vars[j]
                xlim(limits[i])
                ylim(0,1.2)
                if (var in chain.keys()) and (doit[j]==True):
                    if nbins is None: nbins=100
                    bla=np.histogram(chain[var],bins=nbins,normed=True)
                    xhist=(bla[1][0:nbins]+bla[1][1:nbins+1])/2
                    yhist=gaussian_filter1d(bla[0],ss[i]/5/(xhist[1]-xhist[0]))
                    plot(xhist,yhist/max(yhist),color=col)
                    if truevals: plot([truevals[i],truevals[i]],[0,1.2],'k--')
            if (i>j):
                a=subplot(nplots,nplots,num)
                a.tick_params(labelsize=8)
                var0=vars[j]
                var1=vars[i]
                xlim(limits[j])
                ylim(limits[i])
                if i == nplots-1: xlabel(var0)
                if j == 0: ylabel(var1)
                if (var0 in chain.keys()) and (var1 in chain.keys()) and (doit[j]==True) and (doit[i]==True):
                    a0=cont(chain[var0],chain[var1],color=col,nsmooth=sm)
                    if truevals:
                        plot([truevals[j],truevals[j]],limits[i],'k--')
                        plot(limits[j],[truevals[i],truevals[i]],'k--')
                        plot(truevals[j],truevals[i],'k+',ms=10,mew=3)

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

    
chain={}
chain['rho']=rhorec
chain['epsilon']=epsilonrec
clf()
nb=5
nsm=5
a=matrixplot(chain,['rho','epsilon'],'blue',nsm,truevals=[rho,epsilon],limits=[[mrho-nb*srho,mrho+nb*srho],[mepsilon-nb*sepsilon,mepsilon+nb*sepsilon]])







