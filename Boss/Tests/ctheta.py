import numpy as np
import pyfits
import subprocess
from pysimulators import FitsArray

docfmpi = "/Users/hamilton/Python/Boss/Tests/newdocfmpi.py"
mpirun = "openmpirun"


def rnd_on_sphere(nn, dtheta):
	cdth = np.cos(np.radians(dtheta))
	cth = 1-(1-cdth)*np.random.rand(nn)
	th = np.arccos(cth)
	ph = np.random.rand(nn)*2*np.pi
	return th, ph



nb = 5000
dtheta = 90
th, ph = rnd_on_sphere(nb, dtheta)
thrnd, phrnd = rnd_on_sphere(nb*3, dtheta)

clf()
subplot(211,projection='mollweide')
title('Data')
plot(ph-np.pi,np.pi/2-th,',')
subplot(212,projection='mollweide')
title('Random')
plot(phrnd-np.pi,np.pi/2-thrnd,'r,')

ra = np.degrees(ph)
dec = 90 - np.degrees(th)
rarnd = np.degrees(phrnd)
decrnd = 90 - np.degrees(thrnd)

clf()
subplot(211,projection='mollweide')
title('Data')
plot(np.radians(ra-180),np.radians(dec),',')
subplot(212,projection='mollweide')
title('Random')
plot(np.radians(rarnd-180),np.radians(decrnd),'r,')



def radec2fits(fitsname, ra, dec):
  nb = ra.size
  x = ra
  y = dec
  z = ra * 0
  #Writing Fits file
  col0=pyfits.Column(name='x',format='E',array=x)
  col1=pyfits.Column(name='y',format='E',array=y)
  col2=pyfits.Column(name='z',format='E',array=z)
  cols=pyfits.ColDefs([col0,col1,col2])
  tbhdu=pyfits.new_table(cols)
  tbhdu.writeto(fitsname,clobber=True)
  return 1




def run_kdtree(datafile1,datafile2,binsfile,resfile,nproc=None):
    if nproc is None:
        subprocess.call(["/Library/Frameworks/EPD64.framework/Versions/Current/bin/python",
                         docfmpi,
                         "--counter=angular",
                         "-b",binsfile,
                         "-o",resfile,
                         #"-w",
                         datafile1,
                         datafile2])
    else:
        subprocess.call([mpirun,
                         "-np",str(nproc),
                         "/Library/Frameworks/EPD64.framework/Versions/Current/bin/python",
                         docfmpi,
                         "--counter=angular",
                         "-b",binsfile,
                         "-o",resfile,
                         #"-w",
                         datafile1,
                         datafile2])



############ MapReduce Kd-Tree pair counting
datafile = 'data.fits'
randomfile = 'random.fits'
binsfile = 'bins.txt'
thmin = 0
thmax = 90
nbins = 90
nproc = 10

radec2fits(datafile, ra, dec)
radec2fits(randomfile, rarnd, decrnd)

edges = np.linspace(thmin,thmax, nbins+1)
outfile=open(binsfile,'w')
for x in edges:
  outfile.write("%s\n" % x)
outfile.close()


def get_xi(datafile, randomfile,binsfile,nproc=nproc):
	ddfile = 'dd.txt'
	rrfile = 'rr.txt'
	drfile = 'dr.txt'

	# do DD
	print('       - Doing DD : '+str(ra.size)+' elements')
	run_kdtree(datafile,datafile,binsfile,ddfile,nproc=nproc)
	alldd=np.loadtxt(ddfile,skiprows=1)
	# do RR
	print('       - Doing RR : '+str(rarnd.size)+' elements')
	run_kdtree(randomfile,randomfile,binsfile,rrfile,nproc=nproc)
	allrr=np.loadtxt(rrfile,skiprows=1)
	# do DR
	print('       - Doing DR : '+str(ra.size)+'x'+str(rarnd.size)+' pairs')
	run_kdtree(datafile,randomfile,binsfile,drfile,nproc=nproc)
	alldr=np.loadtxt(drfile,skiprows=1)

	dd=alldd[:,2]
	rr=allrr[:,2]
	dr=alldr[:,2]

	# correct DD and RR for double counting
	dd=dd/2
	rr=rr/2

	r=(alldd[:,0]+alldd[:,1])/2

	# Normalize results
	ng=ra.size
	nr=rarnd.size
	dd=dd/(ng*(ng-1)/2)
	rr=rr/(nr*(nr-1)/2)
	dr=dr/(ng*nr)

	# Landy-Szalay and Peebles-Hauser
	ls = (dd-2*dr+rr)/rr
	ph = dd/rr-1

	return r,ls,ph,dd,rr,dr



r, ls, ph = get_xi(datafile, randomfile,binsfile,nproc=nproc)

clf()
plot(r,ls,'ro')
plot(r,ph,'bo')
ylim(-0.1,0.1)


#### Make a MCMC
nb = 100
dtheta = 90
nbmc = 1000
nbrnd = 10
datafile = 'data.fits'
randomfile = 'random.fits'
binsfile = 'bins.txt'
thmin = 0
thmax = 90
nbins = 90
nproc = 8

edges = np.linspace(thmin,thmax, nbins+1)
outfile=open(binsfile,'w')
for x in edges:
  outfile.write("%s\n" % x)
outfile.close()

all_ls = np.zeros((nbmc, nbins))
all_dd = np.zeros((nbmc, nbins))
all_rr = np.zeros((nbmc, nbins))
all_dr = np.zeros((nbmc, nbins))
all_ph = np.zeros((nbmc, nbins))
for i in np.arange(nbmc):
	print("\n MC = "+str(i))
	th, ph = rnd_on_sphere(nb, dtheta)
	thrnd, phrnd = rnd_on_sphere(nb*nbrnd, dtheta)
	ra = np.degrees(ph)
	dec = 90 - np.degrees(th)
	rarnd = np.degrees(phrnd)
	decrnd = 90 - np.degrees(thrnd)
	radec2fits(datafile, ra, dec)
	radec2fits(randomfile, rarnd, decrnd)
	r, ls, ph, dd, rr, dr = get_xi(datafile, randomfile,binsfile,nproc=nproc)
	all_ls[i,:] = ls
	all_ph[i,:] = ph
	all_dd[i,:] = dd
	all_dr[i,:] = dr
	all_rr[i,:] = rr

## Store simulated results
FitsArray([all_ls, all_ph, all_dd, all_dr, all_rr], copy=False).save('all_nb'+str(nb)+'.fits')



## restore simulated results and Analyse them
nb=5000
thels, theph, thedd, thedr, therr = FitsArray('all_nb'+str(nb)+'.fits')

### theoretical errors
gp = np.mean(thedr, axis=0)
p = 2./(nb*(nb-1)*gp)
d = np.mean(thedd/therr, axis=0)
ls_th_err = np.sqrt(d**2*p)


### Get Average
mls =  np.mean(thels, axis=0)
mph =  np.mean(theph, axis=0)

### Get RMS
sls =  np.std(thels, axis=0)
sph =  np.std(theph, axis=0)

### Get intervals containg a fraction of sims
def value_in_fraction(data, level=0.682):
	mdata = np.mean(data, axis=0)
	sh = data.shape
	lim = np.zeros(sh[1])
	for i in np.arange(sh[1]):
		allvals = np.abs(data[:,i]-mdata[i])
		sortedvals = np.sort(allvals)
		lim[i] = sortedvals[np.int(sh[0]*level)]
	return(lim)

lim1_ph = value_in_fraction(theph, level = 0.682)
lim2_ph = value_in_fraction(theph, level = 0.954)
lim3_ph = value_in_fraction(theph, level = 0.997)
lim1_ls = value_in_fraction(thels, level = 0.682)
lim2_ls = value_in_fraction(thels, level = 0.954)
lim3_ls = value_in_fraction(thels, level = 0.997)

clf()
subplot(221)
title("nb="+str(nb))
ylim(-3*np.max(lim3_ph[20:]), 3*np.max(lim3_ph[20:]))
plot(np.degrees(r),mph*0,'k--')
ylabel('Correlation Function')
xlabel('Angle [deg]')
fill_between(np.degrees(r),mph+lim3_ph,y2=mph-lim3_ph,color='blue',alpha=0.1)
fill_between(np.degrees(r),mph+lim2_ph,y2=mph-lim2_ph,color='blue',alpha=0.2)
fill_between(np.degrees(r),mph+lim1_ph,y2=mph-lim1_ph,color='blue',alpha=0.3)
plot(np.degrees(r),mph+sph,'b--')
plot(np.degrees(r),mph-sph,'b--')
plot(np.degrees(r),mph,'bo',label='Peebles-Hauser')
legend(frameon=False,fontsize=10)

subplot(222)
title("nb="+str(nb))
ylim(-3*np.max(lim3_ph[20:]), 3*np.max(lim3_ph[20:]))
plot(np.degrees(r),mls*0,'k--')
ylabel('Correlation Function')
xlabel('Angle [deg]')
fill_between(np.degrees(r),mls+lim3_ls,y2=mls-lim3_ls,color='red',alpha=0.1)
fill_between(np.degrees(r),mph+lim2_ls,y2=mls-lim2_ls,color='red',alpha=0.2)
fill_between(np.degrees(r),mls+lim1_ls,y2=mls-lim1_ls,color='red',alpha=0.3)
plot(np.degrees(r),mls+sls,'r--')
plot(np.degrees(r),mls-sls,'r--')
plot(np.degrees(r),mls,'ro',label='Landy-Szalay')
legend(frameon=False,fontsize=10)

subplot(223)
title("nb="+str(nb))
ylabel('RMS on Correlation Function')
xlabel('Angle [deg]')
ylim(0,np.max(sph))
plot(np.degrees(r),lim3_ph,'b',alpha=0.1)
plot(np.degrees(r),lim2_ph,'b',alpha=0.2)
plot(np.degrees(r),lim1_ph,'b',alpha=0.3)
plot(np.degrees(r),sph,'b--',label='Peebles-Hauser RMS')
plot(np.degrees(r),lim3_ls,'r',alpha=0.1)
plot(np.degrees(r),lim2_ls,'r',alpha=0.2)
plot(np.degrees(r),lim1_ls,'r',alpha=0.3)
plot(np.degrees(r),sls,'r--',label='Landy-Szalay RMS')
plot(np.degrees(r),ls_th_err,'r:',label='Landy-Szalay Theoretical RMS')
legend(frameon=False,fontsize=10)


subplot(2,2,4)
title("nb="+str(nb))
hist(mph,range=[-np.max(sph)/np.sqrt(nbmc), np.max(sph)/np.sqrt(nbmc)],bins=10,alpha=0.3,color='b',label='Peebles-Hauser')
hist(mls,range=[-np.max(sph)/np.sqrt(nbmc), np.max(sph)/np.sqrt(nbmc)],bins=10,alpha=0.3,color='r',label='Landy-Szalay')
xlabel('Correlation Function')
legend(frameon=False,fontsize=10)










