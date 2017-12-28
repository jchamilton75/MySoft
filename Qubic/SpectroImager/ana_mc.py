import os
import qubic
import healpy as hp
import numpy as np
import pylab as plt
import matplotlib as mpl
import sys

######### Read simulation



# Code matt
dbdir = “/sps/hep/qubic/Simus”
import glob
x0 = fits.getdata( dbdir+“/input_cmb_ns128.fits”)

#read simus
name = “mc_128”
files = glob.glob( “%s/%s/simu*_nrec4_outmaps.fits” % (dbdir,name))
len(files)

nsub_band = [1,2,4]

covmap = zeros( (3,3,nside2npix(nside)))+UNSEEN
for nrec in nsub_band:
    print( “read simu for nrec=%d” % nrec)
    allmaps = []
    for f in files:
        allmaps.append( fits.getdata( f.replace(“nrec4",“nrec%d”%nrec)))
    
   allmaps = array(allmaps)
    seen = where(array(allmaps[0,0,:,1],float32) != UNSEEN)[0]
    x0[~seen,:] = UNSEEN
    
   #combine the nsubmaps for each pixel
    print( “combined cov for nrec=%d” % nrec)
    for p in seen:
        for t in [0,1,2]:
            mat = cov( allmaps[:,:,p,t].T)
            if size(mat) == 1: covmap[nsub_band.index(nrec),t,p] = mat
            else: covmap[nsub_band.index(nrec),t,p] = 1./sum(inv(mat))