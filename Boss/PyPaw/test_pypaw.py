from __future__ import division
from pylab import *
from matplotlib.pyplot import *
import numpy as np

from PyPaw import pypaw


##### Read the file as a dictionnary
datad = pypaw.readfile2dict('xillbinsymbb.out.2')

##### simple plots directly using python
clf()
plot(datad['x0'], datad['x1'],',',label='Jim is the best')
xlim([-0.05, 0.05])
ylim([0,0.01])
xlabel('toto')
ylabel('tata')
legend(loc='upper right')

clf()
hist(datad['x0'],range=[-0.02,0.02],bins=100)
mask = (datad['x1'] < 0.001) & (datad['x2'] > 300)
hist(datad['x0'][mask],range=[-0.02,0.02],bins=100)




##### Read the file as a dictionnary
datad = pypaw.readfile2dict('xillbinsymbb.out.2',keys=['a','b','c','d'])

##### simple plots directly using python
clf()
plot(datad['a'], datad['b'],',',label='Jim is the best')
xlim([-0.05, 0.05])
ylim([0,0.01])
xlabel('toto')
ylabel('tata')
legend(loc='upper right')

clf()
hist(datad['a'],range=[-0.02,0.02],bins=100)
mask = (datad['b'] < 0.001) & (datad['c'] > 300)
hist(datad['a'][mask],range=[-0.02,0.02],bins=100)




##### Read the file as a [nlines x ncolumns array]
data = pypaw.readfile('xillbinsymbb.out.2')

##### simple plots directly using python
clf()
plot(data[0,:], data[1,:],',',label='Jim is the best')
xlim([-0.05, 0.05])
ylim([0,0.01])
xlabel('toto')
ylabel('tata')
legend(loc='upper right')

clf()
hist(data[0,:],range=[-0.02,0.02],bins=100)
mask = (data[1,:] < 0.001) & (data[2,:] > 300)
hist(data[0,mask],range=[-0.02,0.02],bins=100)




## also works:
a, b, c, d = pypaw.readfile('xillbinsymbb.out.2')


##### simple plots directly using python
clf()
plot(a,b,',',label='Jim is the best')
xlim([-0.05, 0.05])
ylim([0,0.01])
xlabel('toto')
ylabel('tata')
legend(loc='upper right')

clf()
hist(a,range=[-0.02,0.02],bins=100)
mask = (b < 0.001) & (c > 300)
hist(a[mask],range=[-0.02,0.02],bins=100)

#### same plots using paw-like functions

################## Scatter Plots
## simple use
pypaw.scatter(a, b)

## adding xlabel and ylabel
pypaw.scatter(a, b, xlab='toto',ylab='tata')

## change marker (see http://matplotlib.org/api/markers_api.html )
pypaw.scatter(a, b, marker='o')
pypaw.scatter(a, b, marker='+')
pypaw.scatter(a, b, marker='.')

## make cuts
pypaw.scatter(a, b, cut=(c > 300))
pypaw.scatter(a, b, cut=(c > 300) & (b < 0.005))

## superimpose plots with various cuts
pypaw.scatter(a, b, cut=(c > 300), marker='o')
pypaw.scatter(a, b, cut=(c > 300) & (b < 0.005), clearscreen=False, marker='o')
savefig('niceplot.png')

## use transparency
pypaw.scatter(a, b, cut=(c > 100), marker='.',alpha=0.5)
pypaw.scatter(a, b, cut=(c > 300), clearscreen=False, marker='.', alpha=0.5)
pypaw.scatter(a, b, cut=(c > 400), clearscreen=False, marker='.', alpha=0.5)

## colors
pypaw.scatter(a, b, cut=(c > 300))
pypaw.scatter(a, b, cut=(c > 300), color='red')
pypaw.scatter(a, b, cut=(c > 300) & (b < 0.005), color='yellow')

pypaw.scatter(a, b, cut=(c > 300), marker='o', color='red')
pypaw.scatter(a, b, cut=(c > 300) & (b < 0.005), clearscreen=False, marker='o', color='green')
savefig('niceplot.png')

## add comments
pypaw.scatter(a, b, cut=(c > 100), alpha=0.5, text='option 1')
pypaw.scatter(a, b, cut=(c > 300), clearscreen=False, alpha=0.5, text='option 2')
pypaw.scatter(a, b, cut=(c > 400), clearscreen=False, alpha=0.5, text='option 3')



################# Histograms
pypaw.histo(a)
pypaw.histo(a,bins=100)
pypaw.histo(a,bins=100, range=[-0.02, 0.02])

pypaw.histo(a,bins=100, range=[-0.02, 0.02], cut=(c > 300))


pypaw.histo(a,bins=100, range=[-0.03, 0.03], cut=(c > 100),alpha=0.3, color='red', xlab='truc')
pypaw.histo(a,bins=100, range=[-0.03, 0.03], cut=(c > 300), clearscreen=False, alpha=0.3, color='blue')
pypaw.histo(a,bins=100, range=[-0.03, 0.03], cut=(c > 400), clearscreen=False, alpha=0.3, color='magenta')

## add comments
pypaw.histo(a,bins=100, range=[-0.03, 0.03], cut=(c > 100),alpha=0.3, color='red', xlab='truc', text='Option 1')
pypaw.histo(a,bins=100, range=[-0.03, 0.03], cut=(c > 300), clearscreen=False, alpha=0.3, color='blue', text='Option 2')
pypaw.histo(a,bins=100, range=[-0.03, 0.03], cut=(c > 400), clearscreen=False, alpha=0.3, color='magenta', text='Option 3')

## return the displayed statistical informations
a1=pypaw.histo(a,bins=100, range=[-0.03, 0.03], cut=(c > 100),alpha=0.3, color='red', xlab='truc', text='Option 1')
a2=pypaw.histo(a,bins=100, range=[-0.03, 0.03], cut=(c > 300), clearscreen=False, alpha=0.3, color='blue', text='Option 2')
a3=pypaw.histo(a,bins=100, range=[-0.03, 0.03], cut=(c > 400), clearscreen=False, alpha=0.3, color='magenta', text='Option 3')
print(a1)
print(a2)
print(a3)


n1, m1, s1 = pypaw.histo(a,bins=100, range=[-0.03, 0.03], cut=(c > 100),alpha=0.3, color='red', xlab='truc', text='Option 1')
n2, m2, s2 = pypaw.histo(a,bins=100, range=[-0.03, 0.03], cut=(c > 300), clearscreen=False, alpha=0.3, color='blue', text='Option 2')
n3, m3, s3 = pypaw.histo(a,bins=100, range=[-0.03, 0.03], cut=(c > 400), clearscreen=False, alpha=0.3, color='magenta', text='Option 3')
print(n1,m1,s1)
print(n2,m2,s2)
print(n3,m3,s3)










