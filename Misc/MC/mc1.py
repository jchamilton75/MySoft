# -*- coding: utf-8 -*-
from pylab import *
from numpy import *
from matplotlib import *
import pylab as plt

#1.1 DISTRIBUTION UNIFORME (DU)
#==============================

#Initialisation de la fenêtre graphique
plt.close("all")
plt.figure(1)
#Données initiales (nous allons faire varier le nombre de tirages et le nombre de bins de l'histo)
nbr_tirages=[1e3,1e6]
bins=[50,100]
#DU--1e6 tirage--100 bins
x=rand(nbr_tirages[1])
plt.subplot(231)
h=hist(x,bins[1])
plt.title("DU initiale: %d tirages, %d bins"%(nbr_tirages[1],bins[1]))
#Change le nombre de tirages
x=rand(nbr_tirages[0])
plt.subplot(232)
h=hist(x,bins[1])
plt.title("Changement: %d tirages"%(nbr_tirages[0]))
#Change le nombre de bins
x=rand(nbr_tirages[0])
plt.subplot(233)
h=hist(x,bins[0])
plt.title("Changement: %d bins"%(bins[0]))
#DU entre 2 et 10
x=rand(nbr_tirages[0])*10+2
plt.subplot(234)
h=hist(x,bins[0])
plt.title("Changement: tirages dans [2;12]")
xlim(int(x.min()),int(x.max()+1))
#DU entre 2 et 10, force les bins
x=rand(nbr_tirages[0])*10+2
plt.subplot(235)
h=hist(x,bins[0],range=(3,9))
plt.title("Changement: bins dans [3;9]")
xlim(int(x.min()),int(x.max()+1))

show()

#1.2 DISTRIBUTION NORMALE (DN)
#=============================

#PDF loi gaussienne--------------------
def PDFLG(esperance,ecart_type,valeurs):
	return 1./(ecart_type*sqrt(2*pi))*exp(-0.5*((valeurs-esperance)/ecart_type)**2)
#--------------------------------------

#Initialisation de la fenêtre graphique
plt.figure(2)
#Données initiales
nbr_tirages=1e6
bins=100

############## Loi Gaussienne REDUITE CENTREE (espérance=0,variance=écart type=1) #####################
#Tire 1 million de valeurs
x=randn(nbr_tirages)
print "Distribution gaussienne reduite centree (%d tirages)"%(nbr_tirages)
#Calcul la moyenne, la variance, et l'écart-type
esperance=sum(x)/size(x)
print "moyenne calcul:%f"%(esperance)
print "moyenne fonction mean:%f"%(mean(x))
variance=sum(x**2)/size(x)-esperance**2
ecart_type=sqrt(variance)
print "écart type calcul:%f"%(ecart_type)
print "écart type fonction std:%f"%(std(x))
#Trace les courbes
plt.subplot(211)
plt.title("DNormale centree %d tirages, %d bins"%(nbr_tirages,bins))
plt.text(5,30000,"""moyenne=0
ecart type=1""")
h=hist(x,bins)	#Histogramme
largeur_bins=h[1][1]-h[1][0]
plt.plot(h[1],nbr_tirages*largeur_bins*PDFLG(0,1,h[1]),'r',linewidth=2)	#Forme théorique
#Trace les courbes en log
plt.subplot(212)
plt.title("DGaussienne %d tirages, %d bins, en Log"%(nbr_tirages,bins))
h=hist(x,bins,log=True)	#Histogramme
largeur_bins=h[1][1]-h[1][0]
plt.plot(h[1],nbr_tirages*largeur_bins*PDFLG(0,1,h[1]),'r',linewidth=2)	#Forme théorique
#Erreur sur la moyenne et l'écart type
print "Erreur sur la moyenne: %f"%(mean(x)-0)
print "Erreur sur l'écart type: %f"%(std(x)-1)
print ""

#################"GAUSSIENNE (DG) moyenne=42 ET largeur=pi########################
x=randn(nbr_tirages)*pi+42
print "Distribution gaussienne moyenne=42 et largeur=pi (%d tirages)"%(nbr_tirages)
#Calcul de la moyenne, de la variance, et de l'écart-type
esperance=sum(x)/size(x)
print "moyenne calcul:%f"%(esperance)
print "moyenne fonction mean:%f"%(mean(x))
variance=sum(x**2)/size(x)-esperance**2
ecart_type=sqrt(variance)
print "écart type calcul:%f"%(ecart_type)
print "écart type fonction std:%f"%(std(x))
#Trace les courbes
plt.subplot(211)
plt.text(50,30000,"""moyenne=42
ecart type=pi""")
h=hist(x,bins)	#Histogramme
largeur_bins=h[1][1]-h[1][0]
plt.plot(h[1],nbr_tirages*largeur_bins*PDFLG(42,pi,h[1]),'r',linewidth=2)	#Forme théorique
#Trace les courbes en log
plt.subplot(212)
h=hist(x,bins,log=True)	#Histogramme
largeur_bins=h[1][1]-h[1][0]
plt.plot(h[1],nbr_tirages*largeur_bins*PDFLG(42,pi,h[1]),'r',linewidth=2)	#Forme théorique
print "Erreur sur la moyenne: %f"%(mean(x)-42)
print "Erreur sur l'écart type: %f"%(std(x)-pi)
print ""

show()

##############Résultats
print "On constate d'une part que les fonction mean et std sont très précises, et d'autre part que la moyenne et l'écart type des données simulées ne sont pas exactement égales aux valeurs attendues. De plus en changeant le nombre de tirages, on se rend compte que cette erreur est différente à chaque nouveau tirage, et plus le nombre tiré est grand, meilleur est la précision."
