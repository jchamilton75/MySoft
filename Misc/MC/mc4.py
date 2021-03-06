# -*- coding: utf-8 -*-
from pylab import *
from numpy import *
from matplotlib import *
import pylab as plt


#CALCUL DE PI PAR MC
#===================

#Initialisation de la fenetre graphique
plt.close("all")
plt.figure(1)
#Données initiales
pas=5
nbr_tirages_max=5
nbr_tirages=floor(10**linspace(1,nbr_tirages_max,pas))
nbr_mc=100
total_std_fixe=zeros(pas)
total_std_poisson=zeros(pas)
#Répétition de l'expérience en augmentant le nombre de tirages à chaque pas
for i in xrange(pas):
	calcul_pi_fixe=zeros(nbr_mc)
	calcul_pi_poisson=zeros(nbr_mc)
	#Pour chaque pas on effectue nbr_mc fois le calcul de pi selon chaque méthode
	for j in xrange(nbr_mc):
		#Calcul de pi en tirant nbr_tirages valeurs
		nn_fixe=nbr_tirages[i]
		x=rand(nn_fixe)*2-1#Carré de coté 2 centré en (0,0)
		y=rand(nn_fixe)*2-1
		cercle=where(sqrt(x**2+y**2)<=1)#Cercle de rayon 1 centré en (0,0)
		calcul_pi_fixe[j]=size(cercle)*1./nbr_tirages[i]*4
		#Calcul de pi en tirant un nombre Poissonnien de moyenne nbr_tirage
		nn_poisson=poisson(nbr_tirages[i])
		x=rand(nn_poisson)*2-1#Carré de coté 2 centré en (0,0)
		y=rand(nn_poisson)*2-1
		cercle=where(sqrt(x**2+y**2)<=1)#Cercle de rayon 1 centré en (0,0)
		calcul_pi_poisson[j]=size(cercle)*1./nbr_tirages[i]*4
	#On peut donc calculer pour chaque pas l'écart type sur la valeur calculée de pi
	total_std_fixe[i]=std(calcul_pi_fixe)
	total_std_poisson[i]=std(calcul_pi_poisson)
#Représentation graphique
title("Pecision sur PI")
xlabel('Nombre de tirages')
ylabel('Ecart type')
sf1=loglog(nbr_tirages,total_std_fixe,'ro')
sf2=plot(nbr_tirages,sqrt(pi*(4-pi)/nbr_tirages),'r')
sp1=loglog(nbr_tirages,total_std_poisson,'bo')
sp2=plot(nbr_tirages,2*sqrt(pi)/sqrt(nbr_tirages),'b')
plt.legend([sf1,sf2,sp1,sp2],["mc fixe","theorie fixe","mc poisson","theorie poisson"])
xlim(1,10**(nbr_tirages_max+1))

show()

print "On constate que la précision sur le calcul de pi augmente en fonction du nombre de tirages. De plus, il semble préférable de simuler en tirant à chaque fois n valeurs plutôt qu'un nombre poissonnien de moyenne n. (cf. graphes)"



