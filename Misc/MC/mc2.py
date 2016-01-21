# -*- coding: utf-8 -*-
from pylab import *
from numpy import *
from matplotlib import *
import pylab as plt


#2 ETUDE LOI NORMALE PAR MC
#==========================
#De manière théorique, var(xbar)=var(x)/n, donc std(xbar)=std/sqrt(n)
#On trouve aussi std(std(x))=std/sqrt(2n)

#Initialisation de la fenetre graphique
plt.close("all")
plt.figure(1)
#Données initiales
pas=10
nbr_tirages_max=6
nbr_tirages=floor(10**linspace(1,nbr_tirages_max,pas))
nbr_mc=100
std_mean=zeros(pas)
std_std=zeros(pas)
#Répétition de l'expérience en augmentant le nombre de tirages à chaque pas
for i in xrange(pas):
	stock_mean=zeros(nbr_mc)
	stock_std=zeros(nbr_mc)
	#Pour chaque pas, on effectue nbr_mc fois nbr_tirages selon une loi normale réduite centrée dont on calcul la 		moyenne et l'écart type. A chaque fois qu'on effectue les tirages, on stocke les moyennes et les écart type.
	for j in xrange(nbr_mc):
		x=randn(nbr_tirages[i])
		stock_mean[j]=mean(x)
		stock_std[j]=std(x)
	#On peut donc calculer pour chaque pas l'écart type des moyennes et l'écart type des écart types de tous les 		tirages
	std_mean[i]=std(stock_mean)
	std_std[i]=std(stock_std)

############Représentation graphique
title("Precision sur la moyenne et sur l'ecart type de %d distributions"%(nbr_mc))
xlabel('Nombre de tirages')
ylabel('Ecart type')
m1=loglog(nbr_tirages,std_mean,'ro')
m2=plot(nbr_tirages,1./sqrt(nbr_tirages),'r')
s1=loglog(nbr_tirages,std_std,'bo')
s2=plot(nbr_tirages,1./sqrt(2*nbr_tirages),'b')
plt.legend([m1,m2,s1,s2],["moyenne mc","moyenne theorie","ecart type mc","ecart type theorie"])
xlim(1,10**(nbr_tirages_max+1))

show()

##############Résultats
print "On constate que la précision sur la moyenne et l'écart type d'une loi normale augmente en fonction du nombre de tirages (cf. graphes)"


