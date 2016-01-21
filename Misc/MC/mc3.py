# -*- coding: utf-8 -*-
from pylab import *
from numpy import *
from matplotlib import *
import pylab as plt

#3 PROBLEME DE MONTY HALL
#======================

#Données initiales
nbr_mc=10000
portes=array([0,1,2])
st1=zeros(nbr_mc)
st2=zeros(nbr_mc)
st3=zeros(nbr_mc)
#Répétition de l'expérience nbr_mc fois
for i in xrange(nbr_mc):
	#Préparation du jeu
	porte_kdo=int(rand(1)*3)
	porte_vide=portes[where(portes-porte_kdo!=0)]
	#Choix 1 du joueur
	choix1=int(rand(1)*3)
	non_choix=portes[where(portes-choix1!=0)]
	#Choix du commentateur
	if choix1==porte_kdo:
		porte_restante=array([choix1,non_choix[int(rand(1)*2)]])
	else:
		porte_restante=array([choix1,non_choix[where(non_choix-porte_kdo==0)]])
	#Stratégie 1: on garde la porte initiale
	choix2=choix1
	if choix2==porte_kdo:
		st1[i]=1
	#Stratégie 2: on change de porte
	choix2=porte_restante[where(porte_restante-choix1!=0)]
	if choix2==porte_kdo:
		st2[i]=1
	#stratégie 3: on choisie aléatoirement
	choix2=porte_restante[int(rand(1)*2)]
	if choix2==porte_kdo:
		st3[i]=1
#Calcul des probabilités pour chaque stratégie
proba_st1=sum(st1)/nbr_mc
proba_st2=sum(st2)/nbr_mc
proba_st3=sum(st3)/nbr_mc
print "proba st1 on garde la meme porte:%f"%(proba_st1)
print "proba st2 on change de porte:%f"%(proba_st2)
print "proba st3 on tire à pile ou face:%f"%(proba_st3)
print ""

##########Résultats
print"La meilleur stratégie est donc de changer de porte, car on a alors 2 chances sur 3 de trouver le cadeau"


