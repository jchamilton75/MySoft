# -*- coding: utf-8 -*-
from pylab import *
from numpy import *
from matplotlib import *
from scipy import integrate
import pylab as plt

#6 THEOREME CENTRAL-LIMITE
#=========================

#Le théorème central limite (parfois appelé théorème de la limite centrale) établit la convergence en loi d'une suite de variables aléatoires vers la loi normale. Intuitivement, ce résultat affirme que toute somme de variables aléatoires indépendantes et identiquement distribuées tend vers une variable aléatoire gaussienne. Le théorème central limite admet plusieurs généralisations qui donnent la convergence de sommes de variables aléatoires sous des hypothèses beaucoup plus faibles. Ces généralisations ne nécessitent pas des lois identiques mais font appel à des conditions qui assurent qu'aucune des variables n'exerce une influence significativement plus importante que les autres. Ainsi, ce théorème et ses généralisations offrent une explication à l'omniprésence de la loi normale dans la nature : de nombreux phénomènes sont dus à l'addition d'un grand nombre de petites perturbations aléatoires. (Wikipedia)

######################## Fonction PDF loi gaussienne ##################################
def PDFLG(esperance,ecart_type,valeurs):
	return 1./(ecart_type*sqrt(2*pi))*exp(-0.5*((valeurs-esperance)/ecart_type)**2)
#######################################################################################

############### Fonction Distribution cumulative ############################
def distrib_cumul(nbr_valeurs,pas_valeurs,valeurs,distrib_valeurs):
	y=[0]
	for i in xrange(1,nbr_valeurs):
		y=append(y,y[i-1]+distrib_valeurs[i]*pas_valeurs)
	return y
#############################################################################

#Données pour test avec distribution quelconque
min_u=-10.
max_u=20.
nbr_u=10000
u=linspace(min_u,max_u,nbr_u)
pas_u=(max_u-min_u)/nbr_u
#Intialise les fenetres graphiques
plt.close("all")
#Données initiales
nbr_tirages=1e6
nbr_mc=5

#Test du théorème avec des distribution uniformes identiques
#===========================================================
plt.figure(1)
plt.title("Theoreme central limite avec des distributions uniformes identiques")

x=zeros(nbr_tirages)
meanx=zeros(nbr_mc)
stdx=zeros(nbr_mc)
#Effectue nbr_mc=5 tirages de nbr_tirages=1e6 valeurs a chaque fois selon la meme distribution uniforme qui possede une esperance de meanX et une variance de stdX²
for i in xrange(nbr_mc):
	xi=rand(nbr_tirages)*10+5
	h=hist(xi,100,alpha=0.1,label="Tirage x%d"%(i))
	x=x+xi
	meanx[i]=mean(xi)
	stdx[i]=std(xi)
#Moyenne des tirages et calcul de la variable Z	
X=x/nbr_mc
meanX=mean(meanx)
stdX=mean(stdx)
Z=sqrt(nbr_mc)/stdX*(X-meanX)
#Trace les courbes
h=hist(x,100,alpha=0.6,label="Moyenne des xi")
h=hist(Z,100,alpha=0.5,label="Histogramme de Z")
largeur_bins=h[1][1]-h[1][0]
plt.plot(h[1],nbr_tirages*largeur_bins*PDFLG(0,1,h[1]),linewidth=2,label="Loi normale theorique de Z")
plt.legend()
show()

#Test du théorème avec des distributions identiques composées de 2 gaussienne et d'1 lorentzienne
#================================================================================================
plt.figure(2)
plt.title("Theoreme central limite avec des distributions identiques composees de 2 Gaussiennes et 1 Lorentzienne")
x=zeros(nbr_tirages)
meanx=zeros(nbr_mc)
stdx=zeros(nbr_mc)
for i in xrange(nbr_mc):
	#Forme analytique de la PDF: somme de 2 Gaussiennes + 1 Lorentzienne
	a0,m0,s0,a1,m1,s1,a2,m2,s2=1,-5,0.2,1./2,-2,1./2,3./2,5,1
	PDF=a0*exp(-(u-m0)**2/(2*s0**2))+a1*exp(-(u-m1)**2/(2*s1**2))+a2/(1+(u-m2)**2/s2**2)
	PDFanalytique=PDF/integrate.trapz(PDF,x=u)
	#Distribution cumulative
	Fcumul=distrib_cumul(nbr_u,pas_u,u,PDFanalytique)
	#Génération numérique suivant la PDF
	tirages_unif=rand(nbr_tirages)
	xi=interp(tirages_unif,Fcumul,u)
	h=hist(xi,100,alpha=0.1,label="Tirage x%d"%(i))
	x=x+xi
	meanx[i]=mean(xi)
	stdx[i]=std(xi)
X=x/nbr_mc
meanX=mean(meanx)
stdX=mean(stdx)
Z=sqrt(nbr_mc)/stdX*(X-meanX)
#Courbes
h=hist(x,100,alpha=0.6,label="Moyenne des xi")
h=hist(Z,100,alpha=0.5,label="Histogramme de Z")
largeur_bins=h[1][1]-h[1][0]
plt.plot(h[1],nbr_tirages*largeur_bins*PDFLG(0,1,h[1]),linewidth=2,label="Loi normale theorique de Z")
plt.legend()
show()

#Test du théorème avec des distribution uniformes différentes
#============================================================
plt.figure(3)
plt.title("Theoreme central limite avec des distributions uniformes differentes")
x=zeros(nbr_tirages)
meanx=zeros(nbr_mc)
stdx=zeros(nbr_mc)
for i in xrange(nbr_mc):
	xi=rand(nbr_tirages)*rand(1)*10+2*i
	h=hist(xi,100,alpha=0.1,label="Tirage x%d"%(i))
	x=x+xi
	meanx[i]=mean(xi)
	stdx[i]=std(xi)
X=x/nbr_mc
meanX=mean(meanx)
stdX=mean(stdx)
Z=sqrt(nbr_mc)/stdX*(X-meanX)
#Courbes
h=hist(x,100,alpha=0.6,label="Moyenne des xi")
h=hist(Z,100,alpha=0.5,label="Histogramme de Z")
largeur_bins=h[1][1]-h[1][0]
plt.plot(h[1],nbr_tirages*largeur_bins*PDFLG(0,1,h[1]),linewidth=2,label="Loi normale theorique de Z")
plt.legend()
show()

#Test du théorème avec des distribution uniformes différentes et une distribution composée de 2 gaussienne et d'1 lorentzienne
#================================================================================================================
plt.figure(4)
plt.title("Theoreme central limite: distributions uniformes differentes + distribution 2 Gaussiennes et 1 Lorentzienne")
x=zeros(nbr_tirages)
meanx=zeros(nbr_mc)
stdx=zeros(nbr_mc)
for i in xrange(nbr_mc):
	if i==0:
		#Forme analytique de la PDF: somme de 2 Gaussiennes + 1 Lorentzienne
		a0,m0,s0,a1,m1,s1,a2,m2,s2=1,-5,0.2,1./2,-2,1./2,3./2,5,1
		PDF=a0*exp(-(u-m0)**2/(2*s0**2))+a1*exp(-(u-m1)**2/(2*s1**2))+a2/(1+(u-m2)**2/s2**2)
		PDFanalytique=PDF/integrate.trapz(PDF,x=u)
		#Distribution cumulative
		Fcumul=distrib_cumul(nbr_u,pas_u,u,PDFanalytique)
		#Génération numérique suivant la PDF
		tirages_unif=rand(nbr_tirages)
		xi=interp(tirages_unif,Fcumul,u)
		h=hist(xi,100,alpha=0.1,label="Tirage x%d"%(i))
		x=x+xi
		meanx[i]=mean(xi)
		stdx[i]=std(xi)
	else:
		xi=rand(nbr_tirages)+2*i
		h=hist(xi,100,alpha=0.1,label="Tirage x%d"%(i))
		x=x+xi
		meanx[i]=mean(xi)
		stdx[i]=std(xi)
X=x/nbr_mc
meanX=mean(meanx)
stdX=mean(stdx)
Z=sqrt(nbr_mc)/stdX*(X-meanX)
#Courbes
h=hist(x,100,alpha=0.6,label="Moyenne des xi")
h=hist(Z,100,alpha=0.5,label="Histogramme de Z")
largeur_bins=h[1][1]-h[1][0]
plt.plot(h[1],nbr_tirages*largeur_bins*PDFLG(0,1,h[1]),linewidth=2,label="Loi normale theorique de Z")
plt.legend()
show()

#Résultats
#======================================================================
print "On remarque que des lors que les distributions de chaque tirage ne sont pas identiques, la variables Z formée des sommes des distributions ne converge plus vers une loi normale réduite centrée."


