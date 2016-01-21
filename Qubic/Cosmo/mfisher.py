from numpy import *
from pylab import *

from svd import *

from outils import *
from correlation import *
from cosmology import *

def rapport(z_fg, z_bg1, z_bg2, wo=-1.0, wa=0.0):
    chi_fg  = comoving_distance(z_fg, wo = wo, wa = wa)
    chi_bg1 = comoving_distance(z_bg1, wo = wo, wa = wa)
    chi_bg2 = comoving_distance(z_bg2, wo = wo, wa = wa)

    return (chi_bg2 - chi_fg)/(chi_bg1 - chi_fg)*chi_bg1/chi_bg2


# redshift distribution of LSST galaxies
# from LSST Science Book, p. 73
# zo for i = 25
# for i = 25, density n_gal = 46 gal/arcmin^2
def distri_z(z, zo = 0.2985):
    p_z = 1/(2.*zo) * (z/zo)**2 * exp(-z/zo)
    return p_z

# density_gal = nb galaxies per arcmin^2
# surface_survey = 20000 deg^2 (LSST)
def binning_redshift(nb_bin, bin_min, bin_max, density_gal = 46, surface_survey = 20000, paires = True):
    nb_gal_tot = density_gal * surface_survey * 3600 # ca fait 3.31 10^9 galaxies au total
    n_integre = 100
    mean_z = []
    mean_chi = []
    p_z = []
    nb_gal_bin = []
    limites_bins = []
    # Contient les "piquets"
    limites_bins.append(bin_min)

    wo = -1.0
    wa = 0.0
    delta_wo = 0.05
    delta_wa = 0.05

    # boucle sur les bins en z
    for i in range(0, nb_bin):
        z_bin_min = bin_min + i * (bin_max - bin_min)/nb_bin
        z_bin_max = bin_min + (i+1) * (bin_max - bin_min)/nb_bin
        limites_bins.append(z_bin_max)

        # On calcule la fraction de galaxies dans chaque bin
        z = [(z_bin_min + j * (z_bin_max - z_bin_min)/(n_integre-1)) for j in range(0, n_integre-1)]
        p_z = [(distri_z(z[j])) for j in range(0, n_integre-1)]
        frac_gal = integre_trapeze(z, p_z)
        moyenne = (z_bin_min + z_bin_max)/2.
        #print '      MOYENNE = ', moyenne, z_bin_min, z_bin_max, frac_gal
        mean_z.append(moyenne)
        mean_chi.append(comoving_distance(moyenne))

        nb_gal = frac_gal * nb_gal_tot
        nb_gal_bin.append(nb_gal)


    z_fg = mean_z[0]
    nb_gal_fg = nb_gal_bin[0]

    # Nombre de combinaisons de paires bg1-bg2 = (nb_bin -2)*(nb_bin-1)/2
    if (paires):
        # on prend en compte toutes les paires de bin telles que z_bg2 > z_bg1
        nb_max = nb_bin-1
        nb_paires = (nb_bin-2)*(nb_bin-1)/2.
    else:
        nb_max = 2
        nb_paires = nb_bin-2

    z_bg1 = zeros([nb_paires], floating)
    z_bg2 = zeros([nb_paires], floating)
    nb_gal_bg1 = zeros([nb_paires], floating)
    nb_gal_bg2 = zeros([nb_paires], floating)
    rapp_wo = zeros([nb_paires], floating)
    rapp_wa = zeros([nb_paires], floating)

    nb_p = 0
    for i in range(1, nb_max): # bg1 
        for j in range(i+1, nb_bin): # bg2
            #print nb_p, nb_paires
            z_bg1[nb_p] = mean_z[i]
            nb_gal_bg1[nb_p] = nb_gal_bin[i]
            z_bg2[nb_p] = mean_z[j]
            nb_gal_bg2[nb_p] = nb_gal_bin[j]
            
            deriv_wo = (rapport(z_fg, z_bg1[nb_p], z_bg2[nb_p], wo-delta_wo, wa) - \
                rapport(z_fg, z_bg1[nb_p], z_bg2[nb_p], wo+delta_wo, wa))/(2*delta_wo)
            deriv_wa = (rapport(z_fg, z_bg1[nb_p], z_bg2[nb_p], wo, wa-delta_wa) - \
                rapport(z_fg, z_bg1[nb_p], z_bg2[nb_p], wo, wa+delta_wa))/(2*delta_wa)
            rapp_wo[nb_p] = deriv_wo
            rapp_wa[nb_p] = deriv_wa
            #rapp[nb_p] = rapport(z_fg, z_bg1[nb_p], z_bg2[nb_p])
            nb_p += 1

            #print "    BINNING = ", i, j, z_bg1, z_bg2

    #print "NB_GAL_FG = ", nb_gal_bin

    return(z_fg, nb_gal_fg, z_bg1, nb_gal_bg1, z_bg2, nb_gal_bg2, mean_z, rapp_wo, rapp_wa, limites_bins, nb_gal_bin)  

# Avec des intervalles en z contenant un meme nombre de galaxies
def binning_redshift_cte(nb_bin, bin_min, bin_max, density_gal = 46, surface_survey = 20000, paires = True):
    nb_gal_tot = density_gal * surface_survey * 3600 # ca fait 3.31 10^9 galaxies au total
    n_integre = 100
    mean_z = []
    mean_chi = []
    p_z = []
    nb_gal_bin = []
    limites_bins = []
    # Contient les "piquets"
    #limites_bins.append(bin_min)

    wo = -1.0
    wa = 0.0
    delta_wo = 0.05
    delta_wa = 0.05

    # fraction de galaxies par bin
    # faire des bins avec un nombre constant de galaxies

    # On vire les extremites de la distribution en z
    red_min = numpy.arange(n_integre)*(bin_min)/float(n_integre-1)
    p_z_min = distri_z(red_min)
    frac_min = integre_trapeze(red_min, p_z_min)

    red_max = bin_max + numpy.arange(n_integre)*(10. - bin_max)/float(n_integre-1)
    p_z_max = distri_z(red_max)
    frac_max = integre_trapeze(red_max, p_z_max)
    
    # fraction de galaxies dans chaque bin
    frac_gal = (1.-frac_min-frac_min)/float(nb_bin + 1) # le " +1 ", sinon, ca buggue, parfois...
    nb_gal = frac_gal * nb_gal_tot

    # on definit un intervalle d'integration
    nb_tot = nb_bin * n_integre
    intervalle = (bin_max - bin_min)/float(nb_tot)
    z = bin_min + numpy.arange(nb_tot) * (bin_max - bin_min)/float(nb_tot-1)
    p_z = distri_z(z)

    limites_bins.append(bin_min)
    nb = 0
    som = 0.
    numero_bin = 1
    for i in range(0, nb_tot):
        som += p_z[i] * intervalle
        nb += 1
        
        if (som > frac_gal):
            #print i, som, numero_bin, frac_gal, nb, frac_min, frac_max, z[i]+intervalle, bin_min, bin_max
            limites_bins.append(z[i]+intervalle)
            numero_bin += 1
            som = 0.
            nb = 0
    limites_bins.append(bin_max)

    print 'longueur = ', len(limites_bins), nb_bin, numero_bin
    
    #print 'LIMITES BINS = ', limites_bins

    # boucle sur les bins en z
    for i in range(0, nb_bin):
        moyenne = (limites_bins[i] + limites_bins[i+1])/2.
        #print '      MOYENNE = ', moyenne, z_bin_min, z_bin_max, frac_gal
        mean_z.append(moyenne)
        mean_chi.append(comoving_distance(moyenne))

        nb_gal_bin.append(nb_gal)
    

    z_fg = mean_z[0]
    nb_gal_fg = nb_gal

    # Nombre de combinaisons de paires bg1-bg2 = (nb_bin -2)*(nb_bin-1)/2
    if (paires):
        # on prend en compte toutes les paires de bin telles que z_bg2 > z_bg1
        nb_max = nb_bin-1
        nb_paires = (nb_bin-2)*(nb_bin-1)/2.
    else:
        nb_max = 2
        nb_paires = nb_bin-2

    z_bg1 = zeros([nb_paires], floating)
    z_bg2 = zeros([nb_paires], floating)
    nb_gal_bg1 = zeros([nb_paires], floating)
    nb_gal_bg2 = zeros([nb_paires], floating)
    rapp_wo = zeros([nb_paires], floating)
    rapp_wa = zeros([nb_paires], floating)

    nb_p = 0
    for i in range(1, nb_max): # bg1 
        for j in range(i+1, nb_bin): # bg2
            #print nb_p, nb_paires
            z_bg1[nb_p] = mean_z[i]
            nb_gal_bg1[nb_p] = nb_gal
            z_bg2[nb_p] = mean_z[j]
            nb_gal_bg2[nb_p] = nb_gal
            
            deriv_wo = (rapport(z_fg, z_bg1[nb_p], z_bg2[nb_p], wo-delta_wo, wa) - \
                rapport(z_fg, z_bg1[nb_p], z_bg2[nb_p], wo+delta_wo, wa))/(2*delta_wo)
            deriv_wa = (rapport(z_fg, z_bg1[nb_p], z_bg2[nb_p], wo, wa-delta_wa) - \
                rapport(z_fg, z_bg1[nb_p], z_bg2[nb_p], wo, wa+delta_wa))/(2*delta_wa)
            rapp_wo[nb_p] = deriv_wo
            rapp_wa[nb_p] = deriv_wa
            #rapp[nb_p] = rapport(z_fg, z_bg1[nb_p], z_bg2[nb_p])
            nb_p += 1

            #print "    BINNING = ", i, j, z_bg1, z_bg2

    #print "NB_GAL_FG = ", nb_gal_bin

    return(z_fg, nb_gal_fg, z_bg1, nb_gal_bg1, z_bg2, nb_gal_bg2, mean_z, rapp_wo, rapp_wa, limites_bins, nb_gal_bin)  



# Attention, theta_arcmin contient nb_bin+1 elements jusqu'a theta_max 
#(nb de piquets et non nombre d'intervalles)
def binning_theta(nb_qso, density_gal, nb_bin_theta = 15, theta_arcmin_min = 0.5, theta_arcmin_max=60.):
    # Creation bin logarithmiques en theta
    # Definition des bins en theta
    theta_arcmin = []
    # Intervalle de theta en arcmin
    theta_arcmin = [(log10(theta_arcmin_min)+(log10(theta_arcmin_max)-log10(theta_arcmin_min))* \
                         float(i)/float(nb_bin_theta)) for i in range(0, nb_bin_theta+1)]
    theta_arcmin = [(10**theta_arcmin[i]) for i in range(0,nb_bin_theta+1)]
    # On met theta en radians
    theta_radian = []
    theta_radian = [(theta_arcmin[i] / 60. * math.pi/180.) for i in range(0,nb_bin_theta+1)] 

    #print 'THETA_ARCMIN = ', theta_arcmin

    # Calcul de la surface de chaque bin en theta (arcmin^2)
    # aire_bin contient un element de moins que theta_arcmin
    aire_bin = []
    #aire_bin.append(math.pi*theta_arcmin[0]**2)
    for i in range(0,nb_bin_theta):
        aire_bin.append(math.pi*(theta_arcmin[i+1]**2-theta_arcmin[i]**2))

    #print 'AIRE_BIN = ', aire_bin

    # Nombre de galaxies d'avant-plan dans chaque bin
    nb_gal_bin = []
    nb_gal_bin = [(density_gal * aire_bin[i]) for i in range(0, nb_bin_theta)]

    #print 'NB_GAL_BIN = ', nb_gal_bin

    # Prefacteur
    # norm = facteur de normalisation, si besoin (N_Q * N_G par exemple), pour chaque bin
    norm = []
    norm = [(nb_qso * nb_gal_bin[i]) for i in range(0, nb_bin_theta)]

    # Erreur sur chaque bin
    sigma_bin = []
    sigma_bin = [(sqrt(nb_qso * nb_gal_bin[i])) for i in range(0, nb_bin_theta)]

    return(theta_radian, sigma_bin, norm)


# bin = tableau pour la variable binnee; contenant un element de plus que le nombre de bin 
#        (ie les piquets et non les intervalles)
# sigmabin = tableau d'erreurs sur chaque bin
# ninbin = tableau de dimension celle de bin, contenant que des 1 ???
# params = tableau de strings, pour les deux parametres marginalises
# devparam = tableau de deux valeurs de variation des parametres
# norm = facteur de normalisation, si besoin (N_Q * N_G par exemple), pour chaque bin
def mfisher_correlation(bin, sigmabin, ninbin, params, norm, z_gal, z_qso, h = 0.7, omega_m =0.3, omega_l = 0.7, omega_b = 0.04, wo = -1., wa = 0.0, sigma8 = 0.8, n_s = 1.0, nl = 1):

    #print "BIN = ", bin

    # nombre de valeurs testees pour chaque parametre
    nb_val = 3

    # intervalle de valeurs testees
    devparam = [0.1, 0.1]

    parametres = []

    # Initialisation de la matrice de Fisher
    fmat = zeros([len(params),len(params)], floating)
    
    for i in range(0, len(params)):
        if   params[i] == 'omega_m' : parametres.append(omega_m)
        elif params[i] == 'omega_l' : parametres.append(omega_l)
        elif params[i] == 'sigma8'  : parametres.append(sigma8)
        elif params[i] == 'wo'      : parametres.append(wo)
        elif params[i] == 'wa'      : parametres.append(wa)

    parametres = array(parametres)
    devparam = array(devparam)

    #print 'PARAMETRES = ', parametres, devparam
    
    # On boucle sur les bins (ie dimension de bin - 1)
    for b in range(0,len(bin)-1):
        fb = zeros([len(params), nb_val],floating)
        par_val = zeros([len(params),nb_val], floating)
        #print 'NB_VAL_0 = ', nb_val
        # On boucle sur les parametres
        for p in range(0, len(params)):
            # On boucle sur le nombre de valeurs testees par parametre
            for t in range(0, nb_val):
                # matrice contenant toutes les valeurs testees de tous les parametres oscultes
                par_val[p,t] = parametres[p] - devparam[p] + 2. * devparam[p] * float(t) / float(nb_val-1)
                #print 'par_val = ',p,t, par_val[p,t]

                if   params[p] == 'omega_m' : omega_m = par_val[p][t]
                elif params[p] == 'omega_l' : omega_l = par_val[p][t]
                elif params[p] == 'sigma8'  : sigma8  = par_val[p][t]
                elif params[p] == 'wo'      : wo      = par_val[p][t]
                elif params[p] == 'wa'      : wa      = par_val[p][t]

                #print 'par corr = ', p, t, bin[b], z_gal, z_qso, n_s, omega_m, omega_l, wo, wa, h, omega_b, sigma8, nl

                corr = correlation_brute(bin[b], z_gal, z_qso, n_s=n_s, omega_m=omega_m, omega_l=omega_l, wo=wo, wa=wa, h=h, omega_b=omega_b, sigma8=sigma8, nl=nl)      
                #print 'corr = ', p, t, corr, norm[b], corr*norm[b], bin[b]
                fb[p,t] = corr 

        # Calcul de la derivee par rapport aux parametres
        #print 'PAR_VAL = ', b, par_val
        #print 'FB = ', b, fb
        for p in range(0, len(params)):
            for t in range (0, nb_val):
                fb[p,t] = fb[p,t] * norm[b]

        deriv = zeros([len(params)], floating)
        for p in range(0, len(params)):
            derivee = []
            derivee = derive(par_val[p,:], fb[p,:])
            #print "Derivee = ", b, p, derivee 
            derivee = array(derivee)
            # En interpolant, on recupere la derivee aux points 
            # donnes par les parametres
            par = zeros([1], floating)
            par[0] = parametres[p]
            #print 'PAR = ', par
            # interpol_tab prend un tableau en argument, 
            # il faut transformer le scalaire en tableau
            deriv[p] = interpol_tab(par_val[p,:], derivee, par)[0]

            #print 'DERIV P = ', p ,par_val[p,:], fb[p,:], deriv[p], par

        #print 'DERIV2 = ', deriv
        ninbin = array(ninbin)
        sigmabin = array(sigmabin)

        # Calcul matrice de Fisher
        for p1 in range(0, len(params)):
            for p2 in range(0, len(params)):
                #print "P1, p2 = ", b, p1, p2
                #print fmat[p1,p2]
                #print ninbin[b]
                #print "derivee, p2, p1 = ", deriv[p2], deriv[p1]
                #print "sigma bin[", b, "] = ", sigmabin[b]
                #print "   ****** "
                fmat[p1,p2] = fmat[p1,p2] + ninbin[b]*deriv[p2]*deriv[p1]*1./sigmabin[b]**2
                #fmat[p1,p2] = fmat[p1,p2] + ninbin[b]*deriv[p2]*deriv[p1]*1.*sigmabin[b]

    return fmat


# nb_bin = nombre de bin (en z)
# limites_bins = valeur des piquets en z ; nb_bin+1 valeurs
# z_mean = redshift moyen de chaque bin
# nb_gal_bin = nombre de galaxies dans chaque bin
# params = tableau de strings, pour les deux parametres marginalises
def mfisher_rapport_corr_bg2_sur_bg1(nb_bin, z_fg, nb_gal_fg, z_bg1, nb_gal_bg1, z_bg2, nb_gal_bg2, params, h = 0.7, omega_m =0.3, omega_l = 0.7, omega_b = 0.04, wo = -1., wa = 0.0, sigma8 = 0.8, n_s = 1.0, nl = 1, sigma_eq_1 = False):

    #print "LENGTH = ", len(z_bg1), len(z_bg2), len(nb_gal_bg1), len(nb_gal_bg2)

    # nombre de valeurs testees pour chaque parametre
    nb_val = 3

    # intervalle de valeurs testees
    devparam = [0.1, 0.1]

    parametres = []

    # Initialisation de la matrice de Fisher
    fmat = zeros([len(params),len(params)], floating)
    
    for i in range(0, len(params)):
        if   params[i] == 'omega_m' : parametres.append(omega_m)
        elif params[i] == 'omega_l' : parametres.append(omega_l)
        elif params[i] == 'sigma8'  : parametres.append(sigma8)
        elif params[i] == 'wo'      : parametres.append(wo)
        elif params[i] == 'wa'      : parametres.append(wa)
        
    nb_bin = len(z_bg1)

    parametres = array(parametres)
    devparam = array(devparam)
    sigmabin = zeros(nb_bin)

    #print 'PARAMETRES = ', parametres, devparam
    
    for b in range(0,nb_bin):
        fb = zeros([len(params), nb_val], floating)
        par_val = zeros([len(params),nb_val], floating)
        #print 'NB_VAL_0 = ', nb_val

        # On boucle sur les parametres
        for p in range(0, len(params)):
            # On boucle sur le nombre de valeurs testees par parametre
            for t in range(0, nb_val):
                # matrice contenant toutes les valeurs testees de tous les parametres oscultes
                par_val[p,t] = parametres[p] - devparam[p] + 2. * devparam[p] * float(t) / float(nb_val-1)
                #print 'par_val = ',p,t, par_val[p,t]

                if   params[p] == 'omega_m' : omega_m = par_val[p][t]
                elif params[p] == 'omega_l' : omega_l = par_val[p][t]
                elif params[p] == 'sigma8'  : sigma8  = par_val[p][t]
                elif params[p] == 'wo'      : wo      = par_val[p][t]
                elif params[p] == 'wa'      : wa      = par_val[p][t]

                chi_fg  = comoving_distance(z_fg, Ho = 100.*h, omega_m = omega_m, omega_l = omega_l, wo = wo, wa = wa)
                chi_bg1 = comoving_distance(z_bg1[b], Ho = 100.*h, omega_m = omega_m, omega_l = omega_l, wo = wo, wa = wa)
                chi_bg2 = comoving_distance(z_bg2[b], Ho = 100.*h, omega_m = omega_m, omega_l = omega_l, wo = wo, wa = wa)

                # Le nb de galaxies fg disparait dans le rapport
                # ainsi que l'integrale sur le P(k) de fg
                nb_paires1 = float(nb_gal_fg * nb_gal_bg1[b])
                nb_paires2 = float(nb_gal_fg * nb_gal_bg2[b])

                corr1 = (chi_bg1 - chi_fg)/(chi_bg1 * chi_fg)
                corr2 = (chi_bg2 - chi_fg)/(chi_bg2 * chi_fg)
                
                num1 = nb_paires1 * corr1
                num2 = nb_paires2 * corr2

                rapport = num2/num1

                # Calcul de l'erreur sur le rapport des correlations (sans covariance)

                var_corr2 = nb_paires2
                var_corr1 = nb_paires1
                
                if (not sigma_eq_1):
                    sigmabin[b] = abs(rapport) * sqrt(var_corr1/num1**2 + var_corr2/num2**2)
                else:
                    sigmabin[b] = 1.
                
                #print '            RAPPORT = ', b, p, t, corr1, corr2, rapport, var_corr1, var_corr2, sigmabin[b]

                fb[p,t] = rapport 

        # Calcul de la derivee par rapport aux parametres
        #print 'PAR_VAL = ', b, par_val
        #print 'FB = ', b, fb
        #for p in range(0, len(params)):
        #    for t in range (0, nb_val):
        #        fb[p,t] = fb[p,t] * norm[b]

        deriv = zeros([len(params)], floating)
        for p in range(0, len(params)):
            derivee = []
            derivee = derive(par_val[p,:], fb[p,:])
            #print "Derivee = ", b, p, derivee 
            derivee = array(derivee)
            # En interpolant, on recupere la derivee aux points 
            # donnes par les parametres
            par = zeros([1], floating)
            par[0] = parametres[p]
            #print 'PAR = ', par
            # interpol_tab prend un tableau en argument, 
            # il faut transformer le scalaire en tableau
            deriv[p] = interpol_tab(par_val[p,:], derivee, par)[0]

            #print '     DERIV P = ', p, par_val[p,:], fb[p,:], deriv[p], par

        #print 'DERIV2 = ', deriv
        #ninbin = array(ninbin)
        #sigmabin = array(sigmabin)

        # Calcul matrice de Fisher
        for p1 in range(0, len(params)):
            for p2 in range(0, len(params)):
                #print "P1, p2 = ", b, p1, p2
                #print fmat[p1,p2]
                #print ninbin[b]
                #print deriv[p2], deriv[p1]
                #print sigmabin[b]
                #print "   ****** "
                #fmat[p1,p2] = fmat[p1,p2] + ninbin[b]*deriv[p2]*deriv[p1]*1./sigmabin[b]**2
                fmat[p1,p2] = fmat[p1,p2] + deriv[p2]*deriv[p1]*1./sigmabin[b]**2

    #print "SIGMA_BIN = ", sigmabin

    return fmat, sigmabin



def ellipse(fisher_matrix, center_coord, params, nb_sigma = 1):

    # cf. Coe 2009
    alpha = 1.52 # 68.3 CL
    alpha_95 = 2.48 # 95.4 CL
    #alpha = 3.44 # 99.7 CL

    # Matrice covariante = inverse de la matrice de Fisher
    fisher = matrix(fisher_matrix)

    print "FISHER = ", fisher

    covariance = fisher.I

    #print (covariance[0,0]**2 - covariance[1,1]**2)**2/4., covariance[0,1]**2
    #print sqrt((covariance[0,0]**2 - covariance[1,1]**2)**2/4. + covariance[0,1]**2)
    #print (covariance[0,0]**2 + covariance[1,1]**2)

    #print "covariance[0,0] = ", covariance[0,0]
    #print "covariance[1,0] = ", covariance[1,0]
    #print "covariance[1,1] = ", covariance[1,1]
    #print "covariance[0,1] = ", covariance[0,1]

    gd_axe = sqrt((covariance[0,0] + covariance[1,1])/2. + sqrt((covariance[0,0] - covariance[1,1])**2/4. + covariance[0,1]**2))
    pt_axe = sqrt((covariance[0,0] + covariance[1,1])/2. - sqrt((covariance[0,0] - covariance[1,1])**2/4. + covariance[0,1]**2))
    theta = abs(0.5*arctan(2*covariance[0,1]/(covariance[0,0]-covariance[1,1])))

    rho = covariance[0,1]/(sqrt(covariance[0,0]*covariance[1,1]))
    sigmax = alpha * sqrt(covariance[1,1])
    sigmay = alpha * sqrt(covariance[0,0])

    if rho < 0. :
        theta = -theta

    print "Grand axe = ", gd_axe
    print "Petit axe = ", pt_axe
    print "Coef. Correlation = ", rho
    print "Theta = ", theta * 180./math.pi
    print "Sigma_x = ", alpha * sqrt(covariance[0,0])
    print "Sigma_y = ", alpha * sqrt(covariance[1,1])


    # Figure of Merit
    # Definition du DETF : fom = 1/sqrt(det(cov(wo,wa)))
    detcov = det(covariance)
    fom = 1./(sqrt(detcov))
    print 'Figure of Merit = ', fom, detcov, covariance[0,0] * covariance[1,1] - covariance[0,1]**2

    # Coe 2009
    fom2 = 1./(sqrt(covariance[0,0] * covariance[1,1] * (1. - rho**2)))

    print 'Figure of Merit 2 = ', fom2

    # DETF
    aire_ellipse_95 = alpha_95**2 * math.pi * gd_axe * pt_axe
    fom3 = 1./aire_ellipse_95

    print 'Figure of Merit DETF = ', fom3

    print 'COV = ', covariance

    # Trace des ellipses
    # Tableau de valeurs entre 0 et 2*pi
    angle = []
    nb_angle = 100
    angle = [(2.*math.pi/float(nb_angle-1)*float(i)) for i in range(0, nb_angle)]

    """
    # Singular Value Decomposition (SVD) = diagonalisation ??
    u,w,v = svd(covariance)
    

    print 'U = ', u
    print 'W = ', w
    print 'V = ', v

    uv = zeros([2, nb_angle], floating)
    for i in range(0, nb_angle):
        uv[0,i] = nb_sigma * w[0] * cos(angle[i])
        uv[1,i] = nb_sigma * w[1] * sin(angle[i])
    bla = zeros([2, nb_angle], floating)
    bla = matrix(bla)
    u = matrix(u)
    uv = matrix(uv)

    bla = u*uv

    x = []
    y = []
    x = [(bla[0,i]+center_coord[0]) for i in range(0, nb_angle)]
    y = [(bla[1,i]+center_coord[1]) for i in range(0, nb_angle)]

    #print 'BLA = ', bla

    #print 'x = ', x, ' y = ', y
    """

    # Methode graphique
    # Equation parametrique de l'ellipse
    # x = xo + a sin t
    # y = yo + b cos t
    xt = []
    yt = []
    xt = [(alpha * gd_axe * cos(angle[i])) for i in range(0,nb_angle)]
    yt = [(alpha * pt_axe * sin(angle[i])) for i in range(0,nb_angle)]

    # On multiplie le tout par une matrice de rotation d'angle theta
    #theta =abs(theta)
    print 'theta = ', theta
    xg = []
    yg = []
    xg = [(xt[i] * cos(theta) - yt[i] * sin(theta)) for i in range(0,nb_angle)]
    yg = [(xt[i] * sin(theta) + yt[i] * cos(theta)) for i in range(0,nb_angle)]
    xg = [(xg[i] + center_coord[0]) for i in range(0,nb_angle)]
    yg = [(yg[i] + center_coord[1]) for i in range(0,nb_angle)]

    if   params[0] == 'omega_m' : axe_x = r'$\Omega_m$'
    elif params[0] == 'omega_l' : axe_x = r'$\Omega_{\Lambda}$'
    elif params[0] == 'sigma8'  : axe_x = r'$\sigma_8$'
    elif params[0] == 'wo'      : axe_x = r'$w_0$'
    elif params[0] == 'wa'      : axe_x = r'$w_a$'

    if   params[1] == 'omega_m' : axe_y = r'$\Omega_m$'
    elif params[1] == 'omega_l' : axe_y = r'$\Omega_{\Lambda}$'
    elif params[1] == 'sigma8'  : axe_y = r'$\sigma_8$'
    elif params[1] == 'wo'      : axe_y = r'$w_0$'
    elif params[1] == 'wa'      : axe_y = r'$w_a$'

    theta_deg = theta * 180./math.pi
    return (xg, yg, axe_x, axe_y, sigmax, sigmay, gd_axe, pt_axe, theta_deg, fom3)

"""
# bin = tableau pour la variable binnee - theta, en radians
# sigmabin = tableau d'erreurs sur chaque bin
# ninbin = tableau de dimension celle de bin, contenant que des 1 ???
# params = tableau de strings, pour les deux parametres marginalises
# devparam = tableau de deux valeurs de variation des parametres
def mfisher_rapport_corr_fg1_sur_fg2(bin, densite_gal1, densite_gal2, nb_qso, ninbin, params, z_gal1, z_gal2, z_qso, h = 0.7, omega_m =0.3, omega_l = 0.7, omega_b = 0.04, wo = -1., wa = 0.0, sigma8 = 0.8, n_s = 1.0, nl = 1):

    #print "BIN = ", bin

    # Calcul de la surface de chaque bin en theta (arcmin^2)
    nb_bin = len(bin)-1
    bin_arcmin = []
    bin_arcmin = [(bin[i] * 60. / math.pi*180.) for i in range(0,nb_bin+1)] 
    aire_bin = []
    #aire_bin.append(math.pi*bin_arcmin[0]**2)
    for i in range(0,nb_bin):
        aire_bin.append(math.pi*(bin_arcmin[i+1]**2-bin_arcmin[i]**2))

    #print 'AIRE_BIN = ', aire_bin

    # Nombre de galaxies d'avant-plan dans chaque bin
    nb_gal_bin1 = []
    nb_gal_bin1 = [(densite_gal1 * aire_bin[i]) for i in range(0, nb_bin)]
    nb_gal_bin2 = []
    nb_gal_bin2 = [(densite_gal2 * aire_bin[i]) for i in range(0, nb_bin)]

    #print 'NB_GAL_BIN = ', nb_gal_bin1

    # Prefacteur
    norm1 = []
    norm1 = [(nb_qso * nb_gal_bin1[i]) for i in range(0, nb_bin)]
    norm2 = []
    norm2 = [(nb_qso * nb_gal_bin2[i]) for i in range(0, nb_bin)]

    # nombre de valeurs testees pour chaque parametre
    nb_val = 3

    # intervalle de valeurs testees
    devparam = [0.1, 0.1]

    parametres = []

    # Initialisation de la matrice de Fisher
    fmat = zeros([len(params),len(params)], floating)
    
    for i in range(0, len(params)):
        if   params[i] == 'omega_m' : parametres.append(omega_m)
        elif params[i] == 'omega_l' : parametres.append(omega_l)
        elif params[i] == 'sigma8'  : parametres.append(sigma8)
        elif params[i] == 'wo'      : parametres.append(wo)
        elif params[i] == 'wa'      : parametres.append(wa)

    parametres = array(parametres)
    devparam = array(devparam)
    sigmabin = zeros(nb_bin)

    #print 'PARAMETRES = ', parametres, devparam
    
    # On boucle sur les bins
    for b in range(0,len(bin)-1):
        fb = zeros([len(params), nb_val],floating)
        par_val = zeros([len(params),nb_val], floating)
        #print 'NB_VAL_0 = ', nb_val
        # On boucle sur les parametres
        for p in range(0, len(params)):
            # On boucle sur le nombre de valeurs testees par parametre
            for t in range(0, nb_val):
                # matrice contenant toutes les valeurs testees de tous les parametres oscultes
                par_val[p,t] = parametres[p] - devparam[p] + 2. * devparam[p] * float(t) / float(nb_val-1)
                #print 'par_val = ',p,t, par_val[p,t]

                if   params[p] == 'omega_m' : omega_m = par_val[p][t]
                elif params[p] == 'omega_l' : omega_l = par_val[p][t]
                elif params[p] == 'sigma8'  : sigma8  = par_val[p][t]
                elif params[p] == 'wo'      : wo      = par_val[p][t]
                elif params[p] == 'wa'      : wa      = par_val[p][t]

                #print 'par corr = ', p, t, bin[b], z_gal, z_qso, n_s, omega_m, omega_l, wo, wa, h, omega_b, sigma8, nl
                corr1 = norm1[b] * correlation_brute(bin[b], z_gal1, z_qso, n_s, omega_m, omega_l, wo, wa, h, omega_b, sigma8, nl) 
                corr2 = norm2[b] * correlation_brute(bin[b], z_gal2, z_qso, n_s, omega_m, omega_l, wo, wa, h, omega_b, sigma8, nl) 

                rapport = corr1/corr2

                # Calcul de l'erreur sur le rapport des correlations (sans covariance)
                var_corr1 = nb_gal_bin1[b] * nb_qso
                var_corr2 = nb_gal_bin2[b] * nb_qso
                sigmabin[b] = abs(rapport) * sqrt(var_corr1/corr1**2 + var_corr2/corr2**2)

                #rap = rapport_correlation_fg1_sur_fg2(bin[b], z_gal, z_qso1, z_qso2, n_s, omega_m, omega_l, wo, wa, h, omega_b, sigma8, nl)      
                #print 'corr = ', p, t, corr, norm[b], corr*norm[b], bin[b]
                fb[p,t] = rapport 

        # Calcul de la derivee par rapport aux parametres
        #print 'PAR_VAL = ', b, par_val
        #print 'FB = ', b, fb
        #for p in range(0, len(params)):
        #    for t in range (0, nb_val):
        #        fb[p,t] = fb[p,t] * norm[b]

        #print "SIGMABIN = ", bin_arcmin[b], nb_gal_bin1[b], sigmabin[b]

        deriv = zeros([len(params)], floating)
        for p in range(0, len(params)):
            derivee = []
            derivee = derive(par_val[p,:], fb[p,:])
            #print "Derivee = ", b, p, derivee 
            derivee = array(derivee)
            # En interpolant, on recupere la derivee aux points 
            # donnes par les parametres
            par = zeros([1], floating)
            par[0] = parametres[p]
            #print 'PAR = ', par
            # interpol_tab prend un tableau en argument, 
            # il faut transformer le scalaire en tableau
            deriv[p] = interpol_tab(par_val[p,:], derivee, par)[0]

            #print 'DERIV P = ', p ,par_val[p,:], fb[p,:], deriv[p], par

        #print 'DERIV2 = ', deriv
        ninbin = array(ninbin)
        #sigmabin = array(sigmabin)

        # Calcul matrice de Fisher
        for p1 in range(0, len(params)):
            for p2 in range(0, len(params)):
                #print "P1, p2 = ", b, p1, p2
                #print fmat[p1,p2]
                #print ninbin[b]
                #print deriv[p2], deriv[p1]
                #print sigmabin[b]
                #print "   ****** "
                fmat[p1,p2] = fmat[p1,p2] + ninbin[b]*deriv[p2]*deriv[p1]*1./sigmabin[b]**2

    #print "SIGMA_BIN = ", sigmabin

    return fmat
"""
