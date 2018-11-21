import healpy as hp
import qubic
from pysimulators import FitsArray
from Tools import ReadMC as rmc
import glob

center = qubic.equ2gal(0., -57.)


####################################### Simulations au CC ###############################
nbptg = [4000, 10000, 16000]
nsub = [1,2,3,4]

rep = '/Users/hamilton/Qubic/SpectroImager/CCsims/'
all_fr_noguess = []
all_fc_noguess = []
all_fr_guess = []
all_fc_guess = []
for i in xrange(len(nbptg)):
    fr_noguess = []
    fc_noguess = []
    fr_guess = []
    fc_guess = []
    for j in xrange(len(nsub)):
        fr_noguess.append(glob.glob(rep+'*Ptg_{}_*_Guess_False_*_nf{}_maps_recon.fits'.format(nbptg[i], nsub[j])))
        fc_noguess.append(glob.glob(rep+'*Ptg_{}_*_Guess_False_*_nf{}_maps_convolved.fits'.format(nbptg[i], nsub[j])))
        fr_guess.append(glob.glob(rep+'*Ptg_{}_*_Guess_True_*_nf{}_maps_recon.fits'.format(nbptg[i], nsub[j])))
        fc_guess.append(glob.glob(rep+'*Ptg_{}_*_Guess_True_*_nf{}_maps_recon.fits'.format(nbptg[i], nsub[j])))
    all_fr_noguess.append(fr_noguess)
    all_fc_noguess.append(fc_noguess)
    all_fr_guess.append(fr_guess)
    all_fc_guess.append(fc_guess)
###########################################################################################



####################################### Simulations au NERSC ###############################
# nbptg = [4000, 40000]
# nsub = [1,2,3,4]

# rep = '/Users/hamilton/Qubic/SpectroImager/McCori/Duration20/'
# all_fr_nersc = []
# all_fc_nersc = []
# for i in xrange(len(nbptg)):
#     fr_nersc = []
#     fc_nersc = []
#     for j in xrange(len(nsub)):
#         fr_nersc.append(glob.glob(rep+'mpiQ_Nodes_*_Ptg_{}_Noutmax_*_Tol_1e-4_*_nf{}_maps_recon.fits'.format(nbptg[i], nsub[j])))
#         fc_nersc.append(glob.glob(rep+'mpiQ_Nodes_*_Ptg_{}_Noutmax_*_Tol_1e-4_*_nf{}_maps_convolved.fits'.format(nbptg[i], nsub[j])))
#     all_fr_nersc.append(fr_nersc)
#     all_fc_nersc.append(fc_nersc)
###########################################################################################






######### Analyse ####################
# files_r = [all_fr_nersc]
# files_c = [all_fc_nersc]

files_r = [all_fr_noguess, all_fr_guess, all_fc_noguess, all_fc_guess]
files_c = [all_fc_noguess, all_fc_guess, all_fc_noguess, all_fc_guess]


#### Frequencies
nus = []
for isub in xrange(len(nsub)):
    Nbfreq_in, nus_edge_in, nus_in, deltas_in, Delta_in, Nbbands_in = qubic.compute_freq(150., 0.25, nsub[isub])
    nus.append(nus_in)


results = []
for k in xrange(len(files_r)):
    fr = files_r[k]
    fc = files_c[k]
    all_mrec = []
    all_resid = []
    all_m_autos = []
    all_s_autos = []
    all_m_cross = []
    all_s_cross = []
    for i in xrange(len(nbptg)):
        mrec = []
        resid = []
        m_autos = []
        s_autos = []
        m_cross = []
        s_cross = []
        for j in xrange(len(nsub)):
            print('Ptg: {} / {} and Sub Freq {} / {}'.format(i+1, len(nbptg), j+1, len(nsub)))
            themrec, theresid, theseenmap, ell_binned, them_autos, thes_autos, them_cross, thes_cross= rmc.get_maps_cl(fr[i][j], fconv=fc[i][j])
            mrec.append(themrec)
            resid.append(theresid)
            m_autos.append(them_autos)
            s_autos.append(thes_autos)
            m_cross.append(them_cross)
            s_cross.append(thes_cross)
        all_mrec.append(mrec)
        all_resid.append(resid)
        all_m_autos.append(m_autos)
        all_s_autos.append(s_autos)
        all_m_cross.append(m_cross)
        all_s_cross.append(s_cross)
        all = [all_mrec, all_resid, all_m_autos, all_s_autos, all_m_cross, all_s_cross]
        results.append(all)




all_mrec, all_resid, all_m_autos, all_s_autos, all_m_cross, all_s_cross = results[0]

all_min, all_resid_in, all_m_autos_in, all_s_autos_in, all_m_cross_in, all_s_cross_in = results[2]


ll = np.arange(1,250)

inbsub=3
iptg=0
thespec = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']


isp=2
#figure()
clf()
for i in xrange(nsub[inbsub]):
    subplot(3,1,1)
    p=plot(ell_binned, all_m_autos[iptg][inbsub][i,isp,:], label = 'Sub-{}'.format(i+1))
    errorbar(ell_binned, all_m_autos[iptg][inbsub][i,isp,:], yerr=all_s_autos[iptg][inbsub][i,isp,:],fmt='o', color=p[0].get_color())
    #plot(ell_binned, all_m_autos_in[iptg][inbsub][i,isp,:], ':', color='k')
    plot(ll, rmc.dust_spectra(ll, nus[inbsub][i])[isp], color=p[0].get_color(), alpha=0.5)
    title('Autos '+thespec[isp]+' - ptg={}'.format(nbptg[iptg]))
    #ylim(-0.02,0.5)
    xlim(0,np.max(ll))
    legend()
    subplot(3,1,2)
    p=plot(ell_binned, all_m_cross[iptg][inbsub][i,isp,:])
    errorbar(ell_binned, all_m_cross[iptg][inbsub][i,isp,:], yerr=all_s_cross[iptg][inbsub][i,isp,:],fmt='o', color=p[0].get_color())
    #plot(ell_binned, all_m_cross_in[iptg][inbsub][i,isp,:], ':', color='k')
    plot(ll, rmc.dust_spectra(ll, nus[inbsub][i])[isp], color=p[0].get_color(), alpha=0.5)
    title('Cross '+thespec[isp]+' - ptg={}'.format(nbptg[iptg]))
    #ylim(-0.02,0.12)
    xlim(0,np.max(ll))
    subplot(3,1,3)
    p=plot(ell_binned, all_m_autos[iptg][inbsub][i,isp,:] / all_m_cross[iptg][inbsub][i,isp,:])
    #ylim(-0.02,0.12)
    xlim(0,np.max(ll))


# figure()
# rmc.plotmaps(all_mrec[iptg][inbsub][0,:,:,:])

figure()
rmc.plotmaps(all_resid[iptg][inbsub][0,:,:,:], rng=[5,0.5,0.5])





