from pyoperators import MPI, DiagonalOperator, PackOperator, pcg
from qubic import (
    QubicAcquisition, QubicInstrument,
    QubicScene, create_sweeping_pointings, equ2gal, create_random_pointings)
from MYacquisition import PlanckAcquisition, QubicPlanckAcquisition
from qubic.io import read_map, write_map
from qubic.data import PATH
import numpy as np
import healpy as hp

def statstr(vec):
    m=np.mean(vec)
    s=np.std(vec)
    return '{0:.4f} +/- {1:.4f}'.format(m,s)

def plotinst(inst, color='r'):
  for xyc, quad in zip(inst.detector.center, inst.detector.quadrant): 
      plot(xyc[0],xyc[1],'o', color=color)
  xlim(-0.06, 0.06)

# some display
def display(input, msg='', iplot=1, center=None, nlines=1, reso=5, lims=[50, 5, 5]):
    out = []
    for i, (kind, lim) in enumerate(zip('IQU', lims)):
        map = input[..., i]
        out += [hp.gnomview(map, rot=center, reso=reso, xsize=800, min=-lim,
                            max=lim, title=msg + ' ' + kind,
                            sub=(nlines, 3, iplot + i), return_projected_map=True)]
    return out


def get_qubic_map(instrument, sampling, scene, input_maps, withplanck=True, covlim=0.1, photon_noise=True, 
                    return_tod=False, effective_duration=None):
    acq = QubicAcquisition(instrument, sampling, scene, photon_noise=photon_noise, 
                            effective_duration=effective_duration)
    C = acq.get_convolution_peak_operator()
    coverage = acq.get_coverage()
    observed = coverage > covlim * np.max(coverage)
    acq_restricted = acq[:, :, observed]
    H = acq_restricted.get_operator()
    x0_convolved = C(input_maps)
    if not withplanck:
        pack = PackOperator(observed, broadcast='rightward')
        y_noiseless = H(pack(x0_convolved))
        noise = acq.get_noise()
        y = y_noiseless + noise
        invntt = acq.get_invntt_operator()
        A = H.T * invntt * H
        b = (H.T * invntt)(y)
        preconditioner = DiagonalOperator(1 / coverage[observed], broadcast='rightward')
        solution_qubic = pcg(A, b, M=preconditioner, disp=True, tol=1e-3, maxiter=1000)
        maps = pack.T(solution_qubic['x'])
        maps[~observed] = 0
        x0_convolved[~observed,:]=0    
        if return_tod:
            return(maps, x0_convolved, observed, y_noiseless, noise) 
        else:
            return(maps, x0_convolved, observed) 
    else:
        acq_planck = PlanckAcquisition(150, acq.scene, true_sky=x0_convolved, fix_seed=True)
        acq_fusion = QubicPlanckAcquisition(acq, acq_planck)
        map_planck_obs=acq_planck.get_observation()
        H = acq_fusion.get_operator()
        invntt = acq_fusion.get_invntt_operator()
        y = acq_fusion.get_observation()
        A = H.T * invntt * H
        b = H.T * invntt * y
        solution_fusion = pcg(A, b, disp=True, maxiter=1000, tol=1e-3)
        maps = solution_fusion['x']
        maps[~observed] = 0
        x0_convolved[~observed,:]=0    
        return(maps, x0_convolved, observed)  
    
def get_tod(instrument, sampling, scene, input_maps, withplanck=True, covlim=0.1, photon_noise=True): 
    acq = QubicAcquisition(instrument, sampling, scene, photon_noise=photon_noise)
    C = acq.get_convolution_peak_operator()
    coverage = acq.get_coverage()
    observed = coverage > covlim * np.max(coverage)
    acq_restricted = acq[:, :, observed]
    H = acq_restricted.get_operator()
    x0_convolved = C(input_maps)
    pack = PackOperator(observed, broadcast='rightward')
    y_noiseless = H(pack(x0_convolved))
    noise = acq.get_noise()
    y = y_noiseless + noise
    return (y_noiseless, noise, y)
