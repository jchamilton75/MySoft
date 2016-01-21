from __future__ import division

import matplotlib.pyplot as mp
import numexpr as ne
import numpy as np
from scipy.constants import c, pi

from pyoperators import MaskOperator
from pysimulators import create_fitsheader, DiscreteSurface
from qubic import QubicInstrument
from qubic.beams import GaussianBeam


### code here taken from script by pierre chanial
def ang2vec(theta_rad, phi_rad):
    sintheta = np.sin(theta_rad)
    return np.array([sintheta * np.cos(phi_rad),
                     sintheta * np.sin(phi_rad),
                     np.cos(theta_rad)])

def norm(x):
    return x / np.sqrt(np.sum(x**2, axis=-1))[..., None]





def get_fpimage(qubic,SOURCE_POWER=1., SOURCE_THETA=np.radians(0), SOURCE_PHI=np.radians(45),
                NU=150e9,DELTANU_NU=0.25,npts=512,HORN_OPEN=None,display=True,background=True,saturation=True):
    bg=0
    ### background power is 6 pW
    if background: bg=5.95e-12
        
    ## power in watts at the location of the entry horns
    ## that and phi in radians
    LAMBDA = c / NU              # [m]
    DELTANU = DELTANU_NU * NU    # [Hz]
    FOCAL_LENGTH = 0.3           # [m]

    PRIMARY_BEAM_FWHM = 14       # [degrees]
    SECONDARY_BEAM_FWHM = 14     # [degrees]

    DETECTOR_OFFSET = [0, 0]  # [m]
    SAT_POWER = 20e-12 # [W]

    NPOINT_DETECTOR_PLANE = npts**2 # number of detector plane sampling points
    DETECTOR_PLANE_LIMITS = (np.nanmin(qubic.detector.vertex[..., 0]),
                             np.nanmax(qubic.detector.vertex[..., 0]))  # [m]

    if HORN_OPEN is None: HORN_OPEN = ~qubic.horn.removed #all open

    ########
    # BEAMS
    ########
    primary_beam = GaussianBeam(PRIMARY_BEAM_FWHM)
    secondary_beam = GaussianBeam(SECONDARY_BEAM_FWHM, backward=True)

    ########
    # HORNS
    ########
    nhorn_open = np.sum(HORN_OPEN)
    horn_vec = np.column_stack([qubic.horn.center[HORN_OPEN],
                                np.zeros(nhorn_open)])

    #################
    # DETECTOR PLANE
    #################
    ndet_x = int(np.sqrt(NPOINT_DETECTOR_PLANE))
    a = np.r_[DETECTOR_PLANE_LIMITS[0]:DETECTOR_PLANE_LIMITS[1]:ndet_x*1j]
    det_x, det_y = np.meshgrid(a, a)
    det_spacing = (DETECTOR_PLANE_LIMITS[1] - DETECTOR_PLANE_LIMITS[0]) / ndet_x
    det_vec = np.dstack([det_x, det_y, np.zeros_like(det_x) - FOCAL_LENGTH])
    det_uvec = norm(det_vec)
    det_theta = np.arccos(det_uvec[..., 2])
    
    # solid angle of a detector plane pixel (gnomonic projection)
    central_pixel_sr = np.arctan(det_spacing / FOCAL_LENGTH)**2
    detector_plane_pixel_sr = -central_pixel_sr * np.cos(det_theta)**3

    
    ############
    # DETECTORS
    ############
    qubic.detector.vertex += DETECTOR_OFFSET
    header = create_fitsheader((ndet_x, ndet_x), cdelt=det_spacing, crval=(0, 0),
                               ctype=['X---CAR', 'Y---CAR'], cunit=['m', 'm'])
    detector_plane = DiscreteSurface.fromfits(header)
    integ = MaskOperator(qubic.detector.removed) * \
      detector_plane.get_integration_operator(qubic.detector.vertex)
      
    ###############
    # COMPUTATIONS
    ###############
    """ Phase and transmission from the switches to the focal plane. """
    transmission = np.sqrt(secondary_beam(det_theta) /
                           secondary_beam.sr *
                           detector_plane_pixel_sr)[..., None]
    const = 2j * pi / LAMBDA
    product = np.dot(det_uvec, horn_vec.T)
    modelA = ne.evaluate('transmission * exp(const * product)')

    """ Phase and transmission from the source to the switches. """
    source_uvec = ang2vec(SOURCE_THETA, SOURCE_PHI)
    source_E = np.sqrt(SOURCE_POWER) * np.sqrt(primary_beam(SOURCE_THETA))
    const = 2j * pi / LAMBDA
    product = np.dot(horn_vec, source_uvec)
    modelB = ne.evaluate('source_E * exp(const * product)')

    E = np.sum(modelA * modelB, axis=-1)

    I = np.abs(E)**2
    D = integ(I)

    ##########
    # Adding Background power
    ##########
    nobol = D == 0
    D[~nobol]+=bg

    ##########
    # NOISE
    ##########
    hplanck=6.62606957e-34 # [M2.kg/s]
    NEPbol = 4e-17 #[W/sqrt(hz)]
    N = np.sqrt(2*hplanck*NU*D + 2*D**2/DELTANU + NEPbol**2)
    N[nobol]=0
    
    ##########
    # SATURATION
    ##########
    if saturation:
        saturated = D >= SAT_POWER
        D[saturated]=0
        N[saturated]=0
    
    ##########
    # DISPLAY
    ##########
    if display:
        mp.figure()
        mp.imshow(np.log10(I), interpolation='nearest', origin='lower')
        mp.autoscale(False)
        qubic.detector.plot(transform=detector_plane.topixel)
        mp.figure()
        mp.imshow(D, interpolation='nearest')
        mp.gca().format_coord = lambda x,y: 'x={} y={} z={}'.format(x, y, D[x,y])
        mp.show()
        print('Given {} horns, we get {} W in the detector plane and {} W in the detec'
              'tors.'.format(nhorn_open, np.sum(I), np.sum(D)))

    return(D,N)



def get_fringe(qubic,horns, SOURCE_POWER=1., SOURCE_THETA=np.radians(0), SOURCE_PHI=np.radians(45),
               NU=150e9,DELTANU_NU=0.25,npts=512,display=True,background=True):
    horn_i=horns[0]
    horn_j=horns[1]
    all_open=~qubic.horn.removed
    all_open_i=all_open.copy()
    all_open_i[horn_i[0],horn_i[1]] = False
    all_open_j=all_open.copy()
    all_open_j[horn_j[0],horn_j[1]] = False
    all_open_ij=all_open.copy()
    all_open_ij[horn_i[0],horn_i[1]] = False
    all_open_ij[horn_j[0],horn_j[1]] = False

    Sall,Nall=get_fpimage(qubic,SOURCE_POWER=SOURCE_POWER,SOURCE_THETA=SOURCE_THETA, SOURCE_PHI=SOURCE_PHI,
                          NU=NU,DELTANU_NU=DELTANU_NU,npts=npts,display=display,HORN_OPEN=all_open,
                          background=background)
    S_i,N_i=get_fpimage(qubic,SOURCE_POWER=SOURCE_POWER,SOURCE_THETA=SOURCE_THETA, SOURCE_PHI=SOURCE_PHI,
                          NU=NU,DELTANU_NU=DELTANU_NU,npts=npts,display=display,HORN_OPEN=all_open_i,
                          background=background)
    S_j,N_j=get_fpimage(qubic,SOURCE_POWER=SOURCE_POWER,SOURCE_THETA=SOURCE_THETA, SOURCE_PHI=SOURCE_PHI,
                          NU=NU,DELTANU_NU=DELTANU_NU,npts=npts,display=display,HORN_OPEN=all_open_j,
                          background=background)
    S_ij,N_ij=get_fpimage(qubic,SOURCE_POWER=SOURCE_POWER,SOURCE_THETA=SOURCE_THETA, SOURCE_PHI=SOURCE_PHI,
                          NU=NU,DELTANU_NU=DELTANU_NU,npts=npts,display=display,HORN_OPEN=all_open_ij,
                          background=background)

    saturated = (Sall == 0) | (S_i == 0) | (S_j == 0) | (S_ij == 0)
    Sout=Sall+S_ij-S_i-S_j
    Nout=np.sqrt(Nall**2+N_ij**2+N_i**2+N_j**2)
    Sout[saturated]=0
    Nout[saturated]=0
    return Sout,Nout,[Sall,S_i,S_j,S_ij]

    
