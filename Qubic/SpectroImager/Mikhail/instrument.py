# coding: utf-8
from __future__ import division

import healpy as hp
import numexpr as ne
import numpy as np
from pyoperators import (
    Cartesian2SphericalOperator, DenseBlockDiagonalOperator, DiagonalOperator,
    IdentityOperator, HomothetyOperator, ReshapeOperator, Rotation2dOperator,
    Rotation3dOperator, Spherical2CartesianOperator)
from pyoperators.utils import (
    operation_assignment, pool_threading, product, split)
from pyoperators.utils.ufuncs import abs2
from pysimulators import (
    BeamGaussian, ConvolutionTruncatedExponentialOperator, Instrument, Layout,
    ProjectionOperator)
from pysimulators.geometry import surface_simple_polygon
from pysimulators.interfaces.healpy import (
    Cartesian2HealpixOperator, HealpixConvolutionGaussianOperator)
from pysimulators.sparse import (
    FSRMatrix, FSRRotation2dMatrix, FSRRotation3dMatrix)
from scipy.constants import c, h, k
from qubic import QubicInstrument


__all__ = ['QubicMultibandInstrument']


class QubicMultibandInstrument():
    """
    The QubicMultibandInstrument class
    Represents the QUBIC multiband features 
    as an array of QubicInstrumet objects
    """
    def __init__(self, calibration=None, detector_fknee=0, detector_fslope=1,
                 detector_ncorr=10, detector_nep=4.7e-17, detector_ngrids=1,
                 detector_tau=0.01, 
                 filter_nus=[150e9], filter_relative_bandwidths=[0.25],
                 polarizer=True,
                 primary_beams=None, secondary_beams=None,
                 synthbeam_dtype=np.float32, synthbeam_fraction=0.99,
                 synthbeam_kmax=8,
                 synthbeam_peak150_fwhm=np.radians(0.39268176)):
        '''
        filter_nus -- base frequencies array
        filter_relative_bandwidths -- array of relative bandwidths 
        '''
        self.nsubbands = len(filter_nus)
        if self.nsubbands == 1:
            raise ValueError('Number of subbands must be > 1')
        self.subinstruments = [QubicInstrument(filter_nu=filter_nus[i],
                                    filter_relative_bandwidth=filter_relative_bandwidths[i],
                                    calibration=calibration, detector_fknee=detector_fknee,
                                    detector_ncorr=detector_ncorr, detector_nep=detector_nep,
                                    detector_ngrids=detector_ngrids, detector_tau=detector_tau,
                                    polarizer=polarizer, 
                                    primary_beam=primary_beams[i] if primary_beams is not None else None,
                                    secondary_beam=secondary_beams[i] if secondary_beams is not None else None,
                                    synthbeam_dtype=synthbeam_dtype, synthbeam_fraction=synthbeam_fraction,
                                    synthbeam_kmax=synthbeam_kmax, synthbeam_peak150_fwhm=synthbeam_peak150_fwhm)
                                for i in range(self.nsubbands)]

    def __getitem__(self, i):
        return self.subinstruments[i]

    def __len__(self):
        return len(self.subinstruments)
        
    def get_synthbeam(self, scene, idet=None, theta_max=45):
        sb = map(lambda i: i.get_synthbeam(scene, idet, theta_max),
                 self.subinstruments)
        sb = np.array(sb)
        sb = sb.sum(axis=0)
        return sb