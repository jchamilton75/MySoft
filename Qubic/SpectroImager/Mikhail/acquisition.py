# coding: utf-8
from __future__ import division

import healpy as hp
import numpy as np
from pyoperators import (
    BlockColumnOperator, BlockDiagonalOperator, BlockRowOperator,
    CompositionOperator, DiagonalOperator, I, IdentityOperator,
    MPIDistributionIdentityOperator, MPI, proxy_group, ReshapeOperator,
    rule_manager, pcg)
from pyoperators.utils.mpi import as_mpi
from pysimulators import Acquisition, FitsArray
from pysimulators.interfaces.healpy import (
    HealpixConvolutionGaussianOperator)

from qubic import QubicPlanckAcquisition
from qubic import QubicAcquisition, create_random_pointings
from qubic import QubicInstrument


__all__ = ['QubicMultibandAcquisition',
           'QubicMultibandPlanckAcquisition']

class QubicMultibandAcquisition(object):
    def __init__(self, multiinstrument, sampling, scene, block=None,
                 effective_duration=None,
                 photon_noise=True, max_nbytes=None,
                 nprocs_instrument=None, nprocs_sampling=None,
                 ripples=False, nripples=0,
                 weights=None):
        self.subacqs = [QubicAcquisition(multiinstrument[i], 
                                 sampling, scene=scene, block=block,
                                 effective_duration=effective_duration,
                                 photon_noise=photon_noise, max_nbytes=max_nbytes,
                                 nprocs_instrument=nprocs_instrument, 
                                 nprocs_sampling=nprocs_sampling) for i in range(len(multiinstrument))]
        for a in self[1:]:
            a.comm = self[0].comm
        self.scene = scene
        if weights == None:
            self.weights = np.ones(len(self)) / len(self)
        else:
            self.weights = weights

    def __getitem__(self, i):
        return self.subacqs[i]

    def __len__(self):
        return len(self.subacqs)

    def get_coverage(self):
        return np.array([self.subacqs[i].get_coverage() for i in range(len(self))])

    def get_coverage_mask(self, coverages, covlim=0.2):
        if coverages.shape[0] != len(self):
            raise ValueError('Use QubicMultibandAcquisition.get_coverage method to create input') 
        observed = [(coverages[i] > covlim * np.max(coverages[i])) for i in range(len(self))]
        obs = reduce(np.logical_and, tuple(observed[i] for i in range(len(self))))
        return obs

    def _get_average_instrument_acq(self):
        q0 = self[0].instrument
        nu_min = q0.filter.nu
        nu_max = self[-1].instrument.filter.nu
        nep = q0.detector.nep
        fknee = q0.detector.fknee
        fslope = q0.detector.fslope
        q = QubicInstrument(
            filter_nu=(nu_max + nu_min) / 2.,
            filter_relative_bandwidth=(nu_max - nu_min) / ((nu_max + nu_min) / 2.),
            detector_nep=nep, detector_fknee=fknee, detector_fslope=fslope)
        s_ = self[0].sampling
        nsamplings = self[0].comm.allreduce(len(s_))
        s = create_random_pointings([0., 0.], nsamplings, 10., period=s_.period)
        a = QubicAcquisition(
            q, s, self[0].scene, photon_noise=True,  
            effective_duration=self[0].effective_duration)
        return a

    def get_noise(self):
        a = self._get_average_instrument_acq()
        return a.get_noise()

    def get_operator(self):
        return BlockRowOperator([a.get_operator() * w for a, w in zip(self, self.weights)], 
                                 new_axisin=0)

    def get_invntt_operator(self):
        return self[0].get_invntt_operator()

    def get_observation(self, maps, convolution=True, noiseless=False):
        '''
        Return TOD of a multi band instrument
        maps -- array of maps of length equal to number of subbands
        '''
        maps_convolved = np.zeros((maps.shape))

        if convolution:
            for i in range(len(self)):
                C = self[i].get_convolution_peak_operator()
                maps_convolved[i] = C(maps[i])
            y = self.get_operator() * maps_convolved
        else:
            y = self.get_operator() * maps

        if not noiseless:
            y += self.get_noise()

        if convolution:
            return y, maps_convolved

        return y

    def tod2map(self, tod, tol=1e-5, maxiter=1000, verbose=True):
        H = self.get_operator()
        invntt = self.get_invntt_operator()

        A = H.T * invntt * H
        b = H.T * invntt * tod

        solution = pcg(A, b, disp=verbose, tol=tol, maxiter=maxiter)
        return solution['x']

class QubicMultibandPlanckAcquisition(QubicPlanckAcquisition):
    """
    The QubicMultibandPlanckAcquisition class, which combines the QubicMultiband and Planck
    acquisitions.

    """
    def __init__(self, qubic, planck, weights=None):
        """
        acq = QubicPlanckAcquisition(qubic_acquisition, planck_acquisition)

        Parameters
        ----------
        qubic_acquisition : QubicAcquisition
            The QUBIC acquisition.
        planck_acquisition : PlanckAcquisition
            The Planck acquisition.

        """
        if not isinstance(qubic, QubicMultibandAcquisition):
            raise TypeError('The first argument is not a QubicMultibandAcquisition.')
        if not isinstance(planck, PlanckAcquisition):
            raise TypeError('The second argument is not a PlanckAcquisition.')
        if qubic.scene is not planck.scene:
            raise ValueError('The Qubic and Planck scenes are different.')
        self.qubic = qubic
        self.planck = planck
        if weights == None:
            self.weights = np.ones(len(self)) / len(self)
        else:
            self.weights = weights


    def __len__(self):
        return len(self.qubic)

    def get_operator(self):
        """
        Return the fused observation as an operator.

        """
        p = self.planck
        H = []
        for q, w in zip(self.qubic, self.weights):
            H.append(QubicPlanckAcquisition(q, p).get_operator() * w)
        return BlockRowOperator(H, new_axisin=0)

    def get_observation(self, maps, noiseless=False, convolution=True):
        """
        Return the fused observation.

        Parameters
        ----------
        maps : numpy array of shape (nbands, npix, 3)
            True input multiband maps
        noiseless : boolean, optional
            If set, return a noiseless observation
        convolution : boolean, optional
            If set, 
        """
        obs_qubic_ = self.qubic.get_observation(
            maps, noiseless=noiseless,
            convolution=convolution)
        obs_qubic = obs_qubic_[0] if convolution else obs_qubic_
        obs_planck = self.planck.get_observation(noiseless=noiseless)
        obs = np.r_[obs_qubic.ravel(), obs_planck.ravel()]
        if convolution:
            return obs, obs_qubic_[1]
        return obs

    def tod2map(self, tod, tol=1e-5, maxiter=1000, verbose=True):
        p = self.planck
        H = []
        for q, w in zip(self.qubic, self.weights):
            H.append(QubicPlanckAcquisition(q, p).get_operator() * w)
        invntt = self.get_invntt_operator()

        A_columns = []
        for h1 in H:
            c = []
            for h2 in H:
                c.append(h2.T * invntt * h1)
            A_columns.append(BlockColumnOperator(c, axisout=0))
        A = BlockRowOperator(A_columns, axisin=0)

        b = (self.get_operator()).T * (invntt * tod)
        sh = b.shape
        if len(sh) == 3:
            b = b.reshape((sh[0] * sh[1], sh[2]))
        else:
            b = b.reshape((sh[0] * sh[1]))

        solution = pcg(A, b, disp=verbose, tol=tol, maxiter=maxiter)
        if len(sh) == 3:
            maps_recon = solution['x'].reshape(sh[0], sh[1], sh[2])
        else:
            maps_recon = solution['x'].reshape(sh[0], sh[1])
        return maps_recon
