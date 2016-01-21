from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import time
from pyoperators import (
    DenseOperator, DiagonalOperator, pcg)
from pysimulators import (
    ProjectionOperator, SphericalEquatorial2GalacticOperator, FitsArray)
from qubic import QubicAcquisition, QubicInstrument, create_sweeping_pointings

def pack_projection_inplace(projection, mask):
    n = np.sum(~mask)
    new_index = np.zeros(mask.shape, int)
    new_index[~mask] = np.arange(n)
    matrix = projection.matrix
    matrix.shape = (matrix.shape[0], 3 * n)
    matrix.data.index = new_index[matrix.data.index]
    return ProjectionOperator(matrix, shapein=(n, 3),
                              shapeout=projection.shapeout)


def pack(v, mask):
    return v[~mask, :]


def unpack(v, mask):
    out = np.empty(mask.shape + (3,)) + np.nan
    out[~mask, :] = v
    return out

def map2tod(maps,pointing,instrument_in,detector_list=False,kmax=2):
    #### Detectors
    mask_packed = np.ones(len(instrument_in.detector.packed), bool)
    if detector_list:
        mask_packed[detector_list] = False
        mask_unpacked = instrument_in.unpack(mask_packed)
        instrument = QubicInstrument('monochromatic', removed=mask_unpacked,nside=instrument_in.sky.nside)
    else:
        instrument = instrument_in
        
    #### Observations
    obs = QubicAcquisition(instrument, pointing)
    #C = obs.get_convolution_peak_operator()
    #convmaps=np.transpose(np.array([C(maps[:,0]),C(maps[:,1]),C(maps[:,2])]))
    projection = obs.get_projection_peak_operator(kmax=kmax)
    coverage = projection.pT1()
    mask = coverage == 0
    projection = pack_projection_inplace(projection, mask)
    hwp = obs.get_hwp_operator()
    polgrid = DenseOperator([[0.5, 0.5, 0],
                             [0.5,-0.5, 0]])
    #H = polgrid * hwp * projection * C
    H = polgrid * hwp * projection
    x1 = pack(maps, mask)
    y = H(x1)
    #return y,convmaps
    return y

def tod2map(tod,pointing,instrument_in,detector_list=False,disp=True,kmax=2,displaytime=False):
    t0=time.time()
    #### Detectors
    mask_packed = np.ones(len(instrument_in.detector.packed), bool)
    if detector_list:
        mask_packed[detector_list] = False
        mask_unpacked = instrument_in.unpack(mask_packed)
        instrument = QubicInstrument('monochromatic', removed=mask_unpacked,nside=instrument_in.sky.nside)
    else:
        instrument = instrument_in
        
    #### Observations
    obs = QubicAcquisition(instrument, pointing)
    projection = obs.get_projection_peak_operator(kmax=kmax)
    coverage = projection.pT1()
    mask = coverage == 0
    projection = pack_projection_inplace(projection, mask)
    hwp = obs.get_hwp_operator()
    polgrid = DenseOperator([[0.5, 0.5, 0],
                             [0.5,-0.5, 0]])
    H = polgrid * hwp * projection
    preconditioner = DiagonalOperator(1/coverage[~mask], broadcast='rightward')
    solution = pcg(H.T * H, H.T(tod), M=preconditioner, disp=disp, tol=1e-3)
    output_map = unpack(solution['x'], mask)
    t1=time.time()
    if displaytime: print(' Map done in {0:.4f} seconds'.format(t1-t0))
    return output_map,coverage


def tod2map_perdet(tod,pointing,instrument,detector_list=False,disp=True,kmax=2,displaytime=False):
    if detector_list:
        detlist=detector_list
    else:
        detlist=np.arange(len(instrument.detector.packed))
        
    nside=instrument.sky.nside
    output_maps=np.zeros((12*nside**2,3))
    coverage=np.zeros(12*nside**2)
    for i in np.arange(len(detlist)):
        print(i)
        output_maps_i,coverage_i=tod2map(tod[[detlist[i]],:,:],pointing,instrument,detector_list=[detlist[i]],disp=disp,kmax=kmax,displaytime=displaytime)
        output_maps[:,0] += np.nan_to_num(output_maps_i[:,0]*coverage_i)
        output_maps[:,1] += np.nan_to_num(output_maps_i[:,1]*coverage_i)
        output_maps[:,2] += np.nan_to_num(output_maps_i[:,2]*coverage_i)
        coverage += np.nan_to_num(coverage_i)

    output_maps[:,0]=output_maps[:,0]/coverage
    output_maps[:,1]=output_maps[:,1]/coverage
    output_maps[:,2]=output_maps[:,2]/coverage
    return output_maps,coverage
    
    

def display(input, msg,center=False,lim=[200, 10, 10],reso=5,mask=None):
    for i, (kind, lim) in enumerate(zip('IQU', lim)):
        map = input[..., i].copy()
        if mask is not None: map[mask]=np.nan
        hp.gnomview(map, rot=center, reso=reso, xsize=400, min=-lim, max=lim,
                    title=msg + ' ' + kind, sub=(1, 3, i+1))




def rhoepsilon_from_maps(QUin,QUout,noiseout=None,goodpix=None):
    mapq=QUin[0]
    mapu=QUin[1]
    qprime=QUout[0]
    uprime=QUout[1]
    if noiseout:
        noiseQ=noiseout[0]
        noiseU=noiseout[1]
    else:
        noiseQ=np.ones(len(QUin[0]))
        noiseU=np.ones(len(QUin[1]))
    if goodpix is None:
        goodpix = mapq == mapq
    mapq=mapq[goodpix]
    mapu=mapu[goodpix]
    qprime=qprime[goodpix]
    uprime=uprime[goodpix]
    noiseQ=noiseQ[goodpix]
    noiseU=noiseU[goodpix]
    
    ### voire code mathematica leakage_QU_pointing.nb
    qu_sq2=np.sum(mapq*mapu/noiseQ**2)
    qu_su2=np.sum(mapq*mapu/noiseU**2)
    qpu_sq2=np.sum(qprime*mapu/noiseQ**2)
    qup_su2=np.sum(mapq*uprime/noiseU**2)
    q2_su2=np.sum(mapq**2/noiseU**2)
    u2_su2=np.sum(mapu**2/noiseU**2)
    u2_sq2=np.sum(mapu**2/noiseQ**2)
    q2_sq2=np.sum(mapq**2/noiseQ**2)
    qqp_sq2=np.sum(mapq*qprime/noiseQ**2)
    uup_su2=np.sum(mapu*uprime/noiseU**2)
    rhorec=((qu_sq2 - qu_su2)*(qpu_sq2 - qu_sq2 + qu_su2 - qup_su2) + (q2_su2 + u2_sq2)*(q2_sq2 - qqp_sq2 + u2_su2 - uup_su2))/((qu_sq2 - qu_su2)**2 - (q2_su2 + u2_sq2)*(q2_sq2 + u2_su2))
    epsilonrec=(-(qpu_sq2 - qup_su2)*(q2_sq2 + u2_su2) + (qu_sq2 - qu_su2)*(qqp_sq2 + uup_su2))/((qu_sq2 - qu_su2)**2 - (q2_su2 + u2_sq2)*(q2_sq2 + u2_su2))
    return rhorec,epsilonrec


def mixingmatrix_from_maps(IQUin,IQUout,noiseout=None,goodpix=None):
    ### order of the parameters: {rhoT, rhoQ, rhoU, epsilonTQ, epsilonTU, epsilonQU}
    mapi=IQUin[:,0]
    mapq=IQUin[:,1]
    mapu=IQUin[:,2]
    iprime=IQUout[:,0]
    qprime=IQUout[:,1]
    uprime=IQUout[:,2]
    if noiseout:
        noiseI=noiseout[:,0]
        noiseQ=noiseout[:,1]
        noiseU=noiseout[:,2]
    else:
        noiseI=np.ones(len(IQUin[:,0]))
        noiseQ=np.ones(len(IQUin[:,1]))
        noiseU=np.ones(len(IQUin[:,2]))
    if goodpix is None:
        goodpix = mapq == mapq
    mapi=mapi[goodpix]
    mapq=mapq[goodpix]
    mapu=mapu[goodpix]
    iprime=iprime[goodpix]
    qprime=qprime[goodpix]
    uprime=uprime[goodpix]
    noiseI=noiseI[goodpix]
    noiseQ=noiseQ[goodpix]
    noiseU=noiseU[goodpix]

    ### voire code mathematica leakage_QU_pointing.nb
    t2_st2=np.sum(mapi*mapi/noiseI**2)
    q2_st2=np.sum(mapq*mapq/noiseI**2)
    u2_st2=np.sum(mapu*mapu/noiseI**2)
    qt_st2=np.sum(mapq*mapi/noiseI**2)
    tu_st2=np.sum(mapu*mapi/noiseI**2)
    qu_st2=np.sum(mapu*mapq/noiseI**2)
    q2_sq2=np.sum(mapq**2/noiseQ**2)
    t2_sq2=np.sum(mapi**2/noiseQ**2)
    qt_sq2=np.sum(mapq*mapi/noiseQ**2)
    tu_sq2=np.sum(mapu*mapi/noiseQ**2)
    qu_sq2=np.sum(mapq*mapu/noiseQ**2)
    u2_sq2=np.sum(mapu**2/noiseQ**2)
    t2_su2=np.sum(mapi**2/noiseU**2)
    u2_su2=np.sum(mapu**2/noiseU**2)
    tu_su2=np.sum(mapi*mapu/noiseU**2)
    qt_su2=np.sum(mapi*mapq/noiseU**2)
    qu_su2=np.sum(mapq*mapu/noiseU**2)
    q2_su2=np.sum(mapq**2/noiseU**2)
    ttp_st2=np.sum(iprime*mapi/noiseI**2)
    qqp_sq2=np.sum(mapq*qprime/noiseQ**2)
    uup_su2=np.sum(mapu*uprime/noiseU**2)
    qpt_sq2=np.sum(qprime*mapi/noiseQ**2)
    qtp_sq2=np.sum(iprime*mapq/noiseQ**2)
    tpu_st2=np.sum(iprime*mapu/noiseI**2)
    tup_su2=np.sum(uprime*mapi/noiseU**2)
    qpu_sq2=np.sum(qprime*mapu/noiseQ**2)
    qup_su2=np.sum(mapq*uprime/noiseU**2)

    mat=[[t2_st2, 0, 0, qt_st2, tu_st2, 0],
         [0, q2_sq2, 0, -qt_sq2, 0, qu_sq2],
         [0, 0, u2_su2, 0, -tu_su2, -qu_su2],
         [qt_st2, -qt_sq2, 0, q2_st2 + t2_sq2, qu_st2, -tu_sq2],
         [tu_st2, 0, -tu_su2, qu_st2, t2_su2 + u2_st2, qt_su2],
         [0, qu_sq2, -qu_su2, -tu_sq2, qt_su2, q2_su2 + u2_sq2]]
    mat=np.array(mat)
    matinv=np.linalg.inv(mat)

    vec=[-t2_st2 + ttp_st2,
         -q2_sq2 + qqp_sq2,
         -u2_su2 + uup_su2,
         -qpt_sq2 + qt_sq2 - qt_st2 + qtp_sq2,
         tpu_st2 - tu_st2 + tu_su2 - tup_su2, 
         qpu_sq2 - qu_sq2 + qu_su2 - qup_su2]

    res=np.dot(matinv,vec)
    return res
    
